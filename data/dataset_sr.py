import os, random, numpy as np, glob
import torch, torch.utils.data as data
import utils.utils_image as util

class DatasetSR(data.Dataset):
    """
    SISR dataset.
    - Works as before when ONLY H/L are provided.
    - If you also pass opt['dataroot_Lmk'] (folder with .npy heatmaps), it will
      load K heatmaps per image and CONCAT them with LR along channel dim:
         L_plus = concat([LR_RGB(3c), HM_K(c)], dim=0)  -> (3+K)xHxW
      Use this for landmarks (K=5) -> in_chans=8.
    """

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.n_channels = opt.get('n_channels', 3) or 3
        self.sf = opt.get('scale', 4) or 4
        self.patch_size = opt.get('H_size', 96) or 96
        self.L_size = self.patch_size // self.sf
        self.phase = opt.get('phase', 'train')

        # ---- collect H/L paths ----
        self.paths_H = util.get_image_paths(opt['dataroot_H'])
        assert self.paths_H, 'Error: H path is empty.'
        self.paths_L = util.get_image_paths(opt.get('dataroot_L')) if opt.get('dataroot_L') else None

        # ---- optional landmarks (.npy files) ----
        self.use_lmk = bool(opt.get('dataroot_Lmk'))
        if self.use_lmk:
            lmk_root = opt['dataroot_Lmk']
            self.paths_Lmk = sorted(glob.glob(os.path.join(lmk_root, '*.npy')))
            assert self.paths_Lmk, f'Error: no .npy in {lmk_root}'
            # map by stem for quick lookup
            self.lmk_map = {os.path.splitext(os.path.basename(p))[0]: p for p in self.paths_Lmk}

        # ---- build matched pairs by filename stem ----
        self.pairs = []
        if self.paths_L:
            L_map = {os.path.splitext(os.path.basename(p))[0]: p for p in self.paths_L}
            missing = 0
            for hp in self.paths_H:
                stem = os.path.splitext(os.path.basename(hp))[0]
                lp = L_map.get(stem)
                if lp is None:
                    missing += 1
                    continue
                if self.use_lmk and stem not in self.lmk_map:
                    # skip if no landmark for this sample
                    missing += 1
                    continue
                self.pairs.append((hp, lp))
            if not self.pairs:
                raise RuntimeError('No matched H/L (and Lmk) pairs were found. Check filenames.')
            if missing > 0:
                print(f'[DatasetSR] Warning: {missing} items had no matching pair and were skipped.')
        else:
            # LR will be synthesized on-the-fly; landmark mode requires explicit LR
            if self.use_lmk:
                raise ValueError('Landmarks require explicit LR images (dataroot_L).')
            self.pairs = [(hp, None) for hp in self.paths_H]

        if self.patch_size % self.sf != 0:
            raise ValueError(f'H_size ({self.patch_size}) must be divisible by scale ({self.sf}).')

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        H_path, L_path = self.pairs[index]

        # ----- HR -----
        img_H = util.uint2single(util.imread_uint(H_path, self.n_channels))
        img_H = util.modcrop(img_H, self.sf)

        # ----- LR -----
        if L_path is not None:
            img_L = util.uint2single(util.imread_uint(L_path, self.n_channels))
        else:
            img_L = util.imresize_np(img_H, 1 / self.sf, True)

        # ----- Landmarks (.npy, KxHlrxWlr) -----
        hm = None
        if self.use_lmk:
            stem = os.path.splitext(os.path.basename(L_path))[0]
            lmk_path = self.lmk_map.get(stem)
            if lmk_path is None:
                raise FileNotFoundError(f'No landmark .npy for {stem}')
            hm = np.load(lmk_path).astype(np.float32)  # (K, Hlr, Wlr)
            # basic sanity: heatmap must match LR spatial size
            if hm.shape[1] != img_L.shape[0] or hm.shape[2] != img_L.shape[1]:
                raise ValueError(
                    f'Heatmap size {hm.shape[1:]} != LR size {img_L.shape[:2]} for {stem}. '
                    'Regenerate heatmaps at LR resolution.'
                )

        # ----- training patches & augmentation -----
        if self.phase == 'train':
            H_lr, W_lr, _ = img_L.shape
            rnd_h = random.randint(0, max(0, H_lr - self.L_size))
            rnd_w = random.randint(0, max(0, W_lr - self.L_size))
            # crop LR
            img_L = img_L[rnd_h:rnd_h + self.L_size, rnd_w:rnd_w + self.L_size, :]
            # corresponding HR crop
            rnd_h_H, rnd_w_H = int(rnd_h * self.sf), int(rnd_w * self.sf)
            img_H = img_H[rnd_h_H:rnd_h_H + self.patch_size, rnd_w_H:rnd_w_H + self.patch_size, :]

            # crop heatmaps the same way
            if hm is not None:
                hm = hm[:, rnd_h:rnd_h + self.L_size, rnd_w:rnd_w + self.L_size]

            # augmentation (apply same transform to all)
            mode = random.randint(0, 7)
            img_L = util.augment_img(img_L, mode=mode)
            img_H = util.augment_img(img_H, mode=mode)
            if hm is not None:
                chs = []
                for k in range(hm.shape[0]):
                    ch = hm[k][..., None]
                    ch = util.augment_img(ch, mode=mode)
                    chs.append(ch[..., 0])
                hm = np.stack(chs, axis=0)

        # ----- to tensors -----
        L_tensor = util.single2tensor3(img_L)     # 3xHlrxWlr
        H_tensor = util.single2tensor3(img_H)     # 3xHhrxWhr
        if hm is not None:
            HM_tensor = torch.from_numpy(hm).float()     # KxHlrxWlr
            L_tensor = torch.cat([L_tensor, HM_tensor], dim=0)  # (3+K)xHlrxWlr

        if L_path is None:
            L_path = H_path
        return {'L': L_tensor, 'H': H_tensor, 'L_path': L_path, 'H_path': H_path}
