import os.path
import math
import argparse
import random
import numpy as np
import logging
import time  # ✅ added
import lpips

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch

from utils import utils_logger
from utils import utils_image as util
from utils import utils_option as option
from utils.utils_dist import get_dist_info, init_dist

from data.select_dataset import define_Dataset
from models.select_model import define_Model

# SSIM (modern skimage API)
from skimage.metrics import structural_similarity as ssim


'''
# --------------------------------------------
# PSNR/SSIM trainer for Swinfsr models
# --------------------------------------------
'''

# ---------- SSIM helpers ----------
def rgb_to_y_uint8(img_uint8: np.ndarray) -> np.ndarray:
    """RGB uint8 -> Y (BT.601) uint8"""
    img = img_uint8.astype(np.float32)
    y = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
    return np.clip(y, 0, 255).astype(np.uint8)

def safe_ssim(E_img: np.ndarray, H_img: np.ndarray, border: int = 0, on_y: bool = True) -> float:
    """
    SSIM with optional border shave and Y-channel handling
    """
    if border > 0 and min(E_img.shape[0], E_img.shape[1]) > 2 * border:
        E = E_img[border:-border, border:-border, ...]
        H = H_img[border:-border, border:-border, ...]
    else:
        E, H = E_img, H_img

    if on_y:
        E = rgb_to_y_uint8(E)
        H = rgb_to_y_uint8(H)
        ch_axis = None
    else:
        ch_axis = -1

    h, w = E.shape[:2]
    m = min(h, w)
    win = 7 if m >= 7 else (5 if m >= 5 else 3)
    return float(ssim(E, H, data_range=255, channel_axis=ch_axis, win_size=win))
# -----------------------------------


def run_final_eval(model, test_loader, border, logger, save_dir=None):
    """Evaluate model on test_loader with PSNR, SSIM, LPIPS, and inference time."""
    
    lpips_model = lpips.LPIPS(net='alex').cuda()
    lpips_model.eval()
    
    avg_psnr = 0.0
    avg_ssim = 0.0
    avg_lpips = 0.0
    avg_time = 0.0
    n = 0

    def img_to_lpips_tensor(img):
        im_t = torch.from_numpy(img.astype(np.float32)/127.5 - 1.0).permute(2,0,1).unsqueeze(0).contiguous()
        return im_t.cuda()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for test_data in test_loader:
            n += 1
            image_name_ext = os.path.basename(test_data['L_path'][0])

            model.feed_data(test_data)
            start_t = time.time()
            model.test()
            elapsed = time.time() - start_t
            avg_time += elapsed

            visuals = model.current_visuals()
            E_img = util.tensor2uint(visuals['E'])
            H_img = util.tensor2uint(visuals['H'])

            ps = util.calculate_psnr(E_img, H_img, border=border)
            ss = safe_ssim(E_img, H_img, border=border, on_y=True)

            # LPIPS
            E_lpips = img_to_lpips_tensor(E_img)
            H_lpips = img_to_lpips_tensor(H_img)
            lpips_val = float(lpips_model(E_lpips, H_lpips).item())
            avg_lpips += lpips_val

            avg_psnr += ps
            avg_ssim += ss

            logger.info(f"[EVAL] {image_name_ext} | PSNR: {ps:.2f} | SSIM(Y): {ss:.4f} | LPIPS: {lpips_val:.4f} | Time: {elapsed:.4f}s")
            if save_dir:
                util.imsave(E_img, os.path.join(save_dir, image_name_ext))

    avg_psnr /= max(n, 1)
    avg_ssim /= max(n, 1)
    avg_lpips /= max(n, 1)
    avg_time /= max(n, 1)

    logger.info(f"[FINAL EVAL] Avg PSNR: {avg_psnr:.2f} dB | Avg SSIM(Y): {avg_ssim:.4f} | Avg LPIPS: {avg_lpips:.4f} | "
                f"Avg Inference Time: {avg_time:.4f}s ({1/avg_time:.2f} FPS)")
    print(f"[FINAL EVAL] Avg PSNR: {avg_psnr:.2f} dB | Avg SSIM(Y): {avg_ssim:.4f} | Avg LPIPS: {avg_lpips:.4f} | "
          f"Avg Inference Time: {avg_time:.4f}s ({1/avg_time:.2f} FPS)")






def main(json_path='/home/habiba/Desktop/option/swinfsr/train_swinfsr_x8.json'):
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default=json_path, help='Path to option JSON file.')
    parser.add_argument('--launcher', default='pytorch', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--dist', default=False)

    args = parser.parse_args()
    opt = option.parse(args.opt, is_train=True)
    opt['dist'] = args.dist

    if opt['dist']:
        init_dist('pytorch')
    opt['rank'], opt['world_size'] = get_dist_info()

    if opt['rank'] == 0:
        util.mkdirs((path for key, path in opt['path'].items() if 'pretrained' not in key))

    init_iter_G, init_path_G = option.find_last_checkpoint(opt['path']['models'], net_type='G')
    init_iter_E, init_path_E = option.find_last_checkpoint(opt['path']['models'], net_type='E')
    opt['path']['pretrained_netG'] = init_path_G
    opt['path']['pretrained_netE'] = init_path_E
    init_iter_optimizerG, init_path_optimizerG = option.find_last_checkpoint(opt['path']['models'], net_type='optimizerG')
    opt['path']['pretrained_optimizerG'] = init_path_optimizerG
    current_step = max(init_iter_G, init_iter_E, init_iter_optimizerG)

    border = opt['scale']

    if opt['rank'] == 0:
        option.save(opt)

    opt = option.dict_to_nonedict(opt)

    max_iter = opt['train'].get('max_iter', None)
    final_eval_after_train = opt['train'].get('final_eval_after_train', True)
    final_eval_save_images = opt['train'].get('final_eval_save_images', False)

    if opt['rank'] == 0:
        logger_name = 'train'
        utils_logger.logger_info(logger_name, os.path.join(opt['path']['log'], logger_name + '.log'))
        logger = logging.getLogger(logger_name)
        logger.info(option.dict2str(opt))
    else:
        logger = logging.getLogger('train')

    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    print('Random seed: {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = define_Dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt['dataloader_batch_size']))
            if opt['rank'] == 0:
                logger.info(f'Number of train images: {len(train_set):,d}, iters: {train_size:,d}')
            if opt['dist']:
                train_sampler = DistributedSampler(train_set, shuffle=dataset_opt['dataloader_shuffle'],
                                                   drop_last=True, seed=seed)
                train_loader = DataLoader(train_set,
                                          batch_size=dataset_opt['dataloader_batch_size']//opt['num_gpu'],
                                          shuffle=False,
                                          num_workers=dataset_opt['dataloader_num_workers']//opt['num_gpu'],
                                          drop_last=True,
                                          pin_memory=True,
                                          sampler=train_sampler)
            else:
                train_loader = DataLoader(train_set,
                                          batch_size=dataset_opt['dataloader_batch_size'],
                                          shuffle=dataset_opt['dataloader_shuffle'],
                                          num_workers=dataset_opt['dataloader_num_workers'],
                                          drop_last=True,
                                          pin_memory=True)

        elif phase == 'test':
            test_set = define_Dataset(dataset_opt)
            test_loader = DataLoader(test_set, batch_size=1,
                                     shuffle=False, num_workers=1,
                                     drop_last=False, pin_memory=True)
        else:
            raise NotImplementedError(f"Phase [{phase}] is not recognized.")

    model = define_Model(opt)
    model.init_train()
    if opt['rank'] == 0:
        logger.info(model.info_network())
        logger.info(model.info_params())

    try:
        stop_now = False
        for epoch in range(1000000):
            if opt['dist']:
                train_sampler.set_epoch(epoch + seed)

            for i, train_data in enumerate(train_loader):
                if stop_now:
                    break

                current_step += 1
                model.update_learning_rate(current_step)
                model.feed_data(train_data)
                model.optimize_parameters(current_step)

                if current_step % opt['train']['checkpoint_print'] == 0 and opt['rank'] == 0:
                    logs = model.current_log()
                    message = f"<epoch:{epoch:3d}, iter:{current_step:8,d}, lr:{model.current_learning_rate():.3e}> "
                    for k, v in logs.items():
                        message += f"{k}: {v:.3e} "
                    logger.info(message)

                if current_step % opt['train']['checkpoint_save'] == 0 and opt['rank'] == 0:
                    logger.info('Saving the model.')
                    model.save(current_step)
                if current_step % opt['train']['checkpoint_test'] == 0 and opt['rank'] == 0:
                    avg_psnr = 0.0
                    avg_ssim = 0.0
                    avg_lpips = 0.0
                    avg_time = 0.0
                    idx = 0
                    lpips_model = lpips.LPIPS(net='alex').cuda()
                    lpips_model.eval()

                    def img_to_lpips_tensor(img):
                        im_t = torch.from_numpy(img.astype(np.float32)/127.5 - 1.0).permute(2,0,1).unsqueeze(0).contiguous()
                        return im_t.cuda()

                    for test_data in test_loader:
                        idx += 1
                        image_name_ext = os.path.basename(test_data['L_path'][0])
                        img_name, ext = os.path.splitext(image_name_ext)
                        img_dir = os.path.join(opt['path']['images'], img_name)
                        util.mkdir(img_dir)

                        model.feed_data(test_data)
                        start_t = time.time()
                        model.test()
                        elapsed = time.time() - start_t
                        avg_time += elapsed

                        visuals = model.current_visuals()
                        E_img = util.tensor2uint(visuals['E'])
                        H_img = util.tensor2uint(visuals['H'])

                        save_img_path = os.path.join(img_dir, f"{img_name}_{current_step}.png")
                        util.imsave(E_img, save_img_path)

                        current_psnr = util.calculate_psnr(E_img, H_img, border=border)
                        current_ssim = safe_ssim(E_img, H_img, border=border, on_y=True)
                        
                        # ---- LPIPS computation
                        E_lpips = img_to_lpips_tensor(E_img)
                        H_lpips = img_to_lpips_tensor(H_img)
                        lpips_val = float(lpips_model(E_lpips, H_lpips).item())
                        avg_lpips += lpips_val

                        logger.info(f"{idx:->4d}--> {image_name_ext:>10s} | PSNR: {current_psnr:<4.2f} dB | SSIM(Y): {current_ssim:.4f} | LPIPS: {lpips_val:.4f} | Time: {elapsed:.4f}s")

                        avg_psnr += current_psnr
                        avg_ssim += current_ssim

                    avg_psnr /= idx
                    avg_ssim /= idx
                    avg_lpips /= idx
                    avg_time /= idx

                    logger.info(f"<epoch:{epoch:3d}, iter:{current_step:8,d}, "
                                f"Avg PSNR: {avg_psnr:.2f} dB, Avg SSIM(Y): {avg_ssim:.4f}, Avg LPIPS: {avg_lpips:.4f}, "
                                f"Avg Time: {avg_time:.4f}s ({1/avg_time:.2f} FPS)\n")

                if max_iter is not None and current_step >= max_iter:
                    if opt['rank'] == 0:
                        logger.info(f"Reached max_iter={max_iter}. Saving checkpoint and running final evaluation.")
                        model.save(current_step)
                        if final_eval_after_train:
                            save_dir = None
                            if final_eval_save_images:
                                base_images = opt['path'].get('images', os.path.join(opt['path']['root'], opt['task'], 'images'))
                                save_dir = os.path.join(base_images, f'final_eval_{current_step}')
                            run_final_eval(model, test_loader, border, logger, save_dir=save_dir)
                    stop_now = True

            if stop_now:
                break

    except KeyboardInterrupt:
        if opt['rank'] == 0:
            logger.info("KeyboardInterrupt received. Saving checkpoint.")
            model.save(current_step)
            if final_eval_after_train:
                save_dir = None
                if final_eval_save_images:
                    base_images = opt['path'].get('images', os.path.join(opt['path']['root'], opt['task'], 'images'))
                    save_dir = os.path.join(base_images, f'final_eval_{current_step}')
                run_final_eval(model, test_loader, border, logger, save_dir=save_dir)


if __name__ == '__main__':
    main()



