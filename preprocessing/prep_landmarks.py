import os
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

# ==============================================================================
# WARNING: Please verify and update these paths on your system before running!
# ==============================================================================
IMG_DIR = "/home/habiba/Desktop/Dataset/Split/img_align_celeba"
LANDMARKS_CSV = "/home/habiba/Desktop/Dataset/Splits/list_landmarks_align_celeba.csv"
PART_CSV = "/home/habiba/Downloads/list_eval_partition.csv"

# Output directories for processed data
OUT_HR_TRAIN = "/home/habiba/Desktop/Dataset/Split/Celeba/HR_128x128/train"
OUT_HR_TEST  = "/home/habiba/Desktop/Dataset/Split/Celeba/HR_128x128/test"
OUT_LR_TRAIN = "/home/habiba/Desktop/Dataset/Split/Celeba/LR/X4/train"
OUT_LR_TEST  = "/home/habiba/Desktop/Dataset/Split/Celeba/LR/X4/test"
OUT_HM_TRAIN = "/home/habiba/Desktop/Dataset/Split/Celeba/LR/X4_landmarks/train"
OUT_HM_TEST  = "/home/habiba/Desktop/Dataset/Split/Celeba/LR/X4_landmarks/test"

# Parameters
HR_SIZE = 128   # High Resolution size
LR_SIZE = 32    # Low Resolution size (128 / 4 = 32, hence X4 downsampling)
SIGMA = 1.2     # Standard deviation for Gaussian kernel in heatmap generation
MARGIN_SCALE = 1.0 # DIC-Net style wide-crop margin scale

# Create directories if they don't exist
for d in [OUT_HR_TRAIN, OUT_HR_TEST, OUT_LR_TRAIN, OUT_LR_TEST, OUT_HM_TRAIN, OUT_HM_TEST]:
    os.makedirs(d, exist_ok=True)

# ===== LANDMARK AND HEATMAP UTILITIES =====

def get_landmarks5(row):
    """Extracts 5 facial landmarks from a DataFrame row."""
    return np.array([
        [row['lefteye_x'],   row['lefteye_y']],
        [row['righteye_x'],  row['righteye_y']],
        [row['nose_x'],      row['nose_y']],
        [row['leftmouth_x'], row['leftmouth_y']],
        [row['rightmouth_x'],row['rightmouth_y']],
    ], dtype=np.float32)

def make_heatmaps(pts, H, W, sigma):
    """Generates 5-channel heatmaps (K x H x W) from landmark coordinates."""
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    heatmaps = []
    for (x, y) in pts:
        d2 = (xx - x)**2 + (yy - y)**2
        h = np.exp(-d2 / (2 * sigma**2))
        heatmaps.append(h)
    hm = np.stack(heatmaps, axis=0).astype(np.float32)
    expected_shape = (len(pts), H, W)
    if hm.shape != expected_shape:
        print(f"!!! CRITICAL ERROR: Heatmap generated with shape {hm.shape}, expected {expected_shape}")
        return np.zeros(expected_shape, dtype=np.float32)
    return hm

def crop_face_wide(img, landmarks, margin_scale):
    """DIC-Net inspired crop: Creates a wider bounding box around the 5 landmarks."""
    w, h = img.size
    x_min, y_min = np.min(landmarks, axis=0)
    x_max, y_max = np.max(landmarks, axis=0)
    face_w = x_max - x_min
    face_h = y_max - y_min
    box_size = int(max(face_w, face_h) * (2.5 + margin_scale))
    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2
    left = int(max(0, cx - box_size / 2))
    top = int(max(0, cy - box_size / 2))
    right = int(min(w, cx + box_size / 2))
    bottom = int(min(h, cy + box_size / 2))
    return img.crop((left, top, right, bottom)), (left, top, right, bottom)

# ===== LOAD DATA AND START PROCESSING =====
print(f"Loading data from: {LANDMARKS_CSV} and {PART_CSV}")
part = pd.read_csv(PART_CSV)
split_map = dict(zip(part['image_id'], part['partition']))

landmarks_df = pd.read_csv(LANDMARKS_CSV)
landmarks_df.set_index("image_id", inplace=True)

print(f"Processing CelebA preprocessing (HR={HR_SIZE}x{HR_SIZE}, LR={LR_SIZE}x{LR_SIZE})...")
n_train = n_test = 0

for img_name in tqdm(part['image_id']):
    img_path = os.path.join(IMG_DIR, img_name)
    if not os.path.exists(img_path) or img_name not in landmarks_df.index:
        continue

    split = int(split_map.get(img_name, -1))
    if split not in [0, 2]:  # Only process train (0) and test (2)
        continue

    lmk_row = landmarks_df.loc[img_name]
    pts5 = get_landmarks5(lmk_row)

    try:
        img = Image.open(img_path).convert('RGB')
    except Exception as e:
        print(f"Error opening image {img_name}: {e}")
        continue

    # 1. WIDE CROP
    img_cropped, (l, t, r, b) = crop_face_wide(img, pts5, margin_scale=MARGIN_SCALE)

    # 2. RESIZE HR AND LR
    img_hr = img_cropped.resize((HR_SIZE, HR_SIZE), Image.LANCZOS)
    img_lr = img_hr.resize((LR_SIZE, LR_SIZE), Image.BICUBIC)

    # 3. LANDMARK AND HEATMAP GENERATION
    pts_crop = pts5.copy()
    crop_width = r - l
    crop_height = b - t
    if crop_width <= 0 or crop_height <= 0:
        continue

    pts_crop[:, 0] = (pts_crop[:, 0] - l) * (LR_SIZE / crop_width)
    pts_crop[:, 1] = (pts_crop[:, 1] - t) * (LR_SIZE / crop_height)

    hm = make_heatmaps(pts_crop, LR_SIZE, LR_SIZE, sigma=SIGMA)

    # 4. SAVE FILES
    if split == 0:
        hr_out, lr_out, hm_out = OUT_HR_TRAIN, OUT_LR_TRAIN, OUT_HM_TRAIN
        n_train += 1
    else:
        hr_out, lr_out, hm_out = OUT_HR_TEST, OUT_LR_TEST, OUT_HM_TEST
        n_test += 1

    img_hr.save(os.path.join(hr_out, img_name))
    img_lr.save(os.path.join(lr_out, img_name))
    np.save(os.path.join(hm_out, img_name.replace('.jpg', '.npy')), hm)

print(f"\n✅ Preprocessing Complete for X4.")
print(f"Total Images Processed - Train: {n_train}, Test: {n_test}")
