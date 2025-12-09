# 🚀 SwinFSR: Landmark-Guided Swin Transformer for Face Super-Resolution

SwinFSR is a landmark-guided Swin Transformer model designed for **4× and 8× face super-resolution**.  
It integrates facial landmark heatmaps with a hierarchical Swin Transformer to reconstruct identity-consistent high-resolution facial images under severe degradation.

---

## 🔥 Key Features
- **Landmark-Guided Input:** 8-channel input (RGB + 5 Gaussian landmark heatmaps)  
- **Transformer Backbone:** Swin Transformer with 6 Residual Swin Transformer Blocks (RSTBs)  
- **Multi-Scale SR:** Supports 4× (32→128) and 8× (16→128)  
- **Identity Preservation:** Strong geometric and structural consistency  
- **Evaluation Metrics:** PSNR (Y), SSIM (Y), LPIPS (RGB)

---

## 🧩 Methodology Overview
SwinFSR fuses facial geometry (landmark heatmaps) with transformer-based local–global feature modeling.  
A shallow convolution extracts initial features, and stacked RSTBs enhance facial structure and texture.  
PixelShuffle upsampling reconstructs the high-resolution output.

---

## 🧱 Methodology Diagram

---


#🚀 SwinFSR: Landmark-Guided Swin Transformer for Face Super-Resolution

SwinFSR is a landmark-guided Swin Transformer model designed for 4× and 8× face super-resolution.
It integrates dense landmark heatmaps with hierarchical shifted-window attention, enabling accurate reconstruction of identity-consistent facial details even under extreme low-resolution degradation.

---

## 🔥 Key Features

Landmark-Guided SR: Injects geometric priors using 5-point Gaussian heatmaps

Transformer Backbone: Swin Transformer with 6 Residual Swin Transformer Blocks (RSTBs)

Multiscale SR: Supports 4× (32→128) and 8× (16→128) upscaling

Identity Preservation: Maintains consistent geometry around eyes, lips & nose

Efficient Training: Lightweight and optimized for single-GPU setups

Evaluation Metrics: PSNR (Y), SSIM (Y), LPIPS (RGB)
---

## 🧩 Methodology Overview

SwinFSR fuses facial geometry (landmark heatmaps) with transformer-based local–global modeling.
A shallow 3×3 convolution extracts low-level features, while stacked RSTBs model long-range dependencies and restore fine facial details.
PixelShuffle reconstructs the high-resolution output, supported by a bicubic upsample skip connection for stable identity preservation.

## 🧱 Methodology Diagram

## 🖼 Visual Results
8× Face Super-Resolution (16 → 128)

(You may add 4× results or comparison grids in this section.)

## 📁 Project Structure
SwinFSR/
│── train_swinfsr.py                 # Training + validation
│── test_swinfsr.py                  # Inference/testing (optional)
│── README.md
│
├── models/
│   ├── network_swinfsr.py           # SwinFSR architecture
│   ├── model_base.py
│   ├── model_plain.py
│   ├── select_model.py
│   └── select_network.py
│
├── data/
│   ├── dataset_sr.py                # Loads HR, LR and landmark heatmaps
│   └── select_dataset.py
│
├── preprocessing/
│   └── prep_landmarks.py            # Generate HR crops, LR images & heatmaps
│
├── utils/
│   ├── utils_image.py
│   ├── utils_option.py
│   ├── utils_model.py
│   ├── utils_logger.py
│   ├── utils_dist.py
│   └── utils_modelsummary.py
│
├── options/
│   └── swinfsr/
│        ├── train_swinfsr_sr_celeba_x4.json
│        └── train_swinfsr_sr_celeba_x8.json
│
└── Figures/
    ├── Methodology_Research.png
    └── x8.png

## 📦 Dataset Preparation

SwinFSR is trained on CelebA, preprocessed into:

HR images: 128×128

LR images (4×): 32×32

LR images (8×): 16×16

Landmark heatmaps: 5 Gaussian maps per LR image

Generate HR, LR, and landmark heatmaps:

python preprocessing/prep_landmarks.py


This creates:

HR_128x128/train  
HR_128x128/test  
LR/X4/train  
LR/X4/test  
LR/X4_landmarks/train  
LR/X4_landmarks/test  


(and similarly for X8)

## 🚀 Training
4× Super-Resolution
python train_swinfsr.py --opt options/swinfsr/train_swinfsr_sr_celeba_x4.json

8× Super-Resolution
python train_swinfsr.py --opt options/swinfsr/train_swinfsr_sr_celeba_x8.json


## Training performs:

Automatic checkpoint saving

Testing at specified intervals

Logging PSNR, SSIM, LPIPS

Automatic bicubic comparison

Optional saving of SR visual outputs

## 🔍 Testing / Inference
python test_swinfsr.py --opt options/swinfsr/train_swinfsr_sr_celeba_x4.json --save_results


Results are stored in:

results_swinfsr/


You may also test on any custom LR image folder.

## 📊 Evaluation Metrics

SwinFSR uses standard metrics in the face SR literature:

PSNR (Y-channel)

SSIM (Y-channel)

LPIPS (RGB) using AlexNet backbone

This evaluation protocol follows:

SwinIR (ICCV 2021)

DIC-Net (CVPR 2020)

FSRNet (CVPR 2018)

SPARNet (TIP 2021)

## 🧠 Model Architecture Summary

Input: 8 channels (RGB + 5 landmark heatmaps)

Shallow feature extractor: 3×3 Conv

Deep feature extraction: 6 × RSTBs with shifted-window MHSA

Upsampling: PixelShuffle

Reconstruction: 3×3 Conv

Skip connection: Bicubic LR → HR

## 🧪 Results Summary

SwinFSR achieves:

Superior perceptual sharpness

Accurate identity reconstruction

Clear eye, lip, and nose details

Lower LPIPS compared to CNN/GAN/SwinIR baselines

Strong robustness on extreme low-resolution faces

## 🤝 Acknowledgements

SwinFSR builds upon foundational codebases:

KAIR — https://github.com/cszn/KAIR

SwinIR — https://github.com/JingyunLiang/SwinIR

## 📝 Citation
Shahzad.
“SwinFSR: Landmark-Guided Swin Transformer for Identity-Preserving Face Super-Resolution.”
MS Thesis, 2025.

👨‍💻 Author

Shahzad
MS Thesis — Landmark-Guided Face Super-Resolution (2025)
