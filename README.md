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

```markdown
![SwinFSR Methodology](Figures/Methodology_Research.png)
