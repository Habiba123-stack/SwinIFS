SwinFSR
— SwinFSR: Landmark-Guided Swin Transformer for Face Super-Resolution
SwinFSR

Landmark-Guided Multiscale Swin Transformer for Identity-Preserving Face Super-Resolution

SwinFSR is a lightweight but effective Transformer-based architecture designed for 4× and 8× face super-resolution.
The method integrates facial landmark heatmaps with a hierarchical Swin Transformer to reconstruct identity-consistent high-resolution facial images under extreme degradation.

🔥 Key Features

Landmark-Guided Input: 8-channel input (RGB + 5 Gaussian landmark heatmaps)

Transformer Backbone: Swin Transformer with 6 Residual Swin Transformer Blocks (RSTBs)

Multi-Scale SR: Supports 4× (32→128) and 8× (16→128)

Identity Preservation: Strong geometric and structural consistency

Evaluation Metrics: PSNR (Y), SSIM (Y), LPIPS (RGB)
