# VECC-project
This is the initial software architecture and module-level implementation skeleton. The reconstruction operator A, tomography geometry, uncertainty-map generation, SSIM loss, attention gates, and dataset loaders will be finalized after we receive the exact dataset specification.

project/

├── data/
│
├── models/
│   ├── unet_denoiser.py
│   ├── artifact_cnn.py
│   └── confidence_net.py
│
├── reconstruction/
│   ├── landweber.py
│   └── tikhonov.py
│
├── training/
│   ├── train_denoiser.py
│   ├── train_artifact.py
│   └── train_confidence.py
│
└── inference.py

data/ (Stores everything related to datasets.)
├── train/
├── validation/
├── test/
├── noisy_sinograms/
├── clean_sinograms/
└── uncertainty_maps/

Contains:

Ground truth flame images
Noisy sinograms
Clean sinograms
Uncertainty maps

models/ (Contains all AI architectures.)

reconstruction/ (Contains mathematical reconstruction algorithms.)

training/ (Contains training scripts.)

inference.py (This is the final pipeline. Everything comes together here.)

Workflow:
Noisy Sinogram + Uncertainty Map
           ↓
U-Net Denoiser
           ↓
Clean Sinogram
           ↓
Landweber Reconstruction
           ↓
Tikhonov Refinement
           ↓
Artifact CNN
           ↓
Final Flame Image
           ↓
Confidence Network
           ↓
Confidence Map

Output:
1. Final Flame Image
2. Confidence Map

# The project is divided into five modules. The data folder stores flame images, sinograms, and uncertainty maps. The models folder contains the neural networks for denoising, artifact removal, and confidence estimation. The reconstruction folder contains classical tomography algorithms such as Landweber and Tikhonov. The training folder is used to train each network individually. Finally, inference.py integrates all modules into a complete reconstruction pipeline that produces a reconstructed flame image along with a confidence map.
