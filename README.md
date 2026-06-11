# VECC-project

This is the initial software architecture and module-level implementation skeleton. The reconstruction operator A, tomography geometry, uncertainty-map generation, SSIM loss, attention gates, and dataset loaders will be finalized after we receive the exact dataset specification.

## Project Structure

```text
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
```

## Data Module

`data/` (Stores everything related to datasets.)

```text
data/

├── train/
├── validation/
├── test/
├── noisy_sinograms/
├── clean_sinograms/
└── uncertainty_maps/
```

### Contains

* Ground truth flame images
* Noisy sinograms
* Clean sinograms
* Uncertainty maps

## Models Module

`models/` (Contains all AI architectures.)

## Reconstruction Module

`reconstruction/` (Contains mathematical reconstruction algorithms.)

## Training Module

`training/` (Contains training scripts.)

## Inference Module

`inference.py` (This is the final pipeline. Everything comes together here.)

## Workflow

```text
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
```

## Output

1. Final Flame Image
2. Confidence Map

## Summary

The project is divided into five modules. The data folder stores flame images, sinograms, and uncertainty maps. The models folder contains the neural networks for denoising, artifact removal, and confidence estimation. The reconstruction folder contains classical tomography algorithms such as Landweber and Tikhonov. The training folder is used to train each network individually. Finally, `inference.py` integrates all modules into a complete reconstruction pipeline that produces a reconstructed flame image along with a confidence map.
