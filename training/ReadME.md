# Training

## `train_denoiser.py`

### Used for

Attention Residual U-Net

### Input

* Noisy Sinogram

### Target

* Clean Sinogram

---

## `train_artifact.py`

### Used for

Artifact CNN

### Input

* Artifact Image

### Target

* Ground Truth Image

---

## `train_confidence.py`

### Used for

Confidence Network

### Input

* Image
* Uncertainty Map

### Target

* Confidence Labels
