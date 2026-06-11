# Reconstruction

## `landweber.py`

```text id="xg5e0m"
Denoised Sinogram
        ↓
Landweber Algorithm
        ↓
Initial Flame Image
```

### Job

Convert projection data into image.

---

## `tikhonov.py`

```text id="s5m3kq"
Initial Image
        ↓
Tikhonov Regularization
        ↓
Stabilized Image
```

### Job

Reduce instability and noise amplification.

---

### Think

* Landweber reconstructs.
* Tikhonov refines.
