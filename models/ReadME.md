# Models

## `unet_denoiser.py`

```text id="mhy82m"
Noisy Sinogram
+
Uncertainty Map
        ↓
Attention Residual U-Net
        ↓
Clean Sinogram
```

### Job

Remove noise before reconstruction.

---

## `artifact_cnn.py`

```text id="aqvkgv"
Reconstructed Image
        ↓
Artifact CNN
        ↓
Cleaner Image
```

### Job

Remove:

* Ring artifacts
* Streak artifacts
* Blur

after reconstruction.

---

## `confidence_net.py`

```text id="c7m1is"
Image
+
Uncertainty Map
        ↓
Confidence Network
        ↓
Confidence Map
```

### Job

Tell us which regions are reliable.
