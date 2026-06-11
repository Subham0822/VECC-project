from models.unet_denoiser import AttentionResidualUNet
from models.artifact_cnn import ArtifactRemovalCNN
from models.confidence_net import ConfidenceNet

def run_pipeline(
    noisy_sinogram,
    uncertainty_map
):

    # Step 1
    denoised_sinogram = denoiser(
        [noisy_sinogram,
         uncertainty_map]
    )

    # Step 2
    reconstructed_image = landweber(
        denoised_sinogram
    )

    # Step 3
    reconstructed_image = tikhonov(
        reconstructed_image
    )

    # Step 4
    clean_image = artifact_net(
        reconstructed_image
    )

    # Step 5
    confidence = confidence_net(
        [clean_image,
         uncertainty_map]
    )

    return clean_image, confidence