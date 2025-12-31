"""
Preprocessing Utilities for NeuroFusionXAI

Provides neuroimaging preprocessing functions including:
- Volume normalization
- MNI registration
- Skull stripping
- Resampling
"""

from typing import Optional, Tuple
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom, gaussian_filter


def normalize_volume(
    volume: np.ndarray,
    method: str = 'zscore',
    percentile_range: Tuple[float, float] = (1, 99),
) -> np.ndarray:
    """
    Normalize volume intensities.
    
    Args:
        volume: Input 3D volume
        method: Normalization method ('zscore', 'minmax', 'percentile')
        percentile_range: Percentile range for percentile normalization
    
    Returns:
        Normalized volume
    """
    if method == 'zscore':
        mean = volume.mean()
        std = volume.std()
        return (volume - mean) / (std + 1e-8)
    
    elif method == 'minmax':
        vmin, vmax = volume.min(), volume.max()
        return (volume - vmin) / (vmax - vmin + 1e-8)
    
    elif method == 'percentile':
        low, high = np.percentile(volume, percentile_range)
        volume = np.clip(volume, low, high)
        return (volume - low) / (high - low + 1e-8)
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def resample_volume(
    volume: np.ndarray,
    target_shape: Tuple[int, int, int],
    order: int = 1,
) -> np.ndarray:
    """
    Resample volume to target shape.
    
    Args:
        volume: Input 3D volume
        target_shape: Target (D, H, W) shape
        order: Interpolation order (0=nearest, 1=linear, 3=cubic)
    
    Returns:
        Resampled volume
    """
    factors = [t / s for t, s in zip(target_shape, volume.shape)]
    return zoom(volume, factors, order=order)


def skull_strip(
    volume: np.ndarray,
    threshold: float = 0.1,
    smooth_sigma: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simple skull stripping using intensity thresholding.
    
    Args:
        volume: Input T1-weighted MRI volume
        threshold: Intensity threshold (relative to max)
        smooth_sigma: Gaussian smoothing sigma
    
    Returns:
        Skull-stripped volume and brain mask
    """
    # Smooth volume
    smoothed = gaussian_filter(volume, sigma=smooth_sigma)
    
    # Create initial mask
    thresh_value = threshold * smoothed.max()
    mask = smoothed > thresh_value
    
    # Fill holes and clean up
    from scipy.ndimage import binary_fill_holes, binary_erosion, binary_dilation
    
    mask = binary_fill_holes(mask)
    mask = binary_erosion(mask, iterations=2)
    mask = binary_dilation(mask, iterations=2)
    
    # Apply mask
    stripped = volume * mask
    
    return stripped, mask.astype(np.float32)


def register_to_mni(
    volume: np.ndarray,
    affine: np.ndarray,
    target_shape: Tuple[int, int, int] = (96, 112, 96),
    target_affine: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Register volume to MNI space (simplified version).
    
    For production use, consider using ANTs or FSL for proper registration.
    
    Args:
        volume: Input volume
        affine: Volume affine matrix
        target_shape: Target shape
        target_affine: Target affine (default: 2mm MNI)
    
    Returns:
        Registered volume
    """
    # Simple resampling as placeholder
    # Real implementation should use proper registration tools
    return resample_volume(volume, target_shape)


def preprocess_volume(
    nifti_path: str,
    target_shape: Tuple[int, int, int] = (96, 112, 96),
    normalize: bool = True,
    skull_strip_volume: bool = False,
) -> np.ndarray:
    """
    Complete preprocessing pipeline for a single volume.
    
    Args:
        nifti_path: Path to NIfTI file
        target_shape: Target shape
        normalize: Whether to normalize intensities
        skull_strip_volume: Whether to apply skull stripping
    
    Returns:
        Preprocessed volume
    """
    # Load volume
    img = nib.load(nifti_path)
    volume = img.get_fdata().astype(np.float32)
    affine = img.affine
    
    # Skull stripping
    if skull_strip_volume:
        volume, _ = skull_strip(volume)
    
    # Register/resample to target shape
    volume = resample_volume(volume, target_shape)
    
    # Normalize
    if normalize:
        volume = normalize_volume(volume, method='zscore')
    
    return volume


class Augmentation:
    """
    Data augmentation for 3D neuroimaging volumes.
    
    Args:
        random_flip: Enable random flipping
        random_rotation: Max rotation in degrees
        random_scale: Scale range (min, max)
        gaussian_noise: Gaussian noise std
    """
    
    def __init__(
        self,
        random_flip: bool = True,
        random_rotation: float = 15,
        random_scale: Tuple[float, float] = (0.9, 1.1),
        gaussian_noise: float = 0.01,
    ):
        self.random_flip = random_flip
        self.random_rotation = random_rotation
        self.random_scale = random_scale
        self.gaussian_noise = gaussian_noise
    
    def __call__(self, data: dict) -> dict:
        """Apply augmentations to all modalities."""
        augmented = {}
        
        # Generate random parameters (same for all modalities)
        flip_axes = []
        if self.random_flip:
            for axis in range(3):
                if np.random.random() > 0.5:
                    flip_axes.append(axis + 2)  # +2 for batch and channel dims
        
        scale = np.random.uniform(*self.random_scale) if self.random_scale else 1.0
        
        for modality, volume in data.items():
            aug_volume = volume.clone()
            
            # Random flip
            for axis in flip_axes:
                aug_volume = torch.flip(aug_volume, dims=[axis])
            
            # Add noise
            if self.gaussian_noise > 0:
                noise = torch.randn_like(aug_volume) * self.gaussian_noise
                aug_volume = aug_volume + noise
            
            augmented[modality] = aug_volume
        
        return augmented


# Import torch for Augmentation class
import torch
