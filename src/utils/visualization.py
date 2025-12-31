"""
Visualization Utilities for NeuroFusionXAI

Provides visualization functions for model explanations and results.
"""

from typing import Optional, Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def plot_attention_maps(
    attention_weights: Dict[str, np.ndarray],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 5),
) -> plt.Figure:
    """
    Plot attention maps from the model.
    
    Args:
        attention_weights: Dictionary of attention weights
        save_path: Optional path to save figure
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    n_maps = len(attention_weights)
    fig, axes = plt.subplots(1, n_maps, figsize=figsize)
    
    if n_maps == 1:
        axes = [axes]
    
    for ax, (name, attn) in zip(axes, attention_weights.items()):
        # Average over heads if multi-head
        if len(attn.shape) > 2:
            attn = attn.mean(axis=0)
        
        im = ax.imshow(attn, cmap='viridis', aspect='auto')
        ax.set_title(name)
        ax.set_xlabel('Key')
        ax.set_ylabel('Query')
        plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_shap_values(
    shap_values: Dict[str, np.ndarray],
    feature_names: Optional[List[str]] = None,
    top_k: int = 20,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot SHAP values as bar chart.
    
    Args:
        shap_values: Dictionary of SHAP values per modality
        feature_names: Names of features
        top_k: Number of top features to show
        save_path: Optional save path
    
    Returns:
        Matplotlib figure
    """
    n_modalities = len(shap_values)
    fig, axes = plt.subplots(1, n_modalities, figsize=(6*n_modalities, 8))
    
    if n_modalities == 1:
        axes = [axes]
    
    for ax, (modality, values) in zip(axes, shap_values.items()):
        values = np.array(values).flatten()
        
        # Get top features
        top_indices = np.argsort(np.abs(values))[-top_k:]
        top_values = values[top_indices]
        
        # Create labels
        if feature_names:
            labels = [feature_names[i] for i in top_indices]
        else:
            labels = [f'Feature {i}' for i in top_indices]
        
        # Color based on sign
        colors = ['red' if v < 0 else 'blue' for v in top_values]
        
        ax.barh(range(len(top_values)), top_values, color=colors)
        ax.set_yticks(range(len(top_values)))
        ax.set_yticklabels(labels)
        ax.set_xlabel('SHAP Value')
        ax.set_title(f'{modality} Feature Importance')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_gradcam(
    volume: np.ndarray,
    heatmap: np.ndarray,
    slice_indices: Optional[List[int]] = None,
    save_path: Optional[str] = None,
    alpha: float = 0.4,
) -> plt.Figure:
    """
    Plot Grad-CAM overlay on brain volume slices.
    
    Args:
        volume: Original 3D volume
        heatmap: Grad-CAM heatmap
        slice_indices: Slice indices to show (default: middle slices)
        save_path: Optional save path
        alpha: Overlay transparency
    
    Returns:
        Matplotlib figure
    """
    if slice_indices is None:
        # Use middle slices in each dimension
        D, H, W = volume.shape
        slice_indices = [D//2, H//2, W//2]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Normalize volume
    vol_norm = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)
    
    # Normalize heatmap
    heat_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    
    # Axial slice (z)
    axes[0, 0].imshow(vol_norm[slice_indices[0], :, :], cmap='gray')
    axes[0, 0].set_title(f'Axial (z={slice_indices[0]})')
    
    axes[1, 0].imshow(vol_norm[slice_indices[0], :, :], cmap='gray')
    axes[1, 0].imshow(heat_norm[slice_indices[0], :, :], cmap='jet', alpha=alpha)
    axes[1, 0].set_title('Axial + Grad-CAM')
    
    # Coronal slice (y)
    axes[0, 1].imshow(vol_norm[:, slice_indices[1], :], cmap='gray')
    axes[0, 1].set_title(f'Coronal (y={slice_indices[1]})')
    
    axes[1, 1].imshow(vol_norm[:, slice_indices[1], :], cmap='gray')
    axes[1, 1].imshow(heat_norm[:, slice_indices[1], :], cmap='jet', alpha=alpha)
    axes[1, 1].set_title('Coronal + Grad-CAM')
    
    # Sagittal slice (x)
    axes[0, 2].imshow(vol_norm[:, :, slice_indices[2]], cmap='gray')
    axes[0, 2].set_title(f'Sagittal (x={slice_indices[2]})')
    
    axes[1, 2].imshow(vol_norm[:, :, slice_indices[2]], cmap='gray')
    axes[1, 2].imshow(heat_norm[:, :, slice_indices[2]], cmap='jet', alpha=alpha)
    axes[1, 2].set_title('Sagittal + Grad-CAM')
    
    for ax in axes.flat:
        ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_training_history(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot training history.
    
    Args:
        history: Dictionary of metric histories
        save_path: Optional save path
    
    Returns:
        Matplotlib figure
    """
    n_metrics = len(history)
    n_cols = min(3, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    axes = np.array(axes).flatten()
    
    for ax, (metric, values) in zip(axes, history.items()):
        ax.plot(values)
        ax.set_title(metric)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.3)
    
    # Hide unused axes
    for i in range(len(history), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None,
    normalize: bool = True,
) -> plt.Figure:
    """
    Plot confusion matrix.
    
    Args:
        cm: Confusion matrix
        class_names: Names of classes
        save_path: Optional save path
        normalize: Whether to normalize
    
    Returns:
        Matplotlib figure
    """
    if normalize:
        cm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-8)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel='True label',
        xlabel='Predicted label',
    )
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    
    # Add text annotations
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                   ha='center', va='center',
                   color='white' if cm[i, j] > thresh else 'black')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig
