"""
LIME Explainer for NeuroFusionXAI

This module implements Local Interpretable Model-agnostic Explanations (LIME)
for providing local approximations of model behavior.

Implements Eq. 9:
ξ(x) = argmin_{g∈G} L(f, g, π_x) + Ω(g)

Reference: "Why Should I Trust You?" (Ribeiro et al., 2016)
"""

from typing import Optional, Dict, List, Tuple, Callable, Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge


class LIMEExplainer:
    """
    LIME explainer for neuroimaging data.
    
    Generates local interpretable explanations by fitting a simple
    linear model around the prediction point.
    
    Implements Eq. 9:
    ξ(x) = argmin_{g∈G} L(f, g, π_x) + Ω(g)
    
    Args:
        model: The model to explain
        num_samples: Number of perturbation samples
        num_features: Number of top features to return
        kernel_width: Width of the exponential kernel
        feature_selection: Method for selecting features
    """
    
    def __init__(
        self,
        model: nn.Module,
        num_samples: int = 1000,
        num_features: int = 10,
        kernel_width: float = 0.25,
        feature_selection: str = 'auto',
    ):
        self.model = model
        self.num_samples = num_samples
        self.num_features = num_features
        self.kernel_width = kernel_width
        self.feature_selection = feature_selection
        
        # Interpretable model
        self.interpretable_model = Ridge(alpha=1.0)
        
    def _get_superpixels_3d(
        self,
        volume: np.ndarray,
        n_segments: int = 50,
    ) -> np.ndarray:
        """
        Generate 3D superpixel segmentation for neuroimaging data.
        
        Args:
            volume: 3D volume (D, H, W)
            n_segments: Approximate number of segments
            
        Returns:
            Segmentation labels for each voxel
        """
        # Simple grid-based segmentation for 3D volumes
        D, H, W = volume.shape
        
        # Calculate segment size
        total_voxels = D * H * W
        segment_size = int(np.cbrt(total_voxels / n_segments))
        segment_size = max(4, segment_size)
        
        # Create grid-based segments
        segments = np.zeros_like(volume, dtype=np.int32)
        segment_id = 0
        
        for d in range(0, D, segment_size):
            for h in range(0, H, segment_size):
                for w in range(0, W, segment_size):
                    d_end = min(d + segment_size, D)
                    h_end = min(h + segment_size, H)
                    w_end = min(w + segment_size, W)
                    segments[d:d_end, h:h_end, w:w_end] = segment_id
                    segment_id += 1
                    
        return segments
    
    def _perturb_sample(
        self,
        volume: np.ndarray,
        segments: np.ndarray,
        num_samples: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate perturbations by randomly masking superpixels.
        
        Args:
            volume: Original 3D volume
            segments: Superpixel segmentation
            num_samples: Number of perturbations to generate
            
        Returns:
            Perturbed samples and binary mask indicating active segments
        """
        n_segments = segments.max() + 1
        
        # Generate random binary masks
        masks = np.random.binomial(1, 0.5, size=(num_samples, n_segments))
        
        # First sample is always the original
        masks[0] = 1
        
        # Generate perturbed samples
        perturbed_samples = []
        
        for mask in masks:
            perturbed = volume.copy()
            
            for seg_id in range(n_segments):
                if mask[seg_id] == 0:
                    # Mask this segment (set to mean value)
                    segment_mask = segments == seg_id
                    perturbed[segment_mask] = 0  # or np.mean(volume[segment_mask])
                    
            perturbed_samples.append(perturbed)
            
        return np.array(perturbed_samples), masks
    
    def _compute_kernel_weights(
        self,
        masks: np.ndarray,
    ) -> np.ndarray:
        """
        Compute kernel weights based on distance from original.
        
        Uses exponential kernel: π_x(z) = exp(-D(x,z)² / σ²)
        
        Args:
            masks: Binary masks indicating active segments
            
        Returns:
            Kernel weights for each sample
        """
        # Distance is measured as fraction of segments turned off
        original = masks[0]
        distances = np.sum(masks != original, axis=1) / masks.shape[1]
        
        # Exponential kernel
        weights = np.exp(-(distances ** 2) / (self.kernel_width ** 2))
        
        return weights
    
    def explain(
        self,
        inputs: Dict[str, torch.Tensor],
        target_class: Optional[int] = None,
        modality: str = 'smri',
    ) -> Dict[str, Any]:
        """
        Generate LIME explanation for an input.
        
        Args:
            inputs: Dictionary of input tensors per modality
            target_class: Class to explain (default: predicted class)
            modality: Which modality to explain
            
        Returns:
            Dictionary containing:
                - feature_importance: Importance scores per segment
                - segments: Segment labels
                - local_prediction: Local model prediction
                - r2_score: R² score of local model fit
        """
        self.model.eval()
        device = next(self.model.parameters()).device
        
        # Get original prediction
        with torch.no_grad():
            output = self.model(inputs)
            if isinstance(output, dict):
                original_pred = output['probabilities']
            else:
                original_pred = torch.softmax(output, dim=-1)
                
            if target_class is None:
                target_class = original_pred.argmax(dim=-1).item()
                
        # Extract volume for explanation
        volume = inputs[modality].squeeze().cpu().numpy()  # (D, H, W)
        if len(volume.shape) == 4:
            volume = volume[0]  # Remove channel dim if present
            
        # Generate superpixels
        segments = self._get_superpixels_3d(volume)
        n_segments = segments.max() + 1
        
        # Generate perturbations
        perturbed_samples, masks = self._perturb_sample(
            volume, segments, self.num_samples
        )
        
        # Get predictions for perturbations
        predictions = []
        batch_size = 32
        
        for i in range(0, len(perturbed_samples), batch_size):
            batch = perturbed_samples[i:i + batch_size]
            batch_tensor = torch.from_numpy(batch).float().unsqueeze(1).to(device)
            
            # Create input dict with perturbed modality
            batch_inputs = {k: v.expand(len(batch), -1, -1, -1, -1) 
                          for k, v in inputs.items()}
            batch_inputs[modality] = batch_tensor
            
            with torch.no_grad():
                output = self.model(batch_inputs)
                if isinstance(output, dict):
                    probs = output['probabilities'][:, target_class]
                else:
                    probs = torch.softmax(output, dim=-1)[:, target_class]
                predictions.extend(probs.cpu().numpy())
                
        predictions = np.array(predictions)
        
        # Compute kernel weights
        weights = self._compute_kernel_weights(masks)
        
        # Fit interpretable model (weighted linear regression)
        self.interpretable_model.fit(masks, predictions, sample_weight=weights)
        
        # Get feature importance
        feature_importance = self.interpretable_model.coef_
        
        # Compute R² score
        predicted = self.interpretable_model.predict(masks)
        ss_res = np.sum(weights * (predictions - predicted) ** 2)
        ss_tot = np.sum(weights * (predictions - np.average(predictions, weights=weights)) ** 2)
        r2_score = 1 - (ss_res / (ss_tot + 1e-8))
        
        # Get top features
        top_features = np.argsort(np.abs(feature_importance))[-self.num_features:]
        
        return {
            'feature_importance': feature_importance,
            'top_features': top_features,
            'segments': segments,
            'local_prediction': self.interpretable_model.predict(masks[0:1])[0],
            'r2_score': r2_score,
            'target_class': target_class,
        }
    
    def get_explanation_map(
        self,
        explanation: Dict[str, Any],
    ) -> np.ndarray:
        """
        Convert segment-level explanation to voxel-level map.
        
        Args:
            explanation: Output from explain()
            
        Returns:
            3D importance map matching input volume shape
        """
        segments = explanation['segments']
        importance = explanation['feature_importance']
        
        # Create importance map
        importance_map = np.zeros_like(segments, dtype=np.float32)
        
        for seg_id in range(len(importance)):
            importance_map[segments == seg_id] = importance[seg_id]
            
        return importance_map
    
    def explain_multimodal(
        self,
        inputs: Dict[str, torch.Tensor],
        target_class: Optional[int] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Generate explanations for all modalities.
        
        Args:
            inputs: Dictionary of input tensors
            target_class: Class to explain
            
        Returns:
            Dictionary of explanations per modality
        """
        explanations = {}
        
        for modality in inputs.keys():
            explanations[modality] = self.explain(
                inputs, target_class, modality
            )
            
        return explanations


class LIMETextExplainer:
    """
    LIME explainer for tabular/clinical features.
    
    Used for explaining predictions based on clinical metadata
    in addition to imaging data.
    """
    
    def __init__(
        self,
        model: nn.Module,
        feature_names: List[str],
        num_samples: int = 500,
        num_features: int = 10,
    ):
        self.model = model
        self.feature_names = feature_names
        self.num_samples = num_samples
        self.num_features = num_features
        self.interpretable_model = Ridge(alpha=1.0)
        
    def explain(
        self,
        features: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Generate LIME explanation for tabular features.
        
        Args:
            features: Input feature tensor
            target_class: Class to explain
            
        Returns:
            Explanation dictionary
        """
        self.model.eval()
        device = features.device
        
        # Get original prediction
        with torch.no_grad():
            original_pred = self.model(features)
            if target_class is None:
                target_class = original_pred.argmax(dim=-1).item()
                
        # Convert to numpy
        features_np = features.cpu().numpy().flatten()
        
        # Generate perturbations
        std = np.std(features_np) + 1e-8
        perturbations = np.random.normal(0, std, (self.num_samples, len(features_np)))
        perturbed_samples = features_np + perturbations
        perturbed_samples[0] = features_np  # First sample is original
        
        # Get predictions
        predictions = []
        batch_size = 64
        
        for i in range(0, len(perturbed_samples), batch_size):
            batch = torch.from_numpy(perturbed_samples[i:i + batch_size]).float().to(device)
            
            with torch.no_grad():
                output = self.model(batch)
                probs = torch.softmax(output, dim=-1)[:, target_class]
                predictions.extend(probs.cpu().numpy())
                
        predictions = np.array(predictions)
        
        # Compute weights
        distances = np.sqrt(np.sum(perturbations ** 2, axis=1))
        weights = np.exp(-(distances ** 2) / (0.25 ** 2))
        
        # Fit interpretable model
        self.interpretable_model.fit(perturbed_samples, predictions, sample_weight=weights)
        
        # Get importance
        importance = self.interpretable_model.coef_
        
        return {
            'feature_importance': importance,
            'feature_names': self.feature_names,
            'top_features': np.argsort(np.abs(importance))[-self.num_features:],
            'target_class': target_class,
        }
