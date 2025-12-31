"""
Grad-CAM Implementation for NeuroFusionXAI

This module implements Gradient-weighted Class Activation Mapping for
visual explanations of model predictions.

Implements Eq. 11 and 12:
α_k^c = (1/Z) Σ_i Σ_j (∂y^c / ∂A^k_{i,j})
L^c_{Grad-CAM} = ReLU(Σ_k α_k^c · A^k)

Reference: "Grad-CAM: Visual Explanations from Deep Networks" (Selvaraju et al., 2017)
"""

from typing import Optional, Dict, List, Tuple, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAM:
    """
    Grad-CAM for generating visual explanations.
    
    Args:
        model: Model to explain
        target_layers: Names of layers to compute Grad-CAM for
    """
    
    def __init__(self, model: nn.Module, target_layers: List[str]):
        self.model = model
        self.target_layers = target_layers
        self.activations = {}
        self.gradients = {}
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks on target layers."""
        def get_activation(name):
            def hook(module, input, output):
                self.activations[name] = output.detach()
            return hook
        
        def get_gradient(name):
            def hook(module, grad_input, grad_output):
                self.gradients[name] = grad_output[0].detach()
            return hook
        
        for name, module in self.model.named_modules():
            if name in self.target_layers:
                module.register_forward_hook(get_activation(name))
                module.register_full_backward_hook(get_gradient(name))
    
    def generate(
        self,
        inputs: Dict[str, torch.Tensor],
        target_class: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Generate Grad-CAM heatmaps.
        
        Args:
            inputs: Input tensors
            target_class: Class to explain
            
        Returns:
            Dictionary of heatmaps per target layer
        """
        self.model.eval()
        self.model.zero_grad()
        
        # Forward pass
        output = self.model(inputs)
        if isinstance(output, dict):
            logits = output['logits']
        else:
            logits = output
        
        if target_class is None:
            target_class = logits.argmax(dim=-1).item()
        
        # Backward pass
        one_hot = torch.zeros_like(logits)
        one_hot[0, target_class] = 1
        logits.backward(gradient=one_hot, retain_graph=True)
        
        # Generate heatmaps
        heatmaps = {}
        for layer_name in self.target_layers:
            if layer_name in self.activations and layer_name in self.gradients:
                activation = self.activations[layer_name]
                gradient = self.gradients[layer_name]
                
                # Global average pooling of gradients (Eq. 11)
                weights = gradient.mean(dim=(2, 3, 4) if gradient.dim() == 5 else (2, 3), keepdim=True)
                
                # Weighted combination (Eq. 12)
                cam = (weights * activation).sum(dim=1, keepdim=True)
                cam = F.relu(cam)
                
                # Normalize
                cam = cam - cam.min()
                cam = cam / (cam.max() + 1e-8)
                
                heatmaps[layer_name] = cam.squeeze().cpu().numpy()
        
        return heatmaps


class GradCAM3D(GradCAM):
    """
    3D Grad-CAM specifically for volumetric neuroimaging data.
    
    Args:
        model: Model to explain
        target_layers: Names of target layers
    """
    
    def __init__(self, model: nn.Module, target_layers: List[str]):
        super().__init__(model, target_layers)
    
    def generate_multimodal(
        self,
        inputs: Dict[str, torch.Tensor],
        target_class: Optional[int] = None,
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """Generate Grad-CAM for all modalities."""
        heatmaps = {}
        
        for modality in inputs.keys():
            heatmaps[modality] = self.generate(inputs, target_class)
        
        return heatmaps
    
    def upsample_to_input(
        self,
        heatmap: np.ndarray,
        target_shape: Tuple[int, int, int],
    ) -> np.ndarray:
        """Upsample heatmap to input resolution."""
        heatmap_tensor = torch.from_numpy(heatmap).unsqueeze(0).unsqueeze(0).float()
        upsampled = F.interpolate(heatmap_tensor, size=target_shape, mode='trilinear', align_corners=False)
        return upsampled.squeeze().numpy()
    
    def overlay_on_volume(
        self,
        volume: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.4,
    ) -> np.ndarray:
        """Overlay heatmap on original volume."""
        if heatmap.shape != volume.shape:
            heatmap = self.upsample_to_input(heatmap, volume.shape)
        
        # Normalize volume
        volume_norm = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)
        
        # Blend
        overlay = (1 - alpha) * volume_norm + alpha * heatmap
        return overlay


class IntegratedGradients:
    """
    Integrated Gradients for attribution.
    
    Args:
        model: Model to explain
        n_steps: Number of interpolation steps
    """
    
    def __init__(self, model: nn.Module, n_steps: int = 50):
        self.model = model
        self.n_steps = n_steps
    
    def attribute(
        self,
        inputs: Dict[str, torch.Tensor],
        baselines: Optional[Dict[str, torch.Tensor]] = None,
        target_class: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute integrated gradients."""
        self.model.eval()
        device = next(self.model.parameters()).device
        
        if baselines is None:
            baselines = {k: torch.zeros_like(v) for k, v in inputs.items()}
        
        # Get target class
        with torch.no_grad():
            output = self.model(inputs)
            if isinstance(output, dict):
                logits = output['logits']
            else:
                logits = output
            if target_class is None:
                target_class = logits.argmax(dim=-1).item()
        
        attributions = {}
        
        for modality in inputs.keys():
            input_tensor = inputs[modality].clone().requires_grad_(True)
            baseline_tensor = baselines[modality]
            
            # Interpolate
            scaled_inputs = [
                baseline_tensor + (float(i) / self.n_steps) * (input_tensor - baseline_tensor)
                for i in range(self.n_steps + 1)
            ]
            
            # Compute gradients
            gradients = []
            for scaled_input in scaled_inputs:
                scaled_input = scaled_input.clone().detach().requires_grad_(True)
                modified_inputs = {k: v for k, v in inputs.items()}
                modified_inputs[modality] = scaled_input
                
                output = self.model(modified_inputs)
                if isinstance(output, dict):
                    score = output['logits'][0, target_class]
                else:
                    score = output[0, target_class]
                
                score.backward()
                gradients.append(scaled_input.grad.clone())
            
            # Average gradients
            avg_gradients = torch.stack(gradients).mean(dim=0)
            
            # Compute attribution
            attributions[modality] = (input_tensor - baseline_tensor) * avg_gradients
        
        return attributions
