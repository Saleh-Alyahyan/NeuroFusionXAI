"""
SHAP Explainer for NeuroFusionXAI

This module implements SHapley Additive exPlanations (SHAP) for
computing feature attributions with game-theoretic foundations.

Implements Eq. 10:
φ_i(x) = Σ_{S⊆F\{i}} (|S|!(|F|-|S|-1)!/|F|!) · [f(S∪{i}) - f(S)]

Reference: "A Unified Approach to Interpreting Model Predictions" (Lundberg & Lee, 2017)
"""

from typing import Optional, Dict, List, Any
import math
import numpy as np
import torch
import torch.nn as nn


class SHAPExplainer:
    """
    SHAP explainer for NeuroFusionXAI.
    
    Args:
        model: Model to explain
        background_data: Background dataset for computing expectations
        max_evals: Maximum number of model evaluations
    """
    
    def __init__(
        self,
        model: nn.Module,
        background_data: Optional[Dict[str, torch.Tensor]] = None,
        max_evals: int = 500,
    ):
        self.model = model
        self.background_data = background_data
        self.max_evals = max_evals
        
        if background_data is not None:
            self.background_mean = {
                k: v.mean(dim=0, keepdim=True) for k, v in background_data.items()
            }
        else:
            self.background_mean = None
            
    def _shapley_kernel_weight(self, M: int, s: int) -> float:
        """Compute Shapley kernel weight."""
        if s == 0 or s == M:
            return float('inf')
        return (M - 1) / (math.comb(M, s) * s * (M - s))
    
    def _create_segments(self, shape, n_segments: int) -> np.ndarray:
        """Create 3D grid segments."""
        D, H, W = shape
        segment_size = int(np.cbrt(D * H * W / n_segments))
        segment_size = max(4, segment_size)
        
        segments = np.zeros(shape, dtype=np.int32)
        segment_id = 0
        
        for d in range(0, D, segment_size):
            for h in range(0, H, segment_size):
                for w in range(0, W, segment_size):
                    segments[d:min(d+segment_size, D),
                            h:min(h+segment_size, H),
                            w:min(w+segment_size, W)] = segment_id
                    segment_id += 1
        return segments
    
    def _apply_mask(self, original, baseline, segments, coalition) -> torch.Tensor:
        """Apply coalition mask to create mixed input."""
        result = baseline.clone()
        original_np = original.squeeze().cpu().numpy()
        result_np = result.squeeze().cpu().numpy()
        
        if len(original_np.shape) == 4:
            original_np = original_np[0]
            result_np = result_np[0]
        
        for seg_id, include in enumerate(coalition):
            if include:
                mask = segments == seg_id
                result_np[mask] = original_np[mask]
        
        return torch.from_numpy(result_np).unsqueeze(0).unsqueeze(0).float()
    
    def explain(
        self,
        inputs: Dict[str, torch.Tensor],
        target_class: Optional[int] = None,
        n_segments: int = 20,
    ) -> Dict[str, Any]:
        """Compute SHAP values for input."""
        self.model.eval()
        device = next(self.model.parameters()).device
        
        with torch.no_grad():
            output = self.model(inputs)
            if isinstance(output, dict):
                original_pred = output['probabilities']
            else:
                original_pred = torch.softmax(output, dim=-1)
            if target_class is None:
                target_class = original_pred.argmax(dim=-1).item()
            original_prob = original_pred[0, target_class].item()
        
        if self.background_mean is not None:
            baseline = {k: v.to(device) for k, v in self.background_mean.items()}
        else:
            baseline = {k: torch.zeros_like(v) for k, v in inputs.items()}
        
        with torch.no_grad():
            baseline_output = self.model(baseline)
            if isinstance(baseline_output, dict):
                baseline_prob = baseline_output['probabilities'][0, target_class].item()
            else:
                baseline_prob = torch.softmax(baseline_output, dim=-1)[0, target_class].item()
        
        shap_values = {}
        for modality in inputs.keys():
            shap_values[modality] = self._compute_shap_modality(
                inputs, baseline, modality, target_class, n_segments, device
            )
        
        return {
            'shap_values': shap_values,
            'expected_value': baseline_prob,
            'original_prediction': original_prob,
            'target_class': target_class,
        }
    
    def _compute_shap_modality(
        self, inputs, baseline, modality, target_class, n_segments, device
    ) -> np.ndarray:
        """Compute SHAP values for a single modality."""
        volume = inputs[modality].squeeze().cpu().numpy()
        if len(volume.shape) == 4:
            volume = volume[0]
        
        segments = self._create_segments(volume.shape, n_segments)
        n_actual_segments = segments.max() + 1
        shap_values = np.zeros(n_actual_segments)
        counts = np.zeros(n_actual_segments)
        
        n_samples = min(self.max_evals, 2 ** min(n_actual_segments, 10))
        
        for _ in range(n_samples):
            coalition = np.random.binomial(1, 0.5, n_actual_segments)
            if coalition.sum() == 0 or coalition.sum() == n_actual_segments:
                continue
            
            masked_input = self._apply_mask(
                inputs[modality], baseline[modality], segments, coalition
            ).to(device)
            
            with torch.no_grad():
                modified_inputs = {k: v for k, v in inputs.items()}
                modified_inputs[modality] = masked_input
                output = self.model(modified_inputs)
                if isinstance(output, dict):
                    prob_with = output['probabilities'][0, target_class].item()
                else:
                    prob_with = torch.softmax(output, dim=-1)[0, target_class].item()
            
            for i in range(n_actual_segments):
                if coalition[i] == 1:
                    coalition_without = coalition.copy()
                    coalition_without[i] = 0
                    
                    masked_without = self._apply_mask(
                        inputs[modality], baseline[modality], segments, coalition_without
                    ).to(device)
                    
                    with torch.no_grad():
                        modified_inputs[modality] = masked_without
                        output_without = self.model(modified_inputs)
                        if isinstance(output_without, dict):
                            prob_without = output_without['probabilities'][0, target_class].item()
                        else:
                            prob_without = torch.softmax(output_without, dim=-1)[0, target_class].item()
                    
                    shap_values[i] += prob_with - prob_without
                    counts[i] += 1
        
        counts[counts == 0] = 1
        return shap_values / counts
    
    def get_shap_map(self, shap_result: Dict, modality: str, original_shape) -> np.ndarray:
        """Convert segment SHAP values to voxel-level map."""
        segments = self._create_segments(original_shape, len(shap_result['shap_values'][modality]))
        shap_map = np.zeros(original_shape, dtype=np.float32)
        
        for seg_id, value in enumerate(shap_result['shap_values'][modality]):
            shap_map[segments == seg_id] = value
        
        return shap_map
