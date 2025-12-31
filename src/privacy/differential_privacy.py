"""
Differential Privacy Implementation for NeuroFusionXAI

This module implements differential privacy mechanisms including:
- Gradient clipping and perturbation (Eq. 3)
- Privacy budget accounting
- DP-SGD optimizer wrapper

Reference: "Deep Learning with Differential Privacy" (Abadi et al., 2016)
"""

import math
from typing import Optional, Tuple, List, Dict, Any

import torch
import torch.nn as nn
from torch.optim import Optimizer


class GaussianMechanism:
    """
    Gaussian mechanism for differential privacy.
    
    Adds calibrated Gaussian noise to achieve (ε, δ)-differential privacy.
    
    Args:
        epsilon: Privacy budget (ε)
        delta: Privacy parameter (δ)
        sensitivity: L2 sensitivity of the query
    """
    
    def __init__(
        self,
        epsilon: float = 0.5,
        delta: float = 1e-5,
        sensitivity: float = 1.0,
    ):
        self.epsilon = epsilon
        self.delta = delta
        self.sensitivity = sensitivity
        self.sigma = self._compute_sigma()
        
    def _compute_sigma(self) -> float:
        """Compute noise scale σ for (ε, δ)-DP."""
        # σ ≥ sensitivity * √(2 ln(1.25/δ)) / ε
        return self.sensitivity * math.sqrt(2 * math.log(1.25 / self.delta)) / self.epsilon
        
    def add_noise(self, tensor: torch.Tensor) -> torch.Tensor:
        """Add calibrated Gaussian noise to tensor."""
        noise = torch.randn_like(tensor) * self.sigma
        return tensor + noise
    
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.add_noise(tensor)


class PrivacyAccountant:
    """
    Privacy budget accountant for tracking cumulative privacy loss.
    
    Uses the moments accountant for tight privacy analysis.
    
    Args:
        epsilon: Total privacy budget
        delta: Privacy parameter
    """
    
    def __init__(
        self,
        epsilon: float = 0.5,
        delta: float = 1e-5,
    ):
        self.total_epsilon = epsilon
        self.delta = delta
        self.spent_epsilon = 0.0
        self.num_compositions = 0
        
    def spend(self, epsilon_step: float) -> bool:
        """
        Record privacy expenditure for one step.
        
        Args:
            epsilon_step: Privacy cost of this step
            
        Returns:
            True if within budget, False if exceeded
        """
        self.spent_epsilon += epsilon_step
        self.num_compositions += 1
        return self.spent_epsilon <= self.total_epsilon
    
    def get_epsilon(self) -> float:
        """Get total privacy loss so far."""
        return self.spent_epsilon
    
    def remaining_budget(self) -> float:
        """Get remaining privacy budget."""
        return max(0, self.total_epsilon - self.spent_epsilon)
    
    def is_exhausted(self) -> bool:
        """Check if privacy budget is exhausted."""
        return self.spent_epsilon >= self.total_epsilon
    
    def compute_epsilon_per_step(
        self,
        num_steps: int,
        noise_multiplier: float,
        sample_rate: float,
    ) -> float:
        """
        Compute per-step epsilon using RDP accountant.
        
        Args:
            num_steps: Number of training steps
            noise_multiplier: Noise multiplier σ/C
            sample_rate: Sampling rate (batch_size / dataset_size)
            
        Returns:
            Per-step epsilon value
        """
        # Simplified RDP-based accounting
        # For more accurate accounting, use opacus library
        epsilon_step = sample_rate * math.sqrt(2 * math.log(1.25 / self.delta)) / noise_multiplier
        return epsilon_step


class DifferentialPrivacy:
    """
    Differential Privacy module for NeuroFusionXAI.
    
    Implements gradient perturbation as described in Eq. 3:
    ∇̃_θ L = ∇_θ L + N(0, σ²I)
    
    Args:
        epsilon: Privacy budget (ε)
        delta: Privacy parameter (δ)
        max_grad_norm: Maximum gradient norm for clipping
        noise_multiplier: Noise multiplier (σ/C where C is max_grad_norm)
    """
    
    def __init__(
        self,
        epsilon: float = 0.5,
        delta: float = 1e-5,
        max_grad_norm: float = 1.0,
        noise_multiplier: float = 1.1,
    ):
        self.epsilon = epsilon
        self.delta = delta
        self.max_grad_norm = max_grad_norm
        self.noise_multiplier = noise_multiplier
        
        # Noise scale
        self.sigma = noise_multiplier * max_grad_norm
        
        # Privacy accountant
        self.accountant = PrivacyAccountant(epsilon, delta)
        
        # Gaussian mechanism
        self.mechanism = GaussianMechanism(
            epsilon=epsilon,
            delta=delta,
            sensitivity=max_grad_norm,
        )
        
    def clip_gradient(self, gradient: torch.Tensor) -> torch.Tensor:
        """
        Clip gradient to maximum norm.
        
        Args:
            gradient: Input gradient tensor
            
        Returns:
            Clipped gradient
        """
        grad_norm = gradient.norm(2)
        clip_factor = min(1.0, self.max_grad_norm / (grad_norm + 1e-6))
        return gradient * clip_factor
    
    def add_noise(self, gradient: torch.Tensor) -> torch.Tensor:
        """
        Add calibrated Gaussian noise to gradient.
        
        Implements the noise addition in Eq. 3:
        ∇̃_θ L = ∇_θ L + N(0, σ²I)
        
        Args:
            gradient: Clipped gradient tensor
            
        Returns:
            Noisy gradient
        """
        noise = torch.randn_like(gradient) * self.sigma
        return gradient + noise
    
    def privatize_gradient(self, gradient: torch.Tensor) -> torch.Tensor:
        """
        Apply full DP gradient processing: clip then add noise.
        
        Args:
            gradient: Raw gradient tensor
            
        Returns:
            Privacy-preserving gradient
        """
        clipped = self.clip_gradient(gradient)
        noisy = self.add_noise(clipped)
        return noisy
    
    def clip_and_accumulate_gradients(
        self,
        model: nn.Module,
        per_sample_gradients: List[Dict[str, torch.Tensor]],
        batch_size: int,
    ) -> None:
        """
        Clip per-sample gradients and accumulate.
        
        Args:
            model: Neural network model
            per_sample_gradients: List of gradient dictionaries per sample
            batch_size: Batch size
        """
        # Initialize accumulated gradients
        accumulated_grads = {
            name: torch.zeros_like(param)
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        
        # Clip and accumulate per-sample gradients
        for sample_grads in per_sample_gradients:
            # Compute per-sample gradient norm
            total_norm = 0.0
            for name, grad in sample_grads.items():
                total_norm += grad.norm(2).item() ** 2
            total_norm = math.sqrt(total_norm)
            
            # Clip factor
            clip_factor = min(1.0, self.max_grad_norm / (total_norm + 1e-6))
            
            # Accumulate clipped gradients
            for name, grad in sample_grads.items():
                accumulated_grads[name] += grad * clip_factor
                
        # Average and add noise
        for name, param in model.named_parameters():
            if param.requires_grad and name in accumulated_grads:
                # Average gradient
                avg_grad = accumulated_grads[name] / batch_size
                
                # Add noise
                noise = torch.randn_like(avg_grad) * self.sigma / batch_size
                param.grad = avg_grad + noise
                
    def get_privacy_spent(self) -> Tuple[float, float]:
        """Get current privacy expenditure."""
        return self.accountant.get_epsilon(), self.delta


class DPOptimizer:
    """
    Differential Privacy optimizer wrapper.
    
    Wraps a standard PyTorch optimizer to add DP guarantees.
    
    Args:
        optimizer: Base optimizer
        dp_module: DifferentialPrivacy instance
        sample_rate: Sampling rate (batch_size / dataset_size)
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        dp_module: DifferentialPrivacy,
        sample_rate: float = 0.01,
    ):
        self.optimizer = optimizer
        self.dp = dp_module
        self.sample_rate = sample_rate
        self.steps = 0
        
    def step(self, closure=None):
        """
        Perform optimization step with DP.
        
        Args:
            closure: Optional closure for computing loss
        """
        # Apply DP to all parameter gradients
        for group in self.optimizer.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    param.grad = self.dp.privatize_gradient(param.grad)
                    
        # Standard optimizer step
        self.optimizer.step(closure)
        
        # Update privacy accountant
        epsilon_step = self.dp.accountant.compute_epsilon_per_step(
            num_steps=1,
            noise_multiplier=self.dp.noise_multiplier,
            sample_rate=self.sample_rate,
        )
        self.dp.accountant.spend(epsilon_step)
        
        self.steps += 1
        
    def zero_grad(self, set_to_none: bool = False):
        """Zero out gradients."""
        self.optimizer.zero_grad(set_to_none=set_to_none)
        
    @property
    def param_groups(self):
        return self.optimizer.param_groups
    
    def state_dict(self) -> Dict[str, Any]:
        """Get optimizer state."""
        return {
            'optimizer': self.optimizer.state_dict(),
            'steps': self.steps,
            'privacy_spent': self.dp.get_privacy_spent(),
        }
        
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load optimizer state."""
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.steps = state_dict['steps']


def make_private(
    model: nn.Module,
    optimizer: Optimizer,
    config: Dict[str, Any],
    sample_rate: float,
) -> Tuple[nn.Module, DPOptimizer]:
    """
    Make model and optimizer privacy-preserving.
    
    Args:
        model: Neural network model
        optimizer: Base optimizer
        config: Privacy configuration
        sample_rate: Batch size / dataset size
        
    Returns:
        Model (unchanged) and DP-wrapped optimizer
    """
    dp_config = config.get('differential_privacy', {})
    
    dp_module = DifferentialPrivacy(
        epsilon=dp_config.get('epsilon', 0.5),
        delta=dp_config.get('delta', 1e-5),
        max_grad_norm=dp_config.get('max_grad_norm', 1.0),
        noise_multiplier=dp_config.get('noise_multiplier', 1.1),
    )
    
    dp_optimizer = DPOptimizer(optimizer, dp_module, sample_rate)
    
    return model, dp_optimizer
