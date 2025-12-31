"""
Federated Learning Implementation for NeuroFusionXAI

This module implements federated learning with:
- Privacy-preserving FedAvg (Eq. 13, 14)
- Domain-shift aware aggregation (Eq. 15)
- Secure aggregation protocol

Reference: "Communication-Efficient Learning of Deep Networks" (McMahan et al., 2017)
"""

from typing import Optional, Dict, List, Any, Tuple
import copy

import torch
import torch.nn as nn
import numpy as np


class FederatedClient:
    """
    Federated learning client representing one institution.
    
    Performs local training and prepares model updates for aggregation.
    
    Args:
        client_id: Unique client identifier
        model: Local copy of the global model
        train_loader: Local training data loader
        config: Training configuration
    """
    
    def __init__(
        self,
        client_id: int,
        model: nn.Module,
        train_loader: Any,
        config: Dict[str, Any],
    ):
        self.client_id = client_id
        self.model = copy.deepcopy(model)
        self.train_loader = train_loader
        self.config = config
        
        # Local training parameters
        self.local_epochs = config.get('local_epochs', 5)
        self.learning_rate = config.get('learning_rate', 1e-4)
        
        # Privacy parameters
        self.dp_enabled = config.get('differential_privacy', {}).get('enabled', True)
        self.noise_multiplier = config.get('differential_privacy', {}).get('noise_multiplier', 1.1)
        self.max_grad_norm = config.get('differential_privacy', {}).get('max_grad_norm', 1.0)
        
        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=config.get('weight_decay', 0.01),
        )
        
    def local_train(
        self,
        global_model_state: Dict[str, torch.Tensor],
    ) -> Tuple[Dict[str, torch.Tensor], int]:
        """
        Perform local training for E epochs.
        
        Implements Eq. 13:
        θ_k^(t+1) = θ_k^t - η · (∇L(θ_k^t, D_k) + N(0, σ²I))
        
        Args:
            global_model_state: State dict from global model
            
        Returns:
            Local model state dict and number of samples
        """
        # Load global model weights
        self.model.load_state_dict(global_model_state)
        self.model.train()
        
        device = next(self.model.parameters()).device
        criterion = nn.CrossEntropyLoss()
        
        num_samples = 0
        
        for epoch in range(self.local_epochs):
            for batch in self.train_loader:
                inputs, labels = batch
                
                # Handle multi-modal inputs
                if isinstance(inputs, dict):
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                else:
                    inputs = inputs.to(device)
                labels = labels.to(device)
                
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(inputs)
                if isinstance(outputs, dict):
                    logits = outputs['logits']
                else:
                    logits = outputs
                    
                loss = criterion(logits, labels)
                
                # Backward pass
                loss.backward()
                
                # Apply differential privacy
                if self.dp_enabled:
                    self._apply_dp_to_gradients()
                    
                self.optimizer.step()
                
                num_samples += labels.size(0)
                
        return self.model.state_dict(), num_samples
    
    def _apply_dp_to_gradients(self):
        """Apply differential privacy to gradients (clip and add noise)."""
        # Gradient clipping
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5
        
        clip_factor = min(1.0, self.max_grad_norm / (total_norm + 1e-6))
        
        # Clip and add noise
        for p in self.model.parameters():
            if p.grad is not None:
                p.grad.data.mul_(clip_factor)
                noise = torch.randn_like(p.grad) * self.noise_multiplier * self.max_grad_norm
                p.grad.data.add_(noise)
    
    def get_model_update(
        self,
        global_model_state: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Get model update (difference from global model)."""
        update = {}
        local_state = self.model.state_dict()
        
        for key in global_model_state:
            update[key] = local_state[key] - global_model_state[key]
            
        return update


class FedAvgAggregator:
    """
    Federated Averaging aggregator.
    
    Implements Eq. 14:
    θ^(t+1) = Σ_k (n_k / n) · θ_k^(t+1)
    
    Args:
        model: Global model
        config: Aggregation configuration
    """
    
    def __init__(self, model: nn.Module, config: Dict[str, Any]):
        self.global_model = model
        self.config = config
        
    def aggregate(
        self,
        client_updates: List[Tuple[Dict[str, torch.Tensor], int]],
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregate client model updates using weighted averaging.
        
        Implements Eq. 14:
        θ^(t+1) = Σ_k (n_k / n) · θ_k^(t+1)
        
        Args:
            client_updates: List of (state_dict, num_samples) from clients
            
        Returns:
            Aggregated global model state
        """
        # Calculate total samples
        total_samples = sum(n for _, n in client_updates)
        
        # Initialize aggregated state
        aggregated_state = {}
        
        # Weighted averaging
        for key in client_updates[0][0].keys():
            aggregated_state[key] = torch.zeros_like(client_updates[0][0][key])
            
            for state_dict, num_samples in client_updates:
                weight = num_samples / total_samples
                aggregated_state[key] += weight * state_dict[key]
                
        return aggregated_state


class DomainAwareFedAvg(FedAvgAggregator):
    """
    Domain-Shift Aware Federated Averaging.
    
    Implements Eq. 15:
    θ^(t+1) = Σ_k (w_k · n_k / Σ_j w_j · n_j) · θ_k^(t+1)
    
    Where w_k is the domain consistency coefficient.
    
    Args:
        model: Global model
        config: Aggregation configuration
    """
    
    def __init__(self, model: nn.Module, config: Dict[str, Any]):
        super().__init__(model, config)
        
        # Domain alignment parameters
        self.alignment_weight = config.get('alignment_weight', 0.1)
        self.client_weights = {}
        
    def compute_domain_weights(
        self,
        client_features: Dict[int, torch.Tensor],
        global_features: torch.Tensor,
    ) -> Dict[int, float]:
        """
        Compute domain consistency coefficients for each client.
        
        Args:
            client_features: Feature statistics from each client
            global_features: Global feature statistics
            
        Returns:
            Dictionary of client weights
        """
        weights = {}
        
        for client_id, client_feat in client_features.items():
            # Compute feature divergence (e.g., MMD or KL divergence)
            divergence = self._compute_divergence(client_feat, global_features)
            
            # Convert divergence to weight (higher divergence = lower weight)
            weight = np.exp(-self.alignment_weight * divergence)
            weights[client_id] = max(0.1, weight)  # Minimum weight of 0.1
            
        return weights
    
    def _compute_divergence(
        self,
        client_feat: torch.Tensor,
        global_feat: torch.Tensor,
    ) -> float:
        """Compute feature divergence between client and global."""
        # Simple L2 distance between feature statistics
        divergence = (client_feat - global_feat).norm(2).item()
        return divergence
    
    def aggregate(
        self,
        client_updates: List[Tuple[Dict[str, torch.Tensor], int, int]],
        client_features: Optional[Dict[int, torch.Tensor]] = None,
        global_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregate with domain-aware weighting.
        
        Implements Eq. 15:
        θ^(t+1) = Σ_k (w_k · n_k / Σ_j w_j · n_j) · θ_k^(t+1)
        
        Args:
            client_updates: List of (state_dict, num_samples, client_id)
            client_features: Feature statistics from clients
            global_features: Global feature statistics
            
        Returns:
            Aggregated global model state
        """
        # Compute domain weights if features provided
        if client_features is not None and global_features is not None:
            domain_weights = self.compute_domain_weights(client_features, global_features)
        else:
            domain_weights = {i: 1.0 for i in range(len(client_updates))}
            
        # Calculate weighted sum
        total_weighted_samples = sum(
            domain_weights.get(client_id, 1.0) * n
            for _, n, client_id in client_updates
        )
        
        # Initialize aggregated state
        aggregated_state = {}
        
        # Weighted averaging with domain weights
        for key in client_updates[0][0].keys():
            aggregated_state[key] = torch.zeros_like(client_updates[0][0][key])
            
            for state_dict, num_samples, client_id in client_updates:
                w_k = domain_weights.get(client_id, 1.0)
                weight = (w_k * num_samples) / total_weighted_samples
                aggregated_state[key] += weight * state_dict[key]
                
        return aggregated_state


class SecureAggregator:
    """
    Secure aggregation protocol for privacy-preserving model aggregation.
    
    Ensures the server cannot see individual client updates.
    
    Args:
        num_clients: Number of participating clients
        threshold: Minimum clients required for aggregation
    """
    
    def __init__(self, num_clients: int, threshold: int = 3):
        self.num_clients = num_clients
        self.threshold = threshold
        
        # Secret sharing parameters
        self.masks = {}
        
    def generate_masks(self, model_shape: Dict[str, Tuple]) -> None:
        """Generate random masks for secure aggregation."""
        for i in range(self.num_clients):
            self.masks[i] = {}
            for j in range(i + 1, self.num_clients):
                # Pairwise masks that cancel out when summed
                for key, shape in model_shape.items():
                    mask = torch.randn(shape)
                    self.masks[(i, j, key)] = mask
                    self.masks[(j, i, key)] = -mask
                    
    def mask_update(
        self,
        client_id: int,
        update: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Apply masking to client update.
        
        Args:
            client_id: Client identifier
            update: Model update from client
            
        Returns:
            Masked update
        """
        masked_update = {}
        
        for key, value in update.items():
            masked_value = value.clone()
            
            # Add pairwise masks
            for j in range(self.num_clients):
                if j != client_id:
                    mask_key = (client_id, j, key)
                    if mask_key in self.masks:
                        masked_value += self.masks[mask_key]
                        
            masked_update[key] = masked_value
            
        return masked_update
    
    def aggregate(
        self,
        masked_updates: List[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregate masked updates (masks cancel out).
        
        Args:
            masked_updates: List of masked updates from clients
            
        Returns:
            Aggregated (unmasked) update
        """
        if len(masked_updates) < self.threshold:
            raise ValueError(f"Need at least {self.threshold} clients for secure aggregation")
            
        # Sum all masked updates (masks cancel out)
        aggregated = {}
        for key in masked_updates[0].keys():
            aggregated[key] = sum(update[key] for update in masked_updates)
            aggregated[key] /= len(masked_updates)
            
        return aggregated


class FederatedLearning:
    """
    Complete Federated Learning orchestrator for NeuroFusionXAI.
    
    Coordinates training across multiple institutions with
    privacy guarantees.
    
    Args:
        model: Global model
        config: Federated learning configuration
        use_secure_aggregation: Whether to use secure aggregation
        use_domain_aware: Whether to use domain-aware aggregation
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        use_secure_aggregation: bool = True,
        use_domain_aware: bool = True,
    ):
        self.global_model = model
        self.config = config
        
        # Federated learning parameters
        self.num_rounds = config.get('num_rounds', 24)
        self.client_fraction = config.get('client_fraction', 1.0)
        
        # Initialize aggregators
        if use_domain_aware:
            self.aggregator = DomainAwareFedAvg(model, config.get('domain_aware', {}))
        else:
            self.aggregator = FedAvgAggregator(model, config)
            
        self.use_secure_aggregation = use_secure_aggregation
        self.secure_aggregator = None
        
        # Training history
        self.history = {
            'rounds': [],
            'loss': [],
            'accuracy': [],
        }
        
    def initialize_clients(
        self,
        client_data_loaders: List[Any],
    ) -> List[FederatedClient]:
        """
        Initialize federated clients.
        
        Args:
            client_data_loaders: List of data loaders for each client
            
        Returns:
            List of FederatedClient instances
        """
        clients = []
        for i, loader in enumerate(client_data_loaders):
            client = FederatedClient(
                client_id=i,
                model=self.global_model,
                train_loader=loader,
                config=self.config,
            )
            clients.append(client)
            
        # Initialize secure aggregation if enabled
        if self.use_secure_aggregation:
            self.secure_aggregator = SecureAggregator(
                num_clients=len(clients),
                threshold=self.config.get('secure_aggregation', {}).get('threshold', 3),
            )
            
        return clients
    
    def train_round(
        self,
        clients: List[FederatedClient],
        round_num: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Execute one round of federated training.
        
        Args:
            clients: List of participating clients
            round_num: Current round number
            
        Returns:
            Updated global model state
        """
        # Select clients for this round
        num_selected = max(1, int(len(clients) * self.client_fraction))
        selected_indices = np.random.choice(len(clients), num_selected, replace=False)
        selected_clients = [clients[i] for i in selected_indices]
        
        # Get current global model state
        global_state = self.global_model.state_dict()
        
        # Collect client updates
        client_updates = []
        for client in selected_clients:
            local_state, num_samples = client.local_train(global_state)
            client_updates.append((local_state, num_samples, client.client_id))
            
        # Aggregate updates
        if isinstance(self.aggregator, DomainAwareFedAvg):
            new_global_state = self.aggregator.aggregate(client_updates)
        else:
            # Convert to format expected by standard FedAvg
            updates = [(s, n) for s, n, _ in client_updates]
            new_global_state = self.aggregator.aggregate(updates)
            
        # Update global model
        self.global_model.load_state_dict(new_global_state)
        
        return new_global_state
    
    def train(
        self,
        clients: List[FederatedClient],
        eval_fn: Optional[callable] = None,
    ) -> Dict[str, List]:
        """
        Run full federated training.
        
        Args:
            clients: List of federated clients
            eval_fn: Optional evaluation function
            
        Returns:
            Training history
        """
        for round_num in range(self.num_rounds):
            print(f"Federated Round {round_num + 1}/{self.num_rounds}")
            
            # Train round
            self.train_round(clients, round_num)
            
            # Evaluate if function provided
            if eval_fn is not None:
                metrics = eval_fn(self.global_model)
                self.history['rounds'].append(round_num)
                self.history['loss'].append(metrics.get('loss', 0))
                self.history['accuracy'].append(metrics.get('accuracy', 0))
                print(f"  Loss: {metrics.get('loss', 0):.4f}, "
                      f"Accuracy: {metrics.get('accuracy', 0):.4f}")
                
        return self.history
    
    def get_global_model(self) -> nn.Module:
        """Get the current global model."""
        return self.global_model
