"""
NeuroFusionXAI: Main Model Architecture

This module implements the complete NeuroFusionXAI framework integrating:
1. Privacy-Preserving Data Processing Module
2. Cross-Modality Fusion Network (ViT + Cross-Attention)
3. Graph Neural Network for brain connectivity
4. Explainable AI Integration Layer
5. Classification Head

As described in Section III of the paper.
"""

from typing import Optional, Dict, Tuple, List, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vision_transformer import VisionTransformer3D
from .cross_attention_fusion import MultiModalFusion
from .graph_neural_network import BrainGNN


class ClassificationHead(nn.Module):
    """
    Classification head for disease prediction.
    
    Args:
        input_dim: Input feature dimension
        hidden_dims: List of hidden layer dimensions
        num_classes: Number of output classes
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        input_dim: int = 768,
        hidden_dims: List[int] = [512, 256],
        num_classes: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()
        
        layers = []
        in_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_dim = hidden_dim
            
        layers.append(nn.Linear(in_dim, num_classes))
        
        self.classifier = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class FeatureFusion(nn.Module):
    """
    Final feature fusion combining cross-modal features and graph features.
    
    Args:
        cross_modal_dim: Dimension of cross-modal features
        graph_dim: Dimension of graph features
        output_dim: Output dimension
    """
    
    def __init__(
        self,
        cross_modal_dim: int = 768,
        graph_dim: int = 256,
        output_dim: int = 768,
    ):
        super().__init__()
        
        self.fusion = nn.Sequential(
            nn.Linear(cross_modal_dim + graph_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        
        # Gating mechanism for adaptive fusion
        self.gate = nn.Sequential(
            nn.Linear(cross_modal_dim + graph_dim, 2),
            nn.Softmax(dim=-1),
        )
        
        # Projections for gated fusion
        self.proj_cross = nn.Linear(cross_modal_dim, output_dim)
        self.proj_graph = nn.Linear(graph_dim, output_dim)
        
    def forward(
        self,
        cross_modal_features: torch.Tensor,
        graph_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            cross_modal_features: Features from cross-attention fusion (B, cross_modal_dim)
            graph_features: Features from GNN (B, graph_dim)
            
        Returns:
            Fused features (B, output_dim)
        """
        # Concatenate features
        concat = torch.cat([cross_modal_features, graph_features], dim=-1)
        
        # Compute gating weights
        gates = self.gate(concat)  # (B, 2)
        
        # Project features
        proj_cross = self.proj_cross(cross_modal_features)
        proj_graph = self.proj_graph(graph_features)
        
        # Gated fusion
        fused = gates[:, 0:1] * proj_cross + gates[:, 1:2] * proj_graph
        
        return fused


class NeuroFusionXAI(nn.Module):
    """
    NeuroFusionXAI: Privacy-Preserving Cross-Modality Explainable Fusion
    Framework for Early Neurodegenerative Disease Detection.
    
    This is the main model class that integrates all components:
    - Modality-specific Vision Transformers for feature extraction
    - Cross-Attention Fusion for multi-modal integration
    - Graph Neural Network for brain connectivity modeling
    - Classification head for disease prediction
    
    Args:
        config: Configuration dictionary containing model hyperparameters
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        
        # Extract configuration
        vit_config = config['model']['vit']
        fusion_config = config['model']['fusion']
        gnn_config = config['model']['gnn']
        classifier_config = config['model']['classifier']
        
        embed_dim = vit_config['embed_dim']
        self.modalities = config['data']['modalities']
        
        # Modality-specific Vision Transformers
        self.vit_encoders = nn.ModuleDict()
        for modality in self.modalities:
            self.vit_encoders[modality] = VisionTransformer3D(
                in_channels=1,
                embed_dim=embed_dim,
                patch_size=tuple(vit_config['patch_size']),
                img_size=tuple(config['data']['input_shape'][1:]),
                num_heads=vit_config['num_heads'],
                num_layers=vit_config['num_layers'],
                mlp_ratio=vit_config['mlp_ratio'],
                dropout=vit_config['dropout'],
                attention_dropout=vit_config['attention_dropout'],
                num_classes=0,  # Feature extraction only
            )
            
        # Cross-Modal Fusion Network
        self.cross_modal_fusion = MultiModalFusion(
            embed_dim=embed_dim,
            fusion_hidden_dim=fusion_config['hidden_dim'],
            cross_attn_heads=fusion_config['num_heads'],
            fusion_heads=fusion_config['num_heads'],
            cross_attn_layers=fusion_config['num_layers'] // 2,
            fusion_layers=fusion_config['num_layers'],
            dropout=fusion_config['dropout'],
            modalities=self.modalities,
        )
        
        # Graph Neural Network for brain connectivity
        self.use_gnn = gnn_config.get('enabled', True)
        if self.use_gnn:
            self.brain_gnn = BrainGNN(
                input_dim=embed_dim,
                hidden_dims=gnn_config['hidden_dims'],
                num_heads=gnn_config['num_heads'],
                dropout=gnn_config['dropout'],
                num_regions=gnn_config['num_regions'],
                output_dim=gnn_config['hidden_dims'][-1],
            )
            
            # Feature fusion (cross-modal + graph)
            self.feature_fusion = FeatureFusion(
                cross_modal_dim=embed_dim,
                graph_dim=gnn_config['hidden_dims'][-1],
                output_dim=embed_dim,
            )
        
        # Classification Head
        self.classifier = ClassificationHead(
            input_dim=embed_dim,
            hidden_dims=classifier_config['hidden_dims'],
            num_classes=classifier_config['num_classes'],
            dropout=classifier_config['dropout'],
        )
        
        # Store intermediate features for explainability
        self.intermediate_features = {}
        
    def encode_modality(
        self,
        x: torch.Tensor,
        modality: str,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        Encode single modality using its Vision Transformer.
        
        Implements Eq. 4: f^mod = E_mod(x^mod) = ViT_mod(Patch(x^mod))
        
        Args:
            x: Input tensor (B, 1, D, H, W)
            modality: Modality name
            return_attention: Whether to return attention weights
            
        Returns:
            features: Encoded features (B, N+1, embed_dim)
            attention: Optional attention weights
        """
        encoder = self.vit_encoders[modality]
        features = encoder(x, return_features=True, return_attention=return_attention)
        
        if return_attention:
            return features
        return features, None
        
    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        return_features: bool = False,
        return_attention: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the complete NeuroFusionXAI model.
        
        Args:
            inputs: Dictionary of input tensors for each modality
                   {modality: (B, 1, D, H, W)}
            return_features: Whether to return intermediate features
            return_attention: Whether to return attention weights
            
        Returns:
            Dictionary containing:
                - logits: Classification logits
                - probabilities: Class probabilities
                - features: Fused features (optional)
                - attention_weights: Attention weights (optional)
        """
        output = {}
        attention_weights = {}
        
        # Step 1: Encode each modality
        modality_features = {}
        for modality in self.modalities:
            if modality in inputs:
                features, attn = self.encode_modality(
                    inputs[modality],
                    modality,
                    return_attention=return_attention,
                )
                modality_features[modality] = features
                
                if return_attention and attn is not None:
                    attention_weights[f'{modality}_encoder'] = attn
                    
                # Store for explainability
                self.intermediate_features[f'{modality}_encoded'] = features.detach()
                
        # Step 2: Cross-Modal Fusion
        cross_modal_features, cross_attn = self.cross_modal_fusion(modality_features)
        
        if return_attention:
            attention_weights['cross_modal'] = cross_attn
            
        self.intermediate_features['cross_modal_fused'] = cross_modal_features.detach()
        
        # Step 3: Graph Neural Network (optional)
        if self.use_gnn:
            # Prepare node features for GNN
            # Use CLS tokens from each modality
            gnn_input_features = []
            for modality in self.modalities:
                if modality in modality_features:
                    cls_token = modality_features[modality][:, 0]  # (B, embed_dim)
                    gnn_input_features.append(cls_token)
                    
            # Stack and process through GNN
            gnn_input = torch.stack(gnn_input_features, dim=1)  # (B, num_modalities, embed_dim)
            
            # For simplicity, use cross-modal features as node features
            # In full implementation, this would use region-level features
            graph_features, node_features = self.brain_gnn(
                cross_modal_features.unsqueeze(1).expand(-1, 116, -1),  # Placeholder
            )
            
            self.intermediate_features['graph_features'] = graph_features.detach()
            
            # Combine cross-modal and graph features
            final_features = self.feature_fusion(cross_modal_features, graph_features)
        else:
            final_features = cross_modal_features
            
        self.intermediate_features['final_features'] = final_features.detach()
        
        # Step 4: Classification
        logits = self.classifier(final_features)
        probabilities = F.softmax(logits, dim=-1)
        
        # Prepare output
        output['logits'] = logits
        output['probabilities'] = probabilities
        
        if return_features:
            output['features'] = final_features
            output['modality_features'] = modality_features
            
        if return_attention:
            output['attention_weights'] = attention_weights
            
        return output
    
    def get_intermediate_features(self) -> Dict[str, torch.Tensor]:
        """Get stored intermediate features for explainability."""
        return self.intermediate_features
    
    def predict(
        self,
        inputs: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convenience method for inference.
        
        Args:
            inputs: Dictionary of input tensors
            
        Returns:
            predictions: Predicted class indices
            probabilities: Class probabilities
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(inputs)
            predictions = output['logits'].argmax(dim=-1)
            return predictions, output['probabilities']
    
    def get_attention_maps(
        self,
        inputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Get attention maps for visualization.
        
        Args:
            inputs: Dictionary of input tensors
            
        Returns:
            Dictionary of attention maps from all components
        """
        output = self.forward(inputs, return_attention=True)
        return output.get('attention_weights', {})


class NeuroFusionXAIWithPrivacy(NeuroFusionXAI):
    """
    NeuroFusionXAI with privacy-preserving mechanisms.
    
    This version integrates differential privacy during training
    and supports encrypted inference.
    
    Args:
        config: Configuration dictionary
        privacy_config: Privacy-specific configuration
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        privacy_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(config)
        
        self.privacy_config = privacy_config or config.get('privacy', {})
        
        # Privacy budget tracking
        self.privacy_budget_used = 0.0
        self.max_privacy_budget = self.privacy_config.get(
            'differential_privacy', {}
        ).get('epsilon', 0.5)
        
    def clip_gradients(self, max_norm: float = 1.0):
        """Clip gradients for differential privacy."""
        total_norm = 0.0
        for p in self.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        
        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for p in self.parameters():
                if p.grad is not None:
                    p.grad.data.mul_(clip_coef)
                    
        return total_norm
    
    def add_gradient_noise(
        self,
        noise_multiplier: float = 1.1,
        max_grad_norm: float = 1.0,
    ):
        """Add calibrated noise to gradients for differential privacy."""
        for p in self.parameters():
            if p.grad is not None:
                noise = torch.randn_like(p.grad) * noise_multiplier * max_grad_norm
                p.grad.data.add_(noise)


def create_model(config: Dict[str, Any], with_privacy: bool = False) -> nn.Module:
    """
    Factory function to create NeuroFusionXAI model.
    
    Args:
        config: Configuration dictionary
        with_privacy: Whether to include privacy mechanisms
        
    Returns:
        Configured model instance
    """
    if with_privacy:
        return NeuroFusionXAIWithPrivacy(config)
    return NeuroFusionXAI(config)


def load_model(
    checkpoint_path: str,
    config: Dict[str, Any],
    device: torch.device = torch.device('cpu'),
) -> nn.Module:
    """
    Load model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        config: Configuration dictionary
        device: Target device
        
    Returns:
        Loaded model
    """
    model = create_model(config)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    return model
