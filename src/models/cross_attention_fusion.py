"""
Cross-Modality Fusion Network

This module implements the cross-attention fusion mechanism for combining
features from multiple neuroimaging modalities (sMRI, fMRI, PET) as
described in Section III.C.2 of the NeuroFusionXAI paper.

Key equations implemented:
- Eq. 5: f^{i→j} = MultiHead(f^i, f^j, f^j)
- Eq. 6: f_fused = FusionTransformer([f^{sMRI→fMRI}, f^{sMRI→PET}, f^{fMRI→PET}, ...])
"""

from typing import Optional, Tuple, List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class CrossAttention(nn.Module):
    """
    Cross-Attention mechanism for fusing features from two modalities.
    
    Implements Eq. 5: f^{i→j} = MultiHead(f^i, f^j, f^j)
    where Q comes from modality i and K,V come from modality j.
    
    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        dropout: Dropout rate
        qkv_bias: Whether to use bias in QKV projections
    """
    
    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 16,
        dropout: float = 0.1,
        qkv_bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        # Query projection (from source modality)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        
        # Key and Value projections (from target modality)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        
        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: Query features from source modality (B, N_q, embed_dim)
            key_value: Key/Value features from target modality (B, N_kv, embed_dim)
            attention_mask: Optional mask for attention
            
        Returns:
            output: Cross-attended features (B, N_q, embed_dim)
            attention_weights: Attention weights (B, heads, N_q, N_kv)
        """
        B, N_q, C = query.shape
        N_kv = key_value.shape[1]
        
        # Project Q, K, V
        q = self.q_proj(query).reshape(B, N_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key_value).reshape(B, N_kv, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(key_value).reshape(B, N_kv, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        if attention_mask is not None:
            attn = attn.masked_fill(attention_mask == 0, float('-inf'))
            
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        
        # Apply attention to values
        out = (attn @ v).transpose(1, 2).reshape(B, N_q, C)
        out = self.out_proj(out)
        out = self.proj_dropout(out)
        
        return out, attn


class CrossAttentionBlock(nn.Module):
    """
    Cross-Attention block with residual connections and layer normalization.
    
    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        mlp_ratio: MLP hidden dimension ratio
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # Layer norms
        self.norm1_q = nn.LayerNorm(embed_dim)
        self.norm1_kv = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Cross-attention
        self.cross_attn = CrossAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        
        # MLP
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )
        
    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: Query features (B, N_q, embed_dim)
            key_value: Key/Value features (B, N_kv, embed_dim)
            
        Returns:
            output: Updated query features (B, N_q, embed_dim)
            attention_weights: Attention weights
        """
        # Cross-attention with residual
        normed_q = self.norm1_q(query)
        normed_kv = self.norm1_kv(key_value)
        attn_out, attn_weights = self.cross_attn(normed_q, normed_kv)
        query = query + attn_out
        
        # MLP with residual
        query = query + self.mlp(self.norm2(query))
        
        return query, attn_weights


class ModalityPairFusion(nn.Module):
    """
    Fusion module for a single pair of modalities.
    
    Applies bidirectional cross-attention between two modalities
    and combines the results.
    
    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        num_layers: Number of cross-attention layers
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 16,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # Forward cross-attention (modality_i attends to modality_j)
        self.forward_blocks = nn.ModuleList([
            CrossAttentionBlock(embed_dim, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        # Backward cross-attention (modality_j attends to modality_i)
        self.backward_blocks = nn.ModuleList([
            CrossAttentionBlock(embed_dim, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        # Combination layer
        self.combine = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )
        
    def forward(
        self,
        feat_i: torch.Tensor,
        feat_j: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            feat_i: Features from modality i (B, N, embed_dim)
            feat_j: Features from modality j (B, N, embed_dim)
            
        Returns:
            fused: Fused features (B, N, embed_dim)
            attention_weights: Dictionary of attention weights
        """
        attn_weights = {}
        
        # Forward: i attends to j
        f_i = feat_i
        for idx, block in enumerate(self.forward_blocks):
            f_i, attn = block(f_i, feat_j)
            attn_weights[f'forward_{idx}'] = attn
            
        # Backward: j attends to i
        f_j = feat_j
        for idx, block in enumerate(self.backward_blocks):
            f_j, attn = block(f_j, feat_i)
            attn_weights[f'backward_{idx}'] = attn
            
        # Combine bidirectional features
        # Take CLS tokens or mean pool
        f_i_cls = f_i[:, 0]  # (B, embed_dim)
        f_j_cls = f_j[:, 0]  # (B, embed_dim)
        
        fused = self.combine(torch.cat([f_i_cls, f_j_cls], dim=-1))
        
        return fused, attn_weights


class CrossAttentionFusion(nn.Module):
    """
    Cross-Modality Fusion Network using multi-head cross-attention.
    
    This module combines features from all modality pairs using
    bidirectional cross-attention as described in Eq. 5 and Eq. 6.
    
    Args:
        embed_dim: Embedding dimension from ViT encoders
        num_heads: Number of attention heads
        num_layers: Number of cross-attention layers per pair
        dropout: Dropout rate
        modalities: List of modality names
    """
    
    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 16,
        num_layers: int = 8,
        dropout: float = 0.1,
        modalities: List[str] = ['smri', 'fmri', 'pet'],
    ):
        super().__init__()
        self.modalities = modalities
        self.num_modalities = len(modalities)
        
        # Create fusion modules for each modality pair
        self.pair_fusion = nn.ModuleDict()
        for i, mod_i in enumerate(modalities):
            for j, mod_j in enumerate(modalities):
                if i < j:  # Only create for unique pairs
                    pair_name = f'{mod_i}_{mod_j}'
                    self.pair_fusion[pair_name] = ModalityPairFusion(
                        embed_dim=embed_dim,
                        num_heads=num_heads,
                        num_layers=num_layers // 2,  # Half layers per direction
                        dropout=dropout,
                    )
                    
        # Number of pairs
        self.num_pairs = len(self.pair_fusion)
        
        # Final fusion layer
        self.final_fusion = nn.Sequential(
            nn.Linear(embed_dim * self.num_pairs, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
    def forward(
        self,
        modality_features: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            modality_features: Dictionary mapping modality name to features
                              {modality: (B, N, embed_dim)}
                              
        Returns:
            fused_features: Combined features (B, embed_dim)
            attention_weights: Dictionary of attention weights from all pairs
        """
        pair_features = []
        all_attn_weights = {}
        
        # Process each modality pair
        for pair_name, fusion_module in self.pair_fusion.items():
            mod_i, mod_j = pair_name.split('_')
            feat_i = modality_features[mod_i]
            feat_j = modality_features[mod_j]
            
            fused, attn = fusion_module(feat_i, feat_j)
            pair_features.append(fused)
            
            for key, value in attn.items():
                all_attn_weights[f'{pair_name}_{key}'] = value
                
        # Concatenate all pair features
        combined = torch.cat(pair_features, dim=-1)  # (B, embed_dim * num_pairs)
        
        # Final fusion
        output = self.final_fusion(combined)  # (B, embed_dim)
        
        return output, all_attn_weights


class FusionTransformerBlock(nn.Module):
    """
    Transformer block for the Fusion Transformer that processes
    concatenated cross-attention outputs.
    
    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        mlp_ratio: MLP hidden dimension ratio
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        embed_dim: int = 1024,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # Self-attention
        self.norm1 = nn.LayerNorm(embed_dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # MLP
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (B, N, embed_dim)
            
        Returns:
            Output tensor (B, N, embed_dim)
        """
        # Self-attention with residual
        normed = self.norm1(x)
        attn_out, _ = self.self_attn(normed, normed, normed)
        x = x + self.dropout(attn_out)
        
        # MLP with residual
        x = x + self.mlp(self.norm2(x))
        
        return x


class FusionTransformer(nn.Module):
    """
    Fusion Transformer for processing concatenated cross-modality features.
    
    Implements Eq. 6:
    f_fused = FusionTransformer([f^{sMRI→fMRI}, f^{sMRI→PET}, f^{fMRI→PET}, ...])
    
    Args:
        input_dim: Input dimension (embed_dim from ViT)
        hidden_dim: Hidden dimension for transformer
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        dropout: Dropout rate
        num_modality_pairs: Number of modality pairs
    """
    
    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 1024,
        num_layers: int = 8,
        num_heads: int = 16,
        dropout: float = 0.1,
        num_modality_pairs: int = 3,  # For 3 modalities: 3 pairs
    ):
        super().__init__()
        
        # Input projection
        self.input_proj = nn.Linear(input_dim * num_modality_pairs, hidden_dim)
        
        # Positional encoding for modality pairs
        self.pos_embed = nn.Parameter(torch.zeros(1, num_modality_pairs, hidden_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            FusionTransformerBlock(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        
        # Final layer norm
        self.norm = nn.LayerNorm(hidden_dim)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, input_dim)
        
    def forward(
        self,
        pair_features: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            pair_features: List of features from each modality pair
                          [(B, embed_dim), ...]
                          
        Returns:
            fused_features: Final fused representation (B, embed_dim)
        """
        # Stack pair features: (B, num_pairs, embed_dim)
        x = torch.stack(pair_features, dim=1)
        B, N, D = x.shape
        
        # Flatten and project
        x_flat = x.reshape(B, -1)  # (B, num_pairs * embed_dim)
        x = self.input_proj(x_flat).unsqueeze(1)  # (B, 1, hidden_dim)
        
        # Alternatively, process as sequence
        # This is more aligned with transformer architecture
        x = x.expand(-1, N, -1)  # (B, num_pairs, hidden_dim)
        x = x + self.pos_embed
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
            
        # Final normalization
        x = self.norm(x)
        
        # Global average pooling and output projection
        x = x.mean(dim=1)  # (B, hidden_dim)
        x = self.output_proj(x)  # (B, embed_dim)
        
        return x


class MultiModalFusion(nn.Module):
    """
    Complete Multi-Modal Fusion module combining cross-attention
    and fusion transformer.
    
    This is the main fusion component used in NeuroFusionXAI.
    
    Args:
        embed_dim: Embedding dimension from ViT
        fusion_hidden_dim: Hidden dimension for fusion transformer
        cross_attn_heads: Number of heads for cross-attention
        fusion_heads: Number of heads for fusion transformer
        cross_attn_layers: Number of cross-attention layers
        fusion_layers: Number of fusion transformer layers
        dropout: Dropout rate
        modalities: List of modality names
    """
    
    def __init__(
        self,
        embed_dim: int = 768,
        fusion_hidden_dim: int = 1024,
        cross_attn_heads: int = 16,
        fusion_heads: int = 16,
        cross_attn_layers: int = 4,
        fusion_layers: int = 8,
        dropout: float = 0.1,
        modalities: List[str] = ['smri', 'fmri', 'pet'],
    ):
        super().__init__()
        
        # Cross-attention fusion for modality pairs
        self.cross_fusion = CrossAttentionFusion(
            embed_dim=embed_dim,
            num_heads=cross_attn_heads,
            num_layers=cross_attn_layers,
            dropout=dropout,
            modalities=modalities,
        )
        
        # Fusion transformer
        num_pairs = len(modalities) * (len(modalities) - 1) // 2
        self.fusion_transformer = FusionTransformer(
            input_dim=embed_dim,
            hidden_dim=fusion_hidden_dim,
            num_layers=fusion_layers,
            num_heads=fusion_heads,
            dropout=dropout,
            num_modality_pairs=num_pairs,
        )
        
    def forward(
        self,
        modality_features: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            modality_features: Dictionary of modality features
            
        Returns:
            fused: Final fused representation (B, embed_dim)
            attention_weights: Attention weights from cross-attention
        """
        # Get fused features from cross-attention
        fused, attn_weights = self.cross_fusion(modality_features)
        
        return fused, attn_weights
