"""
3D Vision Transformer for Volumetric Neuroimaging Data

This module implements the Vision Transformer (ViT) architecture adapted for
3D volumetric neuroimaging data (sMRI, fMRI, PET).

Architecture based on:
- "An Image is Worth 16x16 Words" (Dosovitskiy et al., 2020)
- Adapted for 3D medical imaging as described in the NeuroFusionXAI paper
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


class PatchEmbedding3D(nn.Module):
    """
    3D Patch Embedding layer for volumetric neuroimaging data.
    
    Divides 3D volume into non-overlapping patches and projects them
    to embedding dimension.
    
    Args:
        in_channels: Number of input channels (typically 1 for neuroimaging)
        embed_dim: Embedding dimension
        patch_size: Size of 3D patches (D, H, W)
        img_size: Size of input volume (D, H, W)
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        embed_dim: int = 768,
        patch_size: Tuple[int, int, int] = (16, 16, 16),
        img_size: Tuple[int, int, int] = (96, 112, 96),
    ):
        super().__init__()
        self.patch_size = patch_size
        self.img_size = img_size
        self.embed_dim = embed_dim
        
        # Calculate number of patches
        self.num_patches = (
            (img_size[0] // patch_size[0]) *
            (img_size[1] // patch_size[1]) *
            (img_size[2] // patch_size[2])
        )
        
        self.grid_size = (
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
            img_size[2] // patch_size[2],
        )
        
        # 3D Convolution for patch projection
        self.proj = nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        
        # Layer normalization
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, C, D, H, W)
            
        Returns:
            Patch embeddings of shape (B, N, embed_dim)
        """
        B, C, D, H, W = x.shape
        
        # Project patches: (B, C, D, H, W) -> (B, embed_dim, D', H', W')
        x = self.proj(x)
        
        # Flatten spatial dimensions: (B, embed_dim, D', H', W') -> (B, N, embed_dim)
        x = rearrange(x, 'b e d h w -> b (d h w) e')
        
        # Layer normalization
        x = self.norm(x)
        
        return x


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention mechanism for 3D Vision Transformer.
    
    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        dropout: Dropout rate
        attention_dropout: Attention dropout rate
    """
    
    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        # Query, Key, Value projections
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=True)
        
        # Output projection
        self.proj = nn.Linear(embed_dim, embed_dim)
        
        # Dropout
        self.attn_dropout = nn.Dropout(attention_dropout)
        self.proj_dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor of shape (B, N, embed_dim)
            mask: Optional attention mask
            
        Returns:
            Output tensor and attention weights
        """
        B, N, C = x.shape
        
        # Compute Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
            
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        
        # Output projection
        x = self.proj(x)
        x = self.proj_dropout(x)
        
        return x, attn


class MLP(nn.Module):
    """
    MLP block for Vision Transformer.
    
    Args:
        in_features: Input features
        hidden_features: Hidden layer features
        out_features: Output features
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or int(in_features * 4)
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """
    Transformer block with pre-norm architecture.
    
    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        mlp_ratio: MLP hidden dim ratio
        dropout: Dropout rate
        attention_dropout: Attention dropout rate
    """
    
    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
    ):
        super().__init__()
        
        # Pre-norm layers
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Attention
        self.attn = MultiHeadSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            attention_dropout=attention_dropout,
        )
        
        # MLP
        self.mlp = MLP(
            in_features=embed_dim,
            hidden_features=int(embed_dim * mlp_ratio),
            dropout=dropout,
        )
        
    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: Input tensor of shape (B, N, embed_dim)
            return_attention: Whether to return attention weights
            
        Returns:
            Output tensor and optional attention weights
        """
        # Self-attention with residual
        attn_out, attn_weights = self.attn(self.norm1(x))
        x = x + attn_out
        
        # MLP with residual
        x = x + self.mlp(self.norm2(x))
        
        if return_attention:
            return x, attn_weights
        return x, None


class VisionTransformer3D(nn.Module):
    """
    3D Vision Transformer for volumetric neuroimaging data.
    
    This is the modality-specific encoder E_mod described in the paper (Eq. 4):
    f^mod = E_mod(x^mod) = ViT_mod(Patch(x^mod))
    
    Args:
        in_channels: Number of input channels
        embed_dim: Embedding dimension
        patch_size: Size of 3D patches
        img_size: Size of input volume
        num_heads: Number of attention heads
        num_layers: Number of transformer blocks
        mlp_ratio: MLP hidden dim ratio
        dropout: Dropout rate
        attention_dropout: Attention dropout rate
        num_classes: Number of output classes (0 for feature extraction only)
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        embed_dim: int = 768,
        patch_size: Tuple[int, int, int] = (16, 16, 16),
        img_size: Tuple[int, int, int] = (96, 112, 96),
        num_heads: int = 12,
        num_layers: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        num_classes: int = 0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        
        # Patch embedding
        self.patch_embed = PatchEmbedding3D(
            in_channels=in_channels,
            embed_dim=embed_dim,
            patch_size=patch_size,
            img_size=img_size,
        )
        num_patches = self.patch_embed.num_patches
        
        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Position embeddings
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim)
        )
        self.pos_dropout = nn.Dropout(dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                attention_dropout=attention_dropout,
            )
            for _ in range(num_layers)
        ])
        
        # Final layer norm
        self.norm = nn.LayerNorm(embed_dim)
        
        # Classification head (optional)
        if num_classes > 0:
            self.head = nn.Linear(embed_dim, num_classes)
        else:
            self.head = nn.Identity()
            
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights using truncated normal distribution."""
        # Initialize position embeddings
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # Initialize other parameters
        self.apply(self._init_module_weights)
        
    def _init_module_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
                
    def forward_features(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[list]]:
        """
        Extract features without classification head.
        
        Args:
            x: Input tensor of shape (B, C, D, H, W)
            return_attention: Whether to return attention weights
            
        Returns:
            Feature tensor and optional attention weights from all layers
        """
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)
        
        # Add class token
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=B)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add position embeddings
        x = x + self.pos_embed
        x = self.pos_dropout(x)
        
        # Transformer blocks
        attention_weights = []
        for block in self.blocks:
            x, attn = block(x, return_attention=return_attention)
            if return_attention:
                attention_weights.append(attn)
                
        # Final normalization
        x = self.norm(x)
        
        if return_attention:
            return x, attention_weights
        return x, None
        
    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
        return_attention: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, C, D, H, W)
            return_features: Whether to return full feature tensor
            return_attention: Whether to return attention weights
            
        Returns:
            Class token features or logits
        """
        features, attention = self.forward_features(x, return_attention)
        
        if return_features:
            return features
            
        # Extract class token
        cls_features = features[:, 0]
        
        # Classification head
        out = self.head(cls_features)
        
        if return_attention:
            return out, attention
        return out
    
    def get_intermediate_features(
        self,
        x: torch.Tensor,
        layer_indices: list,
    ) -> list:
        """
        Get intermediate features from specific layers.
        
        Args:
            x: Input tensor
            layer_indices: List of layer indices to extract features from
            
        Returns:
            List of feature tensors from specified layers
        """
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)
        
        # Add class token
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=B)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add position embeddings
        x = x + self.pos_embed
        x = self.pos_dropout(x)
        
        # Collect intermediate features
        intermediate_features = []
        for i, block in enumerate(self.blocks):
            x, _ = block(x)
            if i in layer_indices:
                intermediate_features.append(x.clone())
                
        return intermediate_features


def create_vit_for_modality(
    modality: str,
    config: dict,
) -> VisionTransformer3D:
    """
    Factory function to create modality-specific Vision Transformer.
    
    Args:
        modality: Modality name ('smri', 'fmri', 'pet')
        config: Configuration dictionary
        
    Returns:
        Configured VisionTransformer3D instance
    """
    vit_config = config['model']['vit']
    
    return VisionTransformer3D(
        in_channels=1,
        embed_dim=vit_config['embed_dim'],
        patch_size=tuple(vit_config['patch_size']),
        img_size=tuple(config['data']['input_shape'][1:]),
        num_heads=vit_config['num_heads'],
        num_layers=vit_config['num_layers'],
        mlp_ratio=vit_config['mlp_ratio'],
        dropout=vit_config['dropout'],
        attention_dropout=vit_config['attention_dropout'],
        num_classes=0,  # Feature extraction only
    )
