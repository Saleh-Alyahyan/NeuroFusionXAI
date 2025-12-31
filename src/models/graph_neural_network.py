"""
Graph Neural Network for Brain Connectivity Modeling

This module implements the GNN component for modeling brain connectivity
patterns and anatomical relationships as described in Section III.C.3 of
the NeuroFusionXAI paper.

The brain is represented as a graph where:
- Nodes V = {v_1, v_2, ..., v_R} represent R anatomical brain regions
- Edges E represent connectivity strength from functional/structural matrices
"""

from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, Batch


class GraphAttentionLayer(nn.Module):
    """
    Graph Attention Layer with multi-head attention.
    
    Implements the message passing update (Eq. 8):
    h_r^(l+1) = σ(W^(l) · AGGREGATE_{s∈N(r)}(h_s^(l)))
    
    Args:
        in_features: Input feature dimension
        out_features: Output feature dimension
        num_heads: Number of attention heads
        dropout: Dropout rate
        concat: Whether to concatenate or average multi-head outputs
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        concat: bool = True,
    ):
        super().__init__()
        self.gat = GATConv(
            in_channels=in_features,
            out_channels=out_features,
            heads=num_heads,
            dropout=dropout,
            concat=concat,
        )
        self.norm = nn.LayerNorm(out_features * num_heads if concat else out_features)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Node features (N, in_features)
            edge_index: Graph connectivity (2, E)
            edge_attr: Edge attributes (E, edge_features)
            
        Returns:
            Updated node features
        """
        x = self.gat(x, edge_index)
        x = F.elu(x)
        x = self.norm(x)
        x = self.dropout(x)
        return x


class BrainRegionEncoder(nn.Module):
    """
    Encoder for aggregating voxel-level features into brain region features.
    
    Implements Eq. 7:
    h_r^(0) = Aggregate({f_r^mod}_{mod ∈ {sMRI, fMRI, PET}})
    
    Args:
        voxel_dim: Dimension of voxel-level features
        region_dim: Dimension of region-level features
        num_modalities: Number of imaging modalities
    """
    
    def __init__(
        self,
        voxel_dim: int = 768,
        region_dim: int = 256,
        num_modalities: int = 3,
    ):
        super().__init__()
        
        # Modality-specific projections
        self.modality_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(voxel_dim, region_dim),
                nn.LayerNorm(region_dim),
                nn.GELU(),
            )
            for _ in range(num_modalities)
        ])
        
        # Fusion layer for combining modality features
        self.fusion = nn.Sequential(
            nn.Linear(region_dim * num_modalities, region_dim),
            nn.LayerNorm(region_dim),
            nn.GELU(),
        )
        
    def forward(
        self,
        modality_features: List[torch.Tensor],
        region_masks: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            modality_features: List of features from each modality
                              [(B, N_patches, voxel_dim), ...]
            region_masks: Mapping from patches to brain regions
                         (B, N_patches, N_regions)
                         
        Returns:
            Region-level features (B, N_regions, region_dim)
        """
        B = modality_features[0].shape[0]
        N_regions = region_masks.shape[-1]
        
        projected_features = []
        for i, (features, proj) in enumerate(zip(modality_features, self.modality_projections)):
            # Project modality features
            proj_feat = proj(features)  # (B, N_patches, region_dim)
            
            # Aggregate to regions using attention-weighted pooling
            # region_masks: (B, N_patches, N_regions)
            attention = F.softmax(region_masks, dim=1)  # Normalize over patches
            region_feat = torch.einsum('bpr,bpd->brd', attention, proj_feat)
            
            projected_features.append(region_feat)
            
        # Concatenate and fuse modality features
        fused = torch.cat(projected_features, dim=-1)  # (B, N_regions, region_dim * 3)
        region_features = self.fusion(fused)  # (B, N_regions, region_dim)
        
        return region_features


class BrainConnectivityGraph(nn.Module):
    """
    Module for constructing brain connectivity graphs from neuroimaging data.
    
    Creates adjacency matrices from functional and structural connectivity.
    """
    
    def __init__(
        self,
        num_regions: int = 116,
        connectivity_threshold: float = 0.3,
    ):
        super().__init__()
        self.num_regions = num_regions
        self.threshold = connectivity_threshold
        
        # Learnable connectivity prior
        self.connectivity_prior = nn.Parameter(
            torch.randn(num_regions, num_regions) * 0.01
        )
        
    def forward(
        self,
        functional_connectivity: Optional[torch.Tensor] = None,
        structural_connectivity: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            functional_connectivity: FC matrix (B, N_regions, N_regions)
            structural_connectivity: SC matrix (B, N_regions, N_regions)
            
        Returns:
            edge_index: Graph connectivity (2, E)
            edge_attr: Edge weights (E,)
        """
        B = functional_connectivity.shape[0] if functional_connectivity is not None else 1
        
        # Combine connectivity matrices
        connectivity = self.connectivity_prior.unsqueeze(0).expand(B, -1, -1)
        
        if functional_connectivity is not None:
            connectivity = connectivity + functional_connectivity
        if structural_connectivity is not None:
            connectivity = connectivity + structural_connectivity
            
        # Make symmetric
        connectivity = (connectivity + connectivity.transpose(-1, -2)) / 2
        
        # Apply threshold to create sparse graph
        mask = connectivity.abs() > self.threshold
        
        # Convert to edge_index format
        edge_indices = []
        edge_attrs = []
        
        for b in range(B):
            edges = mask[b].nonzero(as_tuple=False).T  # (2, E_b)
            attrs = connectivity[b][mask[b]]  # (E_b,)
            edge_indices.append(edges)
            edge_attrs.append(attrs)
            
        return edge_indices, edge_attrs


class BrainGNN(nn.Module):
    """
    Graph Neural Network for brain connectivity analysis.
    
    This component models brain connectivity patterns and anatomical
    relationships as described in Section III.C.3 of the paper.
    
    Args:
        input_dim: Input feature dimension per region
        hidden_dims: Hidden dimensions for each GAT layer
        num_heads: Number of attention heads
        dropout: Dropout rate
        num_regions: Number of brain regions (default: AAL atlas = 116)
        output_dim: Output feature dimension
    """
    
    def __init__(
        self,
        input_dim: int = 256,
        hidden_dims: List[int] = [256, 128, 64],
        num_heads: int = 8,
        dropout: float = 0.1,
        num_regions: int = 116,
        output_dim: int = 256,
    ):
        super().__init__()
        self.num_regions = num_regions
        self.input_dim = input_dim
        
        # Graph construction
        self.graph_constructor = BrainConnectivityGraph(
            num_regions=num_regions,
        )
        
        # GAT layers
        self.gat_layers = nn.ModuleList()
        
        in_dim = input_dim
        for i, hidden_dim in enumerate(hidden_dims):
            concat = (i < len(hidden_dims) - 1)  # Concat for all but last layer
            self.gat_layers.append(
                GraphAttentionLayer(
                    in_features=in_dim,
                    out_features=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    concat=concat,
                )
            )
            in_dim = hidden_dim * num_heads if concat else hidden_dim
            
        # Final projection
        self.output_proj = nn.Sequential(
            nn.Linear(in_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
        )
        
        # Graph-level pooling
        self.pool_weights = nn.Linear(output_dim, 1)
        
    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
        functional_connectivity: Optional[torch.Tensor] = None,
        structural_connectivity: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            node_features: Region features (B * N_regions, input_dim) or (B, N_regions, input_dim)
            edge_index: Precomputed graph connectivity
            edge_attr: Precomputed edge attributes
            functional_connectivity: FC matrix for graph construction
            structural_connectivity: SC matrix for graph construction
            batch: Batch index for each node
            
        Returns:
            graph_features: Graph-level features (B, output_dim)
            node_features: Updated node features (B * N_regions, output_dim)
        """
        # Handle batched input
        if node_features.dim() == 3:
            B, N, D = node_features.shape
            node_features = node_features.reshape(B * N, D)
            batch = torch.arange(B, device=node_features.device).repeat_interleave(N)
            
        # Construct graph if not provided
        if edge_index is None:
            edge_indices, edge_attrs = self.graph_constructor(
                functional_connectivity,
                structural_connectivity,
            )
            # Combine batched graphs
            edge_index, edge_attr, batch = self._batch_graphs(
                edge_indices, edge_attrs, self.num_regions
            )
            
        # Apply GAT layers
        x = node_features
        for gat_layer in self.gat_layers:
            x = gat_layer(x, edge_index, edge_attr)
            
        # Output projection
        node_out = self.output_proj(x)
        
        # Graph-level pooling with attention
        pool_weights = F.softmax(self.pool_weights(node_out), dim=0)
        
        # Weighted global pooling
        if batch is not None:
            # Scatter mean for batched graphs
            graph_features = self._weighted_scatter(node_out, pool_weights, batch)
        else:
            graph_features = (node_out * pool_weights).sum(dim=0, keepdim=True)
            
        return graph_features, node_out
    
    def _batch_graphs(
        self,
        edge_indices: List[torch.Tensor],
        edge_attrs: List[torch.Tensor],
        num_nodes: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Batch multiple graphs into a single disconnected graph."""
        device = edge_indices[0].device
        
        batched_edges = []
        batched_attrs = []
        batch_idx = []
        
        offset = 0
        for i, (edges, attrs) in enumerate(zip(edge_indices, edge_attrs)):
            batched_edges.append(edges + offset)
            batched_attrs.append(attrs)
            batch_idx.extend([i] * num_nodes)
            offset += num_nodes
            
        edge_index = torch.cat(batched_edges, dim=1)
        edge_attr = torch.cat(batched_attrs)
        batch = torch.tensor(batch_idx, device=device)
        
        return edge_index, edge_attr, batch
    
    def _weighted_scatter(
        self,
        src: torch.Tensor,
        weights: torch.Tensor,
        index: torch.Tensor,
    ) -> torch.Tensor:
        """Weighted scatter mean operation."""
        # Get number of graphs
        num_graphs = index.max().item() + 1
        dim = src.shape[-1]
        
        # Initialize output
        out = torch.zeros(num_graphs, dim, device=src.device)
        
        # Weighted sum
        weighted_src = src * weights
        out.scatter_add_(0, index.unsqueeze(-1).expand_as(weighted_src), weighted_src)
        
        # Normalize by weight sum per graph
        weight_sum = torch.zeros(num_graphs, 1, device=src.device)
        weight_sum.scatter_add_(0, index.unsqueeze(-1), weights)
        weight_sum = weight_sum.clamp(min=1e-8)
        
        return out / weight_sum


class BrainGNNWithRegionEncoder(nn.Module):
    """
    Complete GNN module with region encoding from multimodal features.
    
    This combines the region encoder and GNN for end-to-end processing
    from patch features to graph-level representations.
    
    Args:
        voxel_dim: Dimension of voxel-level features from ViT
        region_dim: Dimension of region-level features
        gnn_hidden_dims: Hidden dimensions for GNN layers
        num_heads: Number of attention heads
        dropout: Dropout rate
        num_regions: Number of brain regions
        num_modalities: Number of imaging modalities
    """
    
    def __init__(
        self,
        voxel_dim: int = 768,
        region_dim: int = 256,
        gnn_hidden_dims: List[int] = [256, 128, 64],
        num_heads: int = 8,
        dropout: float = 0.1,
        num_regions: int = 116,
        num_modalities: int = 3,
    ):
        super().__init__()
        
        # Region encoder
        self.region_encoder = BrainRegionEncoder(
            voxel_dim=voxel_dim,
            region_dim=region_dim,
            num_modalities=num_modalities,
        )
        
        # GNN
        self.gnn = BrainGNN(
            input_dim=region_dim,
            hidden_dims=gnn_hidden_dims,
            num_heads=num_heads,
            dropout=dropout,
            num_regions=num_regions,
            output_dim=region_dim,
        )
        
    def forward(
        self,
        modality_features: List[torch.Tensor],
        region_masks: torch.Tensor,
        functional_connectivity: Optional[torch.Tensor] = None,
        structural_connectivity: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            modality_features: List of ViT features from each modality
            region_masks: Mapping from patches to brain regions
            functional_connectivity: FC matrix
            structural_connectivity: SC matrix
            
        Returns:
            graph_features: Graph-level features (B, region_dim)
            node_features: Updated node features (B * N_regions, region_dim)
        """
        # Encode regions from multimodal features
        region_features = self.region_encoder(modality_features, region_masks)
        
        # Apply GNN
        graph_features, node_features = self.gnn(
            region_features,
            functional_connectivity=functional_connectivity,
            structural_connectivity=structural_connectivity,
        )
        
        return graph_features, node_features


def create_region_masks(
    patch_grid_size: Tuple[int, int, int],
    atlas_path: str,
    device: torch.device,
) -> torch.Tensor:
    """
    Create mapping from ViT patches to brain regions based on atlas.
    
    Args:
        patch_grid_size: Size of patch grid (D', H', W')
        atlas_path: Path to brain atlas (e.g., AAL)
        device: Target device
        
    Returns:
        Region masks tensor (1, N_patches, N_regions)
    """
    # This would typically load an atlas and create the mapping
    # For now, return a placeholder
    D, H, W = patch_grid_size
    N_patches = D * H * W
    N_regions = 116  # AAL atlas
    
    # Random soft assignment as placeholder
    masks = torch.randn(1, N_patches, N_regions, device=device)
    masks = F.softmax(masks, dim=-1)
    
    return masks
