import torch
import torch.nn as nn
import torch.nn.functional as F


class EfficientPatchAttention(nn.Module):
    """
    Lightweight attention mechanism for patch processing.
    Uses a simple linear projection followed by softmax to generate attention weights.
    """
    
    def __init__(self, feature_dim, hidden_dim=32):
        super(EfficientPatchAttention, self).__init__()
        
        # Simple projection to generate attention scores
        self.attention_proj = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Batch of patch embeddings [batch_size, n_patches, feature_dim]
            
        Returns:
            torch.Tensor: Attention weights [batch_size, n_patches, 1]
        """
        # Generate attention scores
        attn_scores = self.attention_proj(x)  # [batch_size, n_patches, 1]
        
        # Create mask for padding (all zeros in input are considered padding)
        mask = (torch.sum(torch.abs(x), dim=2, keepdim=True) > 0).float()
        
        # Apply mask (set padded patches to large negative value)
        attn_scores = attn_scores * mask - 1e10 * (1 - mask)
        
        # Apply softmax to get attention weights (over n_patches dimension)
        attn_weights = F.softmax(attn_scores, dim=1)  # [batch_size, n_patches, 1]
        
        return attn_weights


class ConvBlock(nn.Module):
    """
    A simple convolutional block with batch normalization and residual connection.
    """
    
    def __init__(self, in_channels, out_channels, kernel_size=3, dropout=0.2):
        super(ConvBlock, self).__init__()
        
        self.conv = nn.Conv1d(
            in_channels, 
            out_channels,
            kernel_size=kernel_size,
            padding=kernel_size//2
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        
        # Projection for residual connection if dimensions don't match
        self.residual_proj = nn.Identity()
        if in_channels != out_channels:
            self.residual_proj = nn.Conv1d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        residual = self.residual_proj(x)
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = x + residual
        return x


class LightweightMIL_Conv(nn.Module):
    """
    Lightweight Convolutional Multiple Instance Learning model with configurable depth.
    
    Features:
    - Efficient attention mechanism for patch processing
    - Configurable number of convolutional blocks
    - Weight sharing via grouped convolutions
    - Residual connections for better gradient flow
    """
    
    def __init__(self, feature_dim=512, hidden_dim=64, 
                 num_blocks=2, dropout=0.2, num_classes=2, 
                 max_patches=300, num_groups=10):
        super(LightweightMIL_Conv, self).__init__()
        
        self.feature_dim = feature_dim
        self.max_patches = max_patches
        self.num_groups = num_groups
        
        # Dimension reduction for input features to save parameters
        self.dim_reduction = nn.Linear(feature_dim, hidden_dim)
        
        # Patch attention mechanism
        self.patch_attention = EfficientPatchAttention(hidden_dim, hidden_dim//2)
        
        # Create configurable number of convolutional blocks
        self.conv_blocks = nn.ModuleList()
        
        # First conv block
        self.conv_blocks.append(ConvBlock(
            in_channels=num_groups,
            out_channels=hidden_dim,
            kernel_size=3,
            dropout=dropout
        ))
        
        # Additional conv blocks
        for i in range(1, num_blocks):
            self.conv_blocks.append(ConvBlock(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                kernel_size=3,
                dropout=dropout
            ))
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Final classifier - streamlined
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, x, return_attn=False):
        """
        Forward pass through the Lightweight MIL Conv model.
        
        Args:
            x (torch.Tensor): Batch of patch embeddings [batch_size, n_patches, feature_dim]
            return_attn (bool): Whether to return attention weights
            
        Returns:
            torch.Tensor: Class logits
            (torch.Tensor, optional): Attention weights if return_attn=True
        """
        batch_size, n_patches, _ = x.shape
        
        # Dimension reduction
        x = self.dim_reduction(x)  # [batch_size, n_patches, hidden_dim]
        
        # Compute attention weights
        attn_weights = self.patch_attention(x)  # [batch_size, n_patches, 1]
        
        # Apply attention weights to input features
        weighted_features = x * attn_weights  # [batch_size, n_patches, hidden_dim]
        
        # Group patches into groups of approximately equal size
        patches_per_group = (n_patches + self.num_groups - 1) // self.num_groups
        grouped_features = torch.zeros(batch_size, self.num_groups, x.size(2), device=x.device)
        
        for i in range(self.num_groups):
            start_idx = i * patches_per_group
            end_idx = min((i + 1) * patches_per_group, n_patches)
            
            if start_idx < end_idx:
                # Sum the weighted features in this group
                group_sum = torch.sum(weighted_features[:, start_idx:end_idx, :], dim=1)
                grouped_features[:, i, :] = group_sum
        
        # Process through convolutional blocks
        conv_output = grouped_features
        for conv_block in self.conv_blocks:
            conv_output = conv_block(conv_output)
        
        # Apply global pooling [batch_size, hidden_dim, feature_dim] -> [batch_size, hidden_dim, 1]
        pooled = self.global_pool(conv_output).squeeze(2)  # [batch_size, hidden_dim]
        
        # Apply classifier
        logits = self.classifier(pooled)
        
        if return_attn:
            return logits, attn_weights
        
        return logits


def create_lightweight_conv_model(feature_dim=512, hidden_dim=64, num_blocks=2,
                                 dropout=0.2, num_classes=2, max_patches=300, 
                                 num_groups=10, device=None):
    """
    Create and initialize a Lightweight MIL Conv model.
    
    Args:
        feature_dim (int): Dimension of input features
        hidden_dim (int): Hidden dimension in the model
        num_blocks (int): Number of convolutional blocks
        dropout (float): Dropout rate
        num_classes (int): Number of output classes
        max_patches (int): Maximum number of patches
        num_groups (int): Number of groups for aggregation
        device (torch.device): Device to place the model on
        
    Returns:
        LightweightMIL_Conv: Initialized model
    """
    model = LightweightMIL_Conv(
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
        num_blocks=num_blocks,
        dropout=dropout,
        num_classes=num_classes,
        max_patches=max_patches,
        num_groups=num_groups
    )
    
    # Initialize weights for better training stability
    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() > 1:
            nn.init.kaiming_normal_(param)
    
    if device is not None:
        model = model.to(device)
        
    return model