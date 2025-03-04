import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchAttention(nn.Module):
    """
    Patch-level attention mechanism using convolution.
    
    Applies a 1D convolution to each patch to generate an attention score,
    then normalizes these scores with softmax to create attention weights.
    This allows the model to focus on outlier patches that contain distinctive features.
    """
    
    def __init__(self, feature_dim, dropout=0.1):
        super(PatchAttention, self).__init__()
        
        # Convolutional layer to generate patch importance scores
        self.conv_attn = nn.Conv1d(1, 1, kernel_size=feature_dim, padding=0)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Batch of patch embeddings [batch_size, n_patches, feature_dim]
            
        Returns:
            torch.Tensor: Attention weights [batch_size, n_patches, 1]
        """
        batch_size, n_patches, feature_dim = x.shape
        
        # Reshape for batch processing
        x_flat = x.reshape(batch_size * n_patches, 1, feature_dim)
        
        # Apply convolution to get attention scores
        attn_scores = self.conv_attn(x_flat).reshape(batch_size, n_patches)
        
        # Create mask for padding (all zeros in input are considered padding)
        mask = (torch.sum(torch.abs(x), dim=2) > 0).float()
        
        # Apply mask (set padded patches to large negative value)
        attn_scores = attn_scores * mask - 1e10 * (1 - mask)
        
        # Apply softmax to get attention weights
        attn_weights = F.softmax(attn_scores, dim=1).unsqueeze(2)  # [batch_size, n_patches, 1]
        
        # Apply dropout
        attn_weights = self.dropout(attn_weights)
        
        return attn_weights


class MIL_Conv(nn.Module):
    """
    Convolutional Multiple Instance Learning model with patch-level attention.
    
    This model:
    1. Uses convolution to compute attention weights for each patch
    2. Aggregates weighted patches into groups
    3. Processes the groups with convolutional layers for classification
    """
    
    def __init__(self, feature_dim=512, hidden_dim=128, 
                 dropout=0.3, num_classes=2, max_patches=300, num_groups=10):
        super(MIL_Conv, self).__init__()
        
        self.feature_dim = feature_dim
        self.max_patches = max_patches
        self.num_groups = num_groups
        
        # Patch attention mechanism
        self.patch_attention = PatchAttention(feature_dim, dropout=dropout)
        
        # Convolutional networks for classification
        self.conv_net = nn.Sequential(
            # First conv layer
            nn.Conv1d(num_groups, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # Second conv layer
            nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # Third conv layer
            nn.Conv1d(hidden_dim * 2, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, x, return_attn=False):
        """
        Forward pass through the MIL Conv model.
        
        Args:
            x (torch.Tensor): Batch of patch embeddings [batch_size, n_patches, feature_dim]
            return_attn (bool): Whether to return attention weights
            
        Returns:
            torch.Tensor: Class logits
            (torch.Tensor, optional): Attention weights if return_attn=True
        """
        batch_size, n_patches, _ = x.shape
        
        # Compute attention weights using convolution
        attn_weights = self.patch_attention(x)  # [batch_size, n_patches, 1]
        
        # Apply attention weights to input features
        weighted_features = x * attn_weights  # [batch_size, n_patches, feature_dim]
        
        # Group patches into groups of approximately equal size
        patches_per_group = (n_patches + self.num_groups - 1) // self.num_groups
        grouped_features = torch.zeros(batch_size, self.num_groups, self.feature_dim, device=x.device)
        
        for i in range(self.num_groups):
            start_idx = i * patches_per_group
            end_idx = min((i + 1) * patches_per_group, n_patches)
            
            if start_idx < end_idx:
                # Sum the weighted features in this group
                group_sum = torch.sum(weighted_features[:, start_idx:end_idx, :], dim=1)
                grouped_features[:, i, :] = group_sum
        
        # Process grouped features with convolutional networks
        # [batch_size, num_groups, feature_dim]
        conv_output = self.conv_net(grouped_features)
        
        # Apply global pooling [batch_size, hidden_dim, feature_dim] -> [batch_size, hidden_dim, 1]
        pooled = self.global_pool(conv_output).squeeze(2)  # [batch_size, hidden_dim]
        
        # Apply classifier
        logits = self.classifier(pooled)
        
        if return_attn:
            return logits, attn_weights
        
        return logits


def create_conv_model(feature_dim=512, hidden_dim=128, dropout=0.3, 
                      num_classes=2, max_patches=300, num_groups=10, device=None):
    """
    Create and initialize a MIL Conv model.
    
    Args:
        feature_dim (int): Dimension of input features
        hidden_dim (int): Hidden dimension in the model
        dropout (float): Dropout rate
        num_classes (int): Number of output classes
        max_patches (int): Maximum number of patches
        num_groups (int): Number of groups for aggregation
        device (torch.device): Device to place the model on
        
    Returns:
        MIL_Conv: Initialized model
    """
    model = MIL_Conv(
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
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