import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchAttention(nn.Module):
    """
    Patch-level attention mechanism for MIL.
    
    Applies attention to each patch to determine its importance,
    then normalizes these scores with softmax to create attention weights.
    """
    
    def __init__(self, feature_dim, hidden_dim=64, dropout=0.1):
        super(PatchAttention, self).__init__()
        
        # Attention mechanism using dense layers
        self.attention_proj = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
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


class DenseBlock(nn.Module):
    """
    A dense block with optional batch normalization and residual connection.
    """
    
    def __init__(self, input_dim, output_dim, dropout=0.2, 
                 batch_norm=True, residual=True, activation='relu'):
        super(DenseBlock, self).__init__()
        
        layers = []
        
        # Main dense layer
        layers.append(nn.Linear(input_dim, output_dim))
        
        # Batch normalization
        if batch_norm:
            layers.append(nn.LayerNorm(output_dim))
        
        # Activation function
        if activation == 'relu':
            layers.append(nn.ReLU())
        elif activation == 'gelu':
            layers.append(nn.GELU())
        elif activation == 'leaky_relu':
            layers.append(nn.LeakyReLU(0.1))
        elif activation == 'tanh':
            layers.append(nn.Tanh())
        
        # Dropout
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        
        self.main_path = nn.Sequential(*layers)
        
        # Residual connection
        self.residual = residual
        if residual and input_dim != output_dim:
            self.residual_proj = nn.Linear(input_dim, output_dim)
        else:
            self.residual_proj = nn.Identity()
    
    def forward(self, x):
        main_output = self.main_path(x)
        
        if self.residual:
            residual = self.residual_proj(x)
            return main_output + residual
        
        return main_output


class MIL_Dense(nn.Module):
    """
    Dense-layer Multiple Instance Learning model with patch-level attention.
    
    This model:
    1. Uses attention to compute weights for each patch
    2. Aggregates weighted patches into groups
    3. Processes the groups with dense layers for classification
    """
    
    def __init__(self, feature_dim=512, hidden_dims=[256, 128, 64], 
                 dropout=0.3, num_classes=2, max_patches=300, 
                 num_groups=10, use_top_k=False, batch_norm=True,
                 residual=True, activation='relu'):
        super(MIL_Dense, self).__init__()
        
        self.feature_dim = feature_dim
        self.max_patches = max_patches
        self.num_groups = num_groups
        self.use_top_k = use_top_k
        
        # Patch attention mechanism
        self.patch_attention = PatchAttention(
            feature_dim=feature_dim, 
            hidden_dim=hidden_dims[0] // 2 if hidden_dims else 64,
            dropout=dropout
        )
        
        # Create dense layers
        self.dense_blocks = nn.ModuleList()
        
        # First dense block processes the aggregated groups
        input_dim = num_groups * feature_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            self.dense_blocks.append(
                DenseBlock(
                    input_dim=input_dim,
                    output_dim=hidden_dim,
                    dropout=dropout,
                    batch_norm=batch_norm,
                    residual=residual,
                    activation=activation
                )
            )
            input_dim = hidden_dim
        
        # Final classifier
        final_hidden_dim = hidden_dims[-1] if hidden_dims else feature_dim
        self.classifier = nn.Linear(final_hidden_dim, num_classes)
        
    def forward(self, x, return_attn=False):
        """
        Forward pass through the MIL Dense model.
        
        Args:
            x (torch.Tensor): Batch of patch embeddings [batch_size, n_patches, feature_dim]
            return_attn (bool): Whether to return attention weights
            
        Returns:
            torch.Tensor: Class logits
            (torch.Tensor, optional): Attention weights if return_attn=True
        """
        batch_size, n_patches, _ = x.shape
        
        # Compute attention weights
        attn_weights = self.patch_attention(x)  # [batch_size, n_patches, 1]
        
        # Apply attention weights to input features
        weighted_features = x * attn_weights  # [batch_size, n_patches, feature_dim]
        
        # Top-k selection or group aggregation based on setting
        if self.use_top_k:
            # Sort patches by attention weights
            _, top_indices = torch.sort(attn_weights.squeeze(-1), dim=1, descending=True)
            # Select the top k patches (where k = num_groups)
            top_indices = top_indices[:, :self.num_groups]
            
            # Create a new tensor with only the top k patches
            # Vectorized version using torch.gather - much faster than loops!
            # Expand indices to match feature dimensions: [batch_size, num_groups, feature_dim]
            expanded_indices = top_indices.unsqueeze(-1).expand(-1, -1, self.feature_dim)
            # Gather features at the top indices
            selected_features = torch.gather(weighted_features, 1, expanded_indices)
                    
            # Reshape for dense processing: [batch_size, num_groups * feature_dim]
            flattened_features = selected_features.reshape(batch_size, -1)
            
        else:
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
            
            # Reshape for dense processing: [batch_size, num_groups * feature_dim]
            flattened_features = grouped_features.reshape(batch_size, -1)
        
        # Process through dense blocks
        dense_output = flattened_features
        for block in self.dense_blocks:
            dense_output = block(dense_output)
        
        # Apply classifier
        logits = self.classifier(dense_output)
        
        if return_attn:
            return logits, attn_weights
        
        return logits


def create_dense_model(feature_dim=512, hidden_dims=[256, 128, 64], 
                      dropout=0.3, num_classes=2, max_patches=300, 
                      num_groups=10, use_top_k=False, batch_norm=True,
                      residual=True, activation='relu', device=None):
    """
    Create and initialize a MIL Dense model.
    
    Args:
        feature_dim (int): Dimension of input features
        hidden_dims (list): List of hidden dimensions for each dense layer
        dropout (float): Dropout rate
        num_classes (int): Number of output classes
        max_patches (int): Maximum number of patches
        num_groups (int): Number of groups for aggregation
        use_top_k (bool): Whether to use top-k patch selection
        batch_norm (bool): Whether to use batch normalization
        residual (bool): Whether to use residual connections
        activation (str): Activation function ('relu', 'gelu', 'leaky_relu', or 'tanh')
        device (torch.device): Device to place the model on
        
    Returns:
        MIL_Dense: Initialized model
    """
    model = MIL_Dense(
        feature_dim=feature_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        num_classes=num_classes,
        max_patches=max_patches,
        num_groups=num_groups,
        use_top_k=use_top_k,
        batch_norm=batch_norm,
        residual=residual,
        activation=activation
    )
    
    # Initialize weights for better training stability
    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() > 1:
            nn.init.kaiming_normal_(param)
    
    if device is not None:
        model = model.to(device)
        
    return model