import torch
import torch.nn as nn


class TransformerAttention(nn.Module):
    """Transformer-based attention mechanism for MIL"""
    
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super(TransformerAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        
    def forward(self, x):
        # Create a query vector (simple mean aggregation to start)
        query = torch.mean(x, dim=1, keepdim=True)  # [batch, 1, dim]
        
        # Apply attention (keys and values are both x)
        attn_output, attn_weights = self.multihead_attn(query, x, x)
        
        # Return attention weights for interpretability
        return attn_output.squeeze(1), attn_weights


class MILTransformer(nn.Module):
    """Transformer-based Multiple Instance Learning model"""
    
    def __init__(self, feature_dim=512, hidden_dim=128, num_heads=4, 
                 num_layers=2, dropout=0.3, num_classes=2, max_patches=1000):
        super(MILTransformer, self).__init__()
        
        # Feature projection layer
        self.feature_proj = nn.Linear(feature_dim, hidden_dim)
        
        # Position encoding (learned)
        self.pos_encoder = nn.Parameter(torch.zeros(1, max_patches, hidden_dim))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=num_heads, 
            dim_feedforward=hidden_dim*2,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer, 
            num_layers=num_layers
        )
        
        # Attention-based pooling
        self.attention = TransformerAttention(hidden_dim, num_heads=num_heads, dropout=dropout)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, return_attn=False):
        """
        Forward pass through the MIL model.
        
        Args:
            x (torch.Tensor): Batch of patch embeddings [batch_size, n_patches, feature_dim]
            return_attn (bool): Whether to return attention weights
            
        Returns:
            torch.Tensor: Class logits
            (torch.Tensor, optional): Attention weights if return_attn=True
        """
        batch_size, n_patches, _ = x.shape
        
        # Project features
        x = self.feature_proj(x)  # [batch_size, n_patches, hidden_dim]
        
        # Add positional encoding (trimmed to n_patches)
        pos_encoding = self.pos_encoder[:, :n_patches, :]
        x = x + pos_encoding
        
        # Apply transformer encoder
        x = self.norm1(x)
        x = self.transformer_encoder(x)
        x = self.norm2(x)
        
        # Apply attention pooling
        x, attn_weights = self.attention(x)  # [batch_size, hidden_dim]
        
        # Apply classifier
        x = self.dropout(x)
        logits = self.classifier(x)
        
        if return_attn:
            return logits, attn_weights
        
        return logits


def create_model(feature_dim=512, hidden_dim=128, num_heads=4, num_layers=2, 
                dropout=0.3, num_classes=2, max_patches=1000, device=None):
    """
    Create and initialize a MIL Transformer model.
    
    Args:
        feature_dim (int): Dimension of input features
        hidden_dim (int): Hidden dimension in the model
        num_heads (int): Number of attention heads
        num_layers (int): Number of transformer layers
        dropout (float): Dropout rate
        num_classes (int): Number of output classes
        max_patches (int): Maximum number of patches (for positional encoding)
        device (torch.device): Device to place the model on
        
    Returns:
        MILTransformer: Initialized model
    """
    model = MILTransformer(
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
        num_classes=num_classes,
        max_patches=max_patches
    )
    
    if device is not None:
        model = model.to(device)
        
    return model


# You can add other model architectures here, using a similar interface.
# For example:
"""
class AttentionMIL(nn.Module):
    # Implementation of a standard attention-based MIL model
    ...

def create_attention_mil_model(...):
    # Factory function for the attention-based MIL model
    ...
"""