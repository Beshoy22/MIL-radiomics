import torch
import torch.nn as nn


class LSTMAttention(nn.Module):
    """LSTM-based attention mechanism for MIL"""
    
    def __init__(self, hidden_dim, dropout=0.1):
        super(LSTMAttention, self).__init__()
        self.attention_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        # x shape: [batch_size, n_patches, hidden_dim]
        attn_weights = self.attention_layer(x)  # [batch_size, n_patches, 1]
        
        # Apply attention weights
        context = torch.bmm(torch.transpose(attn_weights, 1, 2), x)  # [batch_size, 1, hidden_dim]
        
        return context.squeeze(1), attn_weights


class MIL_LSTM(nn.Module):
    """LSTM-based Multiple Instance Learning model with optional attention"""
    
    def __init__(self, feature_dim=512, hidden_dim=128, num_layers=2, 
                 dropout=0.3, bidirectional=True, num_classes=2, max_patches=300,
                 use_attention=True):  # Added parameter for toggling attention
        super(MIL_LSTM, self).__init__()
        
        # Feature projection layer
        self.feature_proj = nn.Linear(feature_dim, hidden_dim)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Determine LSTM output dimension
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        # Layer normalization for better training stability
        self.layer_norm = nn.LayerNorm(lstm_output_dim)
        
        # Flag for using attention
        self.use_attention = use_attention
        
        # Attention-based pooling
        if use_attention:
            self.attention = LSTMAttention(lstm_output_dim, dropout=dropout)
        else:
            # Global pooling as alternative when not using attention
            self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Classifier with regularization to prevent overfitting
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_dim, lstm_output_dim // 2),
            nn.LayerNorm(lstm_output_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_output_dim // 2, num_classes)
        )
        
        # Additional dropout
        self.dropout = nn.Dropout(dropout)
        
        # Save dimensions for reference
        self.max_patches = max_patches
        self.lstm_output_dim = lstm_output_dim
        
    def forward(self, x, return_attn=False):
        """
        Forward pass through the MIL LSTM model.
        
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
        
        # Create a mask for padded sequences (zeros in the input are considered padding)
        mask = (torch.sum(torch.abs(x), dim=2) > 0).float().unsqueeze(-1)  # [batch_size, n_patches, 1]
        
        # Apply LSTM
        x, _ = self.lstm(x)  # [batch_size, n_patches, lstm_output_dim]
        
        # Apply mask to zero out padded sequences
        x = x * mask
        
        # Apply layer normalization
        x = self.layer_norm(x)
        
        # Process based on whether we're using attention or not
        if self.use_attention:
            # Apply attention pooling
            x, attn_weights = self.attention(x)  # [batch_size, lstm_output_dim]
        else:
            # Use global average pooling over patches
            # First transpose to [batch_size, lstm_output_dim, n_patches]
            x_transposed = x.transpose(1, 2)
            # Apply pooling
            x = self.global_pool(x_transposed).squeeze(2)  # [batch_size, lstm_output_dim]
            # Create dummy attention weights for compatibility when return_attn=True
            attn_weights = torch.ones(batch_size, 1, n_patches) / n_patches
            attn_weights = attn_weights.to(x.device)
        
        # Apply dropout for regularization
        x = self.dropout(x)
        
        # Apply classifier
        logits = self.classifier(x)
        
        if return_attn:
            return logits, attn_weights
        
        return logits


def create_lstm_model(feature_dim=512, hidden_dim=128, num_layers=2, dropout=0.3, 
                     bidirectional=True, num_classes=2, max_patches=300, 
                     use_attention=True, device=None):
    """
    Create and initialize a MIL LSTM model.
    
    Args:
        feature_dim (int): Dimension of input features
        hidden_dim (int): Hidden dimension in the model
        num_layers (int): Number of LSTM layers
        dropout (float): Dropout rate
        bidirectional (bool): Whether to use bidirectional LSTM
        num_classes (int): Number of output classes
        max_patches (int): Maximum number of patches
        use_attention (bool): Whether to use attention mechanism
        device (torch.device): Device to place the model on
        
    Returns:
        MIL_LSTM: Initialized model
    """
    model = MIL_LSTM(
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        bidirectional=bidirectional,
        num_classes=num_classes,
        max_patches=max_patches,
        use_attention=use_attention  # Pass the attention flag
    )
    
    # Initialize weights for better training stability
    for name, param in model.named_parameters():
        if 'weight' in name and 'lstm' not in name:  # Skip LSTM weights which have their own initialization
            if len(param.shape) >= 2:  # Only apply Kaiming initialization to parameters with at least 2 dimensions
                nn.init.kaiming_normal_(param)
            else:
                nn.init.normal_(param, mean=0.0, std=0.01)  # Use normal initialization for 1D parameters
        elif 'bias' in name:
            nn.init.zeros_(param)
    
    if device is not None:
        model = model.to(device)
        
    return model