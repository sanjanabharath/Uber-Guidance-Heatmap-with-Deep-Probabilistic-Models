import torch
import torch.nn as nn
from fast_transformers.builders import TransformerEncoderBuilder


class FastTransformerModel(nn.Module):
    def __init__(
        self,
        n_layers=4,
        n_heads=12,
        query_dimensions=64,
        value_dimensions=64,
        feed_forward_dimensions=3072,
        attention_type="full",
        activation="gelu",
        input_dim=768,
        hidden_dim=512,
        output_dim=256,
        num_classes=10
    ):
        """
        Fast Transformer Model with 4 neural layers
        
        Args:
            n_layers: Number of transformer layers
            n_heads: Number of attention heads
            query_dimensions: Dimension of query vectors
            value_dimensions: Dimension of value vectors
            feed_forward_dimensions: Dimension of feed-forward layer
            attention_type: Type of attention ("full", "linear", etc.)
            activation: Activation function
            input_dim: Input feature dimension (should be n_heads * query_dimensions)
            hidden_dim: Hidden layer dimension
            output_dim: Output layer dimension
            num_classes: Number of output classes
        """
        super(FastTransformerModel, self).__init__()
        
        self.input_dim = input_dim
        self.n_heads = n_heads
        self.query_dimensions = query_dimensions
        
        # Build transformer encoder
        self.transformer = TransformerEncoderBuilder.from_kwargs(
            n_layers=n_layers,
            n_heads=n_heads,
            query_dimensions=query_dimensions,
            value_dimensions=value_dimensions,
            feed_forward_dimensions=feed_forward_dimensions,
            attention_type=attention_type,
            activation=activation
        ).get()
        
        # 4 Neural layers after transformer
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, output_dim)
        self.layer4 = nn.Linear(output_dim, num_classes)
        
        # Activation and normalization
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(0.1)
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(output_dim)
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # Pass through transformer
        transformer_out = self.transformer(x)  # (batch_size, seq_len, input_dim)
        
        # Pool the sequence dimension (mean pooling)
        pooled = torch.mean(transformer_out, dim=1)  # (batch_size, input_dim)
        
        # Layer 1
        x = self.layer1(pooled)
        x = self.layer_norm1(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        # Layer 2
        x = self.layer2(x)
        x = self.layer_norm1(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        # Layer 3
        x = self.layer3(x)
        x = self.layer_norm2(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        # Layer 4 (output layer)
        x = self.layer4(x)
        
        return x


# Example usage
if __name__ == "__main__":
    # Initialize model
    model = FastTransformerModel(
        n_layers=4,
        n_heads=12,
        query_dimensions=64,
        value_dimensions=64,
        feed_forward_dimensions=3072,
        attention_type="full",
        input_dim=768,  # 12 heads * 64 dimensions
        hidden_dim=512,
        output_dim=256,
        num_classes=10
    )
    
    # Create sample input
    batch_size = 10
    seq_length = 512
    input_dim = 768  # 12 * 64
    
    x = torch.rand(batch_size, seq_length, input_dim)
    
    # Forward pass
    output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")