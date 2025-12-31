import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MixtureSameFamily, Categorical, Normal
import math


class FastTransformer(nn.Module):
    def __init__(self, d_model, num_heads=4, ff_dim=256, dropout=0.1):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            d_model, 
            num_heads=num_heads, 
            dropout=dropout, 
            batch_first=True
        )
        
        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        x: (batch_size, seq_len, d_model)
        """
        # Self-attention block
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feed-forward block
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        
        return x


class EmbeddingLayer(nn.Module):
    """
    Embedding layer for categorical features.
    Each categorical feature gets its own embedding.
    """
    def __init__(self, categorical_cardinalities, embedding_dim=16):
        """
        Args:
            categorical_cardinalities: dict {feature_name: max_category_value}
                e.g., {'hour': 24, 'day_of_week': 7, 'region': 42}
            embedding_dim: dimension of each embedding
        """
        super().__init__()
        self.embeddings = nn.ModuleDict()
        self.embedding_dim = embedding_dim
        
        for feature_name, cardinality in categorical_cardinalities.items():
            # +1 for padding/unknown values
            self.embeddings[feature_name] = nn.Embedding(cardinality + 1, embedding_dim)
    
    def forward(self, categorical_inputs):
        """
        Args:
            categorical_inputs: dict {feature_name: tensor of shape (batch_size,)}
        
        Returns:
            embedded: (batch_size, num_categorical_features, embedding_dim)
        """
        embedded_list = []
        for feature_name, embedding_layer in self.embeddings.items():
            embedded = embedding_layer(categorical_inputs[feature_name])
            embedded_list.append(embedded)
        
        # Stack embeddings: (batch_size, num_features, embedding_dim)
        return torch.stack(embedded_list, dim=1)


class GMMEarningsModel(nn.Module):
    """
    Deep Probabilistic Earnings Forecasting Model using Gaussian Mixture Models.
    
    Architecture:
    1. Embedding layer: Categorical features → embeddings
    2. Fast Transformer: Process embeddings with attention
    3. Concatenate with scalar features
    4. Dense layers (Layer 1-4): 128 → 128 → 128 → 128
    5. Output heads:
       - Weights head: K mixture component weights (softmax)
       - Means head: K means for each component
       - Stds head: K standard deviations (softplus to ensure positivity)
    6. Loss: Negative log-likelihood with zero-inflation (truncated Gaussian)
    
    Handles zero-inflated target: P(y=0) is modeled separately,
    then P(y|y>0) is modeled with truncated Gaussian mixture.
    """
    
    def __init__(
        self,
        categorical_cardinalities,  # dict {feature_name: max_value}
        num_scalar_features,        # int: number of continuous features
        embedding_dim=16,           # embedding dimension for categorical features
        hidden_dim=128,             # hidden dimension for dense layers
        num_modes=3,                # number of Gaussian modes in mixture
        transformer_heads=4,        # number of attention heads
        dropout=0.1,
    ):
        super().__init__()
        self.num_modes = num_modes
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        # ============ INPUT PROCESSING ============
        
        # Categorical embeddings
        self.embedding_layer = EmbeddingLayer(categorical_cardinalities, embedding_dim)
        num_categorical_features = len(categorical_cardinalities)
        
        # Fast Transformer for embeddings
        self.transformer = FastTransformer(
            d_model=embedding_dim,
            num_heads=transformer_heads,
            ff_dim=256,
            dropout=dropout
        )
        
        # Transformer output: (batch_size, num_categorical_features, embedding_dim)
        # Flatten to: (batch_size, num_categorical_features * embedding_dim)
        transformer_output_dim = num_categorical_features * embedding_dim
        
        # ============ DENSE LAYERS (Layer 1-4) ============
        # Input: concatenated embeddings + scalar features
        # Path: transformer_output_dim + num_scalar_features → Layer1 → Layer2 → Layer3 → Layer4
        
        input_dim = transformer_output_dim + num_scalar_features
        
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        self.layer4 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # ============ OUTPUT HEADS ============
        
        # Weights head: outputs K logits for mixture components (softmax)
        self.weights_head = nn.Linear(hidden_dim, num_modes)
        
        # Means head: outputs K means
        self.mu_head = nn.Linear(hidden_dim, num_modes)
        
        # Standard deviation head: outputs K stds (softplus to ensure σ > 0)
        self.sigma_head = nn.Linear(hidden_dim, num_modes)
        
        # Zero-inflation head: P(y=0) vs P(y>0)
        # Binary logit: outputs probability of zero earnings
        self.zero_prob_head = nn.Linear(hidden_dim, 1)
    
    def forward(self, categorical_inputs, scalar_inputs):
        """
        Args:
            categorical_inputs: dict {feature_name: LongTensor of shape (batch_size,)}
            scalar_inputs: FloatTensor of shape (batch_size, num_scalar_features)
        
        Returns:
            weights: (batch_size, num_modes) - mixture component probabilities
            mus: (batch_size, num_modes) - component means
            sigmas: (batch_size, num_modes) - component standard deviations
            zero_prob: (batch_size,) - probability of zero earnings (for zero-inflation)
        """
        # ============ STEP 1: Embed categorical features ============
        # (batch_size, num_categorical_features, embedding_dim)
        embedded = self.embedding_layer(categorical_inputs)
        
        # ============ STEP 2: Fast Transformer ============
        # (batch_size, num_categorical_features, embedding_dim)
        transformed = self.transformer(embedded)
        
        # Flatten transformer output
        # (batch_size, num_categorical_features * embedding_dim)
        transformed_flat = transformed.reshape(transformed.size(0), -1)
        
        # ============ STEP 3: Concatenate with scalar features ============
        # (batch_size, transformer_output_dim + num_scalar_features)
        combined = torch.cat([transformed_flat, scalar_inputs], dim=1)
        
        # ============ STEP 4: Dense layers ============
        # Layer 1-4: 128 → 128 → 128 → 128
        layer1_out = self.layer1(combined)
        layer2_out = self.layer2(layer1_out)
        layer3_out = self.layer3(layer2_out)
        layer4_out = self.layer4(layer3_out)
        
        # ============ STEP 5: Output heads ============
        
        # Weights: softmax to ensure sum=1 and all ≥ 0
        weights_logits = self.weights_head(layer4_out)
        weights = torch.softmax(weights_logits, dim=-1)  # (batch_size, num_modes)
        
        # Means: unbounded
        mus = self.mu_head(layer4_out)  # (batch_size, num_modes)
        
        # Standard deviations: softplus + epsilon for numerical stability
        sigmas = F.softplus(self.sigma_head(layer4_out)) + 1e-6  # (batch_size, num_modes)
        
        # Zero-inflation probability: sigmoid to ensure [0, 1]
        zero_prob = torch.sigmoid(self.zero_prob_head(layer4_out)).squeeze(-1)  # (batch_size,)
        
        return weights, mus, sigmas, zero_prob


def gmm_nll_with_zero_inflation(y, weights, mus, sigmas, zero_prob, epsilon=1e-8):
    """
    Negative Log-Likelihood for zero-inflated Gaussian Mixture Model.
    
    Loss decomposes as:
    - For y_n = 0: log p(y_n=0) = log(π_0 + (1-π_0) * Φ_truncated(0))
                                  where π_0 = zero_prob
    - For y_n > 0: log p(y_n) = log[(1-π_0) * GMM(y_n | truncated)]
    
    Φ_truncated is the CDF of truncated Gaussian at 0.
    
    Args:
        y: Target earnings (batch_size,)
        weights: Mixture weights (batch_size, num_modes)
        mus: Component means (batch_size, num_modes)
        sigmas: Component stds (batch_size, num_modes)
        zero_prob: P(y=0) (batch_size,)
        epsilon: small value for numerical stability
    
    Returns:
        nll: scalar negative log-likelihood
    """
    
    # Create normal distributions for each mode
    # (batch_size, num_modes)
    normal_dists = Normal(mus, sigmas)
    
    # Compute log-probabilities under each component
    # (batch_size, num_modes)
    log_probs = normal_dists.log_prob(y.unsqueeze(-1))
    
    # Log-prob under mixture (before zero-inflation adjustment)
    # (batch_size,)
    mixture_log_prob = torch.logsumexp(
        torch.log(weights + epsilon) + log_probs,
        dim=-1
    )
    
    # ============ ZERO-INFLATION COMPONENT ============
    # For truncated Gaussian at 0, compute CDF at 0
    # Φ(0) = CDF at 0 for each component
    
    z = -mus / sigmas  # standardized score at y=0
    cdf_at_zero = torch.erfc(z / math.sqrt(2)) / 2.0  # CDF of standard normal
    
    # Mixture CDF at zero: weighted average of component CDFs
    mixture_cdf_at_zero = (weights * cdf_at_zero).sum(dim=-1)
    
    # Log-probability of zero earnings
    # P(y=0) = π_0 + (1-π_0) * Φ_mixture(0)
    # where Φ_mixture(0) = weighted average of CDFs at 0
    prob_zero = zero_prob + (1 - zero_prob) * mixture_cdf_at_zero
    log_prob_zero = torch.log(prob_zero + epsilon)
    
    # ============ COMBINE ZERO-INFLATION AND CONTINUOUS PART ============
    # For y > 0: log P(y) = log(1-π_0) + log(GMM(y | truncated))
    # For y = 0: log P(y) = log_prob_zero
    
    # Split batch into zero and non-zero
    mask_zero = (y < epsilon)  # boolean mask for zero earnings
    mask_nonzero = ~mask_zero
    
    # Initialize loss
    loss = torch.zeros_like(y)
    
    # For zero earnings: use zero-inflation component
    if mask_zero.any():
        loss[mask_zero] = -log_prob_zero[mask_zero]
    
    # For non-zero earnings: use (1-π_0) * truncated_gmm
    if mask_nonzero.any():
        # Log-prob of non-zero under truncated mixture
        # We approximate: log(truncated_pdf) ≈ log(untruncated_pdf) - log(1 - CDF(0))
        log_prob_nonzero = mixture_log_prob[mask_nonzero] - torch.log(1 - mixture_cdf_at_zero[mask_nonzero] + epsilon)
        loss[mask_nonzero] = -(torch.log(1 - zero_prob[mask_nonzero] + epsilon) + log_prob_nonzero)
    
    return loss.mean()


def gmm_nll_simple(y, weights, mus, sigmas, epsilon=1e-8):
    """
    Simple Negative Log-Likelihood for Gaussian Mixture Model (without zero-inflation).
    Use this if your data doesn't have many zeros.
    
    Args:
        y: Target earnings (batch_size,)
        weights: Mixture weights (batch_size, num_modes)
        mus: Component means (batch_size, num_modes)
        sigmas: Component stds (batch_size, num_modes)
        epsilon: small value for numerical stability
    
    Returns:
        nll: scalar negative log-likelihood
    """
    
    # Create mixture distribution
    # MixtureSameFamily expects:
    #   - mixing_distribution: Categorical over components
    #   - component_distribution: Normal distributions for each component
    
    # Expand dimensions for batch compatibility
    # weights: (batch_size, num_modes)
    # mus, sigmas: (batch_size, num_modes)
    
    try:
        # Create categorical distribution for mixture weights
        mix_dist = Categorical(weights)
        
        # Create normal distribution for each component
        comp_dist = Normal(mus, sigmas)
        
        # Create mixture distribution
        mixture = MixtureSameFamily(mix_dist, comp_dist)
        
        # Compute negative log-likelihood
        nll = -mixture.log_prob(y).mean()
        
        return nll
    
    except Exception as e:
        # Fallback: manual computation for stability
        log_probs = Normal(mus, sigmas).log_prob(y.unsqueeze(-1))
        mixture_log_prob = torch.logsumexp(
            torch.log(weights + epsilon) + log_probs,
            dim=-1
        )
        return -mixture_log_prob.mean()


# ============ EXAMPLE USAGE ============

if __name__ == "__main__":
    
    # Setup: Define categorical features (hour, day_of_week, region)
    categorical_cardinalities = {
        'hour': 24,                # 0-23 hours
        'day_of_week': 7,          # 0-6 (Mon-Sun)
        'region': 42,              # 42 hex-9 zones
    }
    
    num_scalar_features = 5  # e.g., surge_multiplier, request_count, weather, holiday, etc.
    num_modes = 3            # 3-mode Gaussian mixture
    batch_size = 32
    
    # Initialize model
    model = GMMEarningsModel(
        categorical_cardinalities=categorical_cardinalities,
        num_scalar_features=num_scalar_features,
        embedding_dim=16,
        hidden_dim=128,
        num_modes=num_modes,
        transformer_heads=4,
        dropout=0.1,
    )
    
    # Create dummy inputs
    categorical_inputs = {
        'hour': torch.randint(0, 24, (batch_size,)),
        'day_of_week': torch.randint(0, 7, (batch_size,)),
        'region': torch.randint(0, 42, (batch_size,)),
    }
    
    scalar_inputs = torch.randn(batch_size, num_scalar_features)
    targets = torch.abs(torch.randn(batch_size)) * 100  # synthetic earnings (non-negative)
    
    # Forward pass
    weights, mus, sigmas, zero_prob = model(categorical_inputs, scalar_inputs)
    
    print("✓ Model forward pass successful!")
    print(f"  Weights shape: {weights.shape} (should be [{batch_size}, {num_modes}])")
    print(f"  Means shape: {mus.shape}")
    print(f"  Stds shape: {sigmas.shape}")
    print(f"  Zero-prob shape: {zero_prob.shape}")
    print(f"  Weights sum (should be ~1.0): {weights[0].sum().item():.4f}")
    
    # Compute loss with zero-inflation
    loss_with_zero = gmm_nll_with_zero_inflation(targets, weights, mus, sigmas, zero_prob)
    print(f"\n✓ Loss (with zero-inflation): {loss_with_zero.item():.4f}")
    
    # Compute loss without zero-inflation (fallback)
    loss_simple = gmm_nll_simple(targets, weights, mus, sigmas)
    print(f"✓ Loss (simple GMM): {loss_simple.item():.4f}")
    
    # Print model summary
    print(f"\n✓ Model parameters: {sum(p.numel() for p in model.parameters()):,}")