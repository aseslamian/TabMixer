import torch
import torch.nn as nn
import torch.nn.functional as F

class TabMixer(nn.Module):
    """
    A TabMixer block that mixes information along token and feature dimensions.
    Expects input shape: [batch, dim_features, dim_tokens].
    """
    def __init__(
        self,
        dim_tokens: int,
        dim_features: int,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5
    ):
        super().__init__()

        self.dim_tokens = dim_tokens
        self.dim_features = dim_features

        self.layer_norm1 = nn.LayerNorm(dim_features, eps=layer_norm_eps)
        self.layer_norm2 = nn.LayerNorm(dim_features, eps=layer_norm_eps)

        self.mlp1 = nn.Sequential(
            nn.Linear(dim_tokens, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, dim_tokens)
        )

        self.mlp2 = nn.Sequential(
            nn.Linear(dim_features, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, dim_features)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: Tensor of shape [batch, dim_features, dim_tokens]
        """
        x_res = x.clone()

        # Normalize along feature dimension
        y = self.layer_norm1(x)

        # Process tokens: transpose so tokens are the last dimension
        y_t = y.transpose(-1, -2)         # [batch, dim_tokens, dim_features]
        y_t = self.mlp1(y_t)               # process tokens
        y_t = y_t.transpose(-1, -2)        # back to [batch, dim_features, dim_tokens]
        y_t = F.gelu(y_t)

        # Process features with mlp2
        x2 = self.mlp2(y)

        # Elementwise combination, further norm, activation, and residual connection
        x_out = x2 * y_t
        x_out = self.layer_norm2(x_out)
        x_out = F.silu(x_out)
        x_out = x_out + x_res

        return x_out
