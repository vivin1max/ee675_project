import torch
import torch.nn as nn


class CoordinatedDropout(nn.Module):

    
    def __init__(self, p: float = 0.1):

        super(CoordinatedDropout, self).__init__()
        self.p = p
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if not self.training or self.p == 0:
            return x
        
        # Create channel mask (batch, 1, features)
        # Same mask is applied across all time steps
        batch_size, time_steps, n_features = x.shape
        
        mask = torch.bernoulli(
            torch.ones(batch_size, 1, n_features, device=x.device) * (1 - self.p)
        )
        
        # Scale by dropout probability to maintain expected value
        mask = mask / (1 - self.p)
        
        return x * mask
    
    def extra_repr(self) -> str:
        return f'p={self.p}'


class SpatialDropout1D(nn.Module):

    
    def __init__(self, p: float = 0.1):

        super(SpatialDropout1D, self).__init__()
        self.dropout = nn.Dropout2d(p)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # Reshape to (batch, features, time, 1) for Dropout2d
        x = x.permute(0, 2, 1).unsqueeze(-1)  # (B, F, T, 1)
        x = self.dropout(x)
        x = x.squeeze(-1).permute(0, 2, 1)  # (B, T, F)
        return x


class TemporalDropout(nn.Module):

    
    def __init__(self, p: float = 0.1):

        super(TemporalDropout, self).__init__()
        self.p = p
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if not self.training or self.p == 0:
            return x
        
        # Create temporal mask (batch, time, 1)
        # Same mask is applied across all features
        batch_size, time_steps, n_features = x.shape
        
        mask = torch.bernoulli(
            torch.ones(batch_size, time_steps, 1, device=x.device) * (1 - self.p)
        )
        
        # Scale by dropout probability
        mask = mask / (1 - self.p)
        
        return x * mask
    
    def extra_repr(self) -> str:
        return f'p={self.p}'


def apply_coordinated_dropout(
    x: torch.Tensor,
    p: float = 0.1,
    training: bool = True
) -> torch.Tensor:

    if not training or p == 0:
        return x
    
    batch_size, time_steps, n_features = x.shape
    
    mask = torch.bernoulli(
        torch.ones(batch_size, 1, n_features, device=x.device) * (1 - p)
    )
    mask = mask / (1 - p)
    
    return x * mask
