from abc import ABC, abstractmethod
from typing import Dict, Tuple

import torch
import torch.nn as nn


class BaseCTCModel(ABC, nn.Module):

    
    def __init__(
        self,
        neural_dim: int,
        n_classes: int,
        hidden_dim: int,
        layer_dim: int,
        nDays: int = 24,
        dropout: float = 0.4,
        device: str = "cuda",
        strideLen: int = 4,
        kernelLen: int = 32,
        gaussianSmoothWidth: float = 2.0,
        **kwargs
    ):

        super(BaseCTCModel, self).__init__()
        
        self.neural_dim = neural_dim
        self.n_classes = n_classes
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.nDays = nDays
        self.dropout = dropout
        self.device = device
        self.strideLen = strideLen
        self.kernelLen = kernelLen
        self.gaussianSmoothWidth = gaussianSmoothWidth
    
    @abstractmethod
    def forward(
        self,
        neuralInput: torch.Tensor,
        dayIdx: torch.Tensor
    ) -> torch.Tensor:

        pass
    
    @abstractmethod
    def get_model_config(self) -> Dict:

        pass
    
    def compute_output_length(self, input_length: torch.Tensor) -> torch.Tensor:

        return ((input_length - self.kernelLen) / self.strideLen).long()
    
    def count_parameters(self) -> int:

        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_summary(self) -> str:

        config = self.get_model_config()
        summary = [
            f"Model: {self.__class__.__name__}",
            f"Parameters: {self.count_parameters():,}",
            "\nConfiguration:",
        ]
        for key, value in config.items():
            summary.append(f"  {key}: {value}")
        return "\n".join(summary)


class GaussianSmoothing(nn.Module):

    
    def __init__(self, channels: int, kernel_size: int, sigma: float, dim: int = 1):

        super(GaussianSmoothing, self).__init__()
        
        import math
        import numbers
        
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim
        
        # Create Gaussian kernel
        kernel = 1
        meshgrids = torch.meshgrid(
            [torch.arange(size, dtype=torch.float32) for size in kernel_size],
            indexing='ij'
        )
        
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= (
                1 / (std * math.sqrt(2 * math.pi))
                * torch.exp(-(((mgrid - mean) / std) ** 2) / 2)
            )
        
        # Normalize
        kernel = kernel / torch.sum(kernel)
        
        # Reshape for depthwise convolution
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))
        
        self.register_buffer('weight', kernel)
        self.groups = channels
        
        # Select convolution function
        if dim == 1:
            self.conv = nn.functional.conv1d
        elif dim == 2:
            self.conv = nn.functional.conv2d
        elif dim == 3:
            self.conv = nn.functional.conv3d
        else:
            raise RuntimeError(f"Only 1, 2, 3 dimensions supported. Got {dim}.")
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:

        return self.conv(input, weight=self.weight, groups=self.groups, padding='same')
