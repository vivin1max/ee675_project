from typing import Dict

import torch
import torch.nn as nn

from .base_ctc import BaseCTCModel, GaussianSmoothing


class BaselineGRU(BaseCTCModel):
    def __init__(
        self,
        neural_dim: int = 256,
        n_classes: int = 40,
        hidden_dim: int = 1024,
        layer_dim: int = 5,
        nDays: int = 24,
        dropout: float = 0.4,
        device: str = "cuda",
        strideLen: int = 4,
        kernelLen: int = 32,
        gaussianSmoothWidth: float = 2.0,
        bidirectional: bool = True,
        use_layernorm: bool = False,  # ABLATION: LayerNorm option
        **kwargs
    ):
        super(BaselineGRU, self).__init__(
            neural_dim=neural_dim,
            n_classes=n_classes,
            hidden_dim=hidden_dim,
            layer_dim=layer_dim,
            nDays=nDays,
            dropout=dropout,
            device=device,
            strideLen=strideLen,
            kernelLen=kernelLen,
            gaussianSmoothWidth=gaussianSmoothWidth,
        )
        
        self.bidirectional = bidirectional
        self.use_layernorm = use_layernorm
        
        # Input layer nonlinearity
        self.inputLayerNonlinearity = nn.Softsign()
        
        # Temporal stride/kernel windowing using unfold
        self.unfolder = nn.Unfold(
            (self.kernelLen, 1),
            dilation=1,
            padding=0,
            stride=self.strideLen
        )
        
        # Gaussian temporal smoothing
        self.gaussianSmoother = GaussianSmoothing(
            neural_dim, 20, self.gaussianSmoothWidth, dim=1
        )
        
        # Day-specific adaptation parameters
        # Initialize with identity matrices
        self.dayWeights = nn.Parameter(torch.randn(nDays, neural_dim, neural_dim))
        self.dayBias = nn.Parameter(torch.zeros(nDays, 1, neural_dim))
        
        for x in range(nDays):
            self.dayWeights.data[x, :, :] = torch.eye(neural_dim)
        
        # Multi-layer bidirectional GRU
        self.gru_decoder = nn.GRU(
            neural_dim * self.kernelLen,  # Input size after windowing
            hidden_dim,
            layer_dim,
            batch_first=True,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
        )
        
        # Initialize GRU weights (orthogonal for hidden-to-hidden, Xavier for input-to-hidden)
        for name, param in self.gru_decoder.named_parameters():
            if 'weight_hh' in name:
                nn.init.orthogonal_(param)
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
        
        # Per-day input layers (initialized near identity)
        for x in range(nDays):
            setattr(self, f"inpLayer{x}", nn.Linear(neural_dim, neural_dim))
        
        for x in range(nDays):
            thisLayer = getattr(self, f"inpLayer{x}")
            thisLayer.weight = nn.Parameter(
                thisLayer.weight + torch.eye(neural_dim)
            )
        
        # Optional LayerNorm after GRU (ABLATION)
        if self.use_layernorm:
            norm_dim = hidden_dim * 2 if self.bidirectional else hidden_dim
            self.layer_norm = nn.LayerNorm(norm_dim)
        else:
            self.layer_norm = None
        
        # Output layer for CTC (includes blank token)
        if self.bidirectional:
            self.fc_decoder_out = nn.Linear(hidden_dim * 2, n_classes + 1)
        else:
            self.fc_decoder_out = nn.Linear(hidden_dim, n_classes + 1)
    
    def forward(self, neuralInput: torch.Tensor, dayIdx: torch.Tensor) -> torch.Tensor:

        # Apply Gaussian smoothing (requires channel-first format)
        neuralInput = torch.permute(neuralInput, (0, 2, 1))  # (B, D, T)
        neuralInput = self.gaussianSmoother(neuralInput)
        neuralInput = torch.permute(neuralInput, (0, 2, 1))  # (B, T, D)
        
        # Apply day-specific adaptation layer
        dayWeights = torch.index_select(self.dayWeights, 0, dayIdx)  # (B, D, D)
        transformedNeural = torch.einsum(
            'btd,bdk->btk', neuralInput, dayWeights
        ) + torch.index_select(self.dayBias, 0, dayIdx)  # (B, T, D)
        
        transformedNeural = self.inputLayerNonlinearity(transformedNeural)
        
        # Apply temporal stride/kernel windowing
        # Unfold expects (B, C, H, W) format
        stridedInputs = torch.permute(
            self.unfolder(
                torch.unsqueeze(torch.permute(transformedNeural, (0, 2, 1)), 3)
            ),
            (0, 2, 1),
        )  # (B, T', D * kernelLen)
        
        # Initialize hidden state
        if self.bidirectional:
            h0 = torch.zeros(
                self.layer_dim * 2,
                transformedNeural.size(0),
                self.hidden_dim,
                device=self.device,
            ).requires_grad_()
        else:
            h0 = torch.zeros(
                self.layer_dim,
                transformedNeural.size(0),
                self.hidden_dim,
                device=self.device,
            ).requires_grad_()
        
        # Apply GRU
        hid, _ = self.gru_decoder(stridedInputs, h0.detach())
        
        # Apply LayerNorm if enabled (ABLATION)
        if self.layer_norm is not None:
            hid = self.layer_norm(hid)
        
        # Output layer
        seq_out = self.fc_decoder_out(hid)  # (B, T', n_classes + 1)
        
        return seq_out
    
    def get_model_config(self) -> Dict:

        return {
            'model_type': 'baseline_gru',
            'neural_dim': self.neural_dim,
            'n_classes': self.n_classes,
            'hidden_dim': self.hidden_dim,
            'layer_dim': self.layer_dim,
            'nDays': self.nDays,
            'dropout': self.dropout,
            'strideLen': self.strideLen,
            'kernelLen': self.kernelLen,
            'gaussianSmoothWidth': self.gaussianSmoothWidth,
            'bidirectional': self.bidirectional,
        }


# Alias for compatibility
GRUDecoder = BaselineGRU
