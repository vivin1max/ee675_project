from typing import Dict

import torch
import torch.nn as nn

from .base_ctc import BaseCTCModel, GaussianSmoothing


class TransformerCTC(BaseCTCModel):

    
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
        nhead: int = 8,
        dim_feedforward: int = 2048,
        **kwargs
    ):

        super(TransformerCTC, self).__init__(
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
        
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        
        # Input layer nonlinearity
        self.inputLayerNonlinearity = nn.Softsign()
        
        # Temporal stride/kernel windowing
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
        self.dayWeights = nn.Parameter(torch.randn(nDays, neural_dim, neural_dim))
        self.dayBias = nn.Parameter(torch.zeros(nDays, 1, neural_dim))
        
        for x in range(nDays):
            self.dayWeights.data[x, :, :] = torch.eye(neural_dim)
        
        # Per-day input layers
        for x in range(nDays):
            setattr(self, f"inpLayer{x}", nn.Linear(neural_dim, neural_dim))
        
        for x in range(nDays):
            thisLayer = getattr(self, f"inpLayer{x}")
            thisLayer.weight = nn.Parameter(
                thisLayer.weight + torch.eye(neural_dim)
            )
        
        # Project windowed input to transformer dimension
        self.input_projection = nn.Linear(neural_dim * self.kernelLen, hidden_dim)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout, max_len=5000)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=layer_dim,
            norm=nn.LayerNorm(hidden_dim)
        )
        
        # Output layer for CTC
        self.fc_decoder_out = nn.Linear(hidden_dim, n_classes + 1)
    
    def forward(self, neuralInput: torch.Tensor, dayIdx: torch.Tensor) -> torch.Tensor:

        # Apply Gaussian smoothing
        neuralInput = torch.permute(neuralInput, (0, 2, 1))
        neuralInput = self.gaussianSmoother(neuralInput)
        neuralInput = torch.permute(neuralInput, (0, 2, 1))
        
        # Apply day-specific adaptation
        dayWeights = torch.index_select(self.dayWeights, 0, dayIdx)
        transformedNeural = torch.einsum(
            'btd,bdk->btk', neuralInput, dayWeights
        ) + torch.index_select(self.dayBias, 0, dayIdx)
        
        transformedNeural = self.inputLayerNonlinearity(transformedNeural)
        
        # Apply temporal stride/kernel windowing
        stridedInputs = torch.permute(
            self.unfolder(
                torch.unsqueeze(torch.permute(transformedNeural, (0, 2, 1)), 3)
            ),
            (0, 2, 1),
        )  # (B, T', D * kernelLen)
        
        # Project to transformer dimension
        x = self.input_projection(stridedInputs)  # (B, T', hidden_dim)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Apply transformer encoder
        x = self.transformer_encoder(x)  # (B, T', hidden_dim)
        
        # Output layer
        seq_out = self.fc_decoder_out(x)  # (B, T', n_classes + 1)
        
        return seq_out
    
    def get_model_config(self) -> Dict:

        return {
            'model_type': 'transformer_ctc',
            'neural_dim': self.neural_dim,
            'n_classes': self.n_classes,
            'hidden_dim': self.hidden_dim,
            'layer_dim': self.layer_dim,
            'nDays': self.nDays,
            'dropout': self.dropout,
            'strideLen': self.strideLen,
            'kernelLen': self.kernelLen,
            'gaussianSmoothWidth': self.gaussianSmoothWidth,
            'nhead': self.nhead,
            'dim_feedforward': self.dim_feedforward,
        }


class PositionalEncoding(nn.Module):

    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):

        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
