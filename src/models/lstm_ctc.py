from typing import Dict

import torch
import torch.nn as nn

from .base_ctc import BaseCTCModel, GaussianSmoothing


class LSTMCTC(BaseCTCModel):

    
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
        **kwargs
    ):

        super(LSTMCTC, self).__init__(
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
        
        # Multi-layer bidirectional LSTM 
        self.lstm_decoder = nn.LSTM(
            neural_dim * self.kernelLen,
            hidden_dim,
            layer_dim,
            batch_first=True,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
        )
        
        # Initialize LSTM weights
        for name, param in self.lstm_decoder.named_parameters():
            if 'weight_hh' in name:
                nn.init.orthogonal_(param)
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
        
        # Per-day input layers
        for x in range(nDays):
            setattr(self, f"inpLayer{x}", nn.Linear(neural_dim, neural_dim))
        
        for x in range(nDays):
            thisLayer = getattr(self, f"inpLayer{x}")
            thisLayer.weight = nn.Parameter(
                thisLayer.weight + torch.eye(neural_dim)
            )
        
        # Output layer for CTC
        if self.bidirectional:
            self.fc_decoder_out = nn.Linear(hidden_dim * 2, n_classes + 1)
        else:
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
        )
        
        # Initialize hidden and cell states for LSTM
        num_directions = 2 if self.bidirectional else 1
        h0 = torch.zeros(
            self.layer_dim * num_directions,
            transformedNeural.size(0),
            self.hidden_dim,
            device=self.device,
        ).requires_grad_()
        
        c0 = torch.zeros(
            self.layer_dim * num_directions,
            transformedNeural.size(0),
            self.hidden_dim,
            device=self.device,
        ).requires_grad_()
        
        # Apply LSTM
        hid, _ = self.lstm_decoder(stridedInputs, (h0.detach(), c0.detach()))
        
        # Output layer
        seq_out = self.fc_decoder_out(hid)
        
        return seq_out
    
    def get_model_config(self) -> Dict:

        return {
            'model_type': 'lstm_ctc',
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
