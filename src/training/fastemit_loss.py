import torch
import torch.nn as nn
import torch.nn.functional as F


class FastEmitCTCLoss(nn.Module):

    
    def __init__(
        self,
        blank: int = 0,
        reduction: str = 'mean',
        zero_infinity: bool = True,
        fastemit_lambda: float = 0.0,
    ):

        super(FastEmitCTCLoss, self).__init__()
        self.blank = blank
        self.reduction = reduction
        self.zero_infinity = zero_infinity
        self.fastemit_lambda = fastemit_lambda
        
        # Standard CTC loss - use 'mean' reduction since per-sample losses are too large
        self.ctc_loss = nn.CTCLoss(
            blank=blank,
            reduction='mean',  # Use mean reduction for proper scale
            zero_infinity=zero_infinity,
        )
    
    def forward(
        self,
        log_probs: torch.Tensor,
        targets: torch.Tensor,
        input_lengths: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> torch.Tensor:

        # Debug: check shapes
        batch_size = input_lengths.shape[0]
        
        # Ensure log_probs is in (T, N, C) format for CTC loss
        if log_probs.dim() == 3:
            # If first dimension doesn't match batch size, it's likely (N, T, C) format
            if log_probs.shape[0] == batch_size and log_probs.shape[1] != batch_size:
                log_probs = log_probs.permute(1, 0, 2)  # (N, T, C) -> (T, N, C)
        
        # Compute standard CTC loss
        ctc_loss = self.ctc_loss(log_probs, targets, input_lengths, target_lengths)
        
        # Add FastEmit regularization if enabled
        if self.fastemit_lambda > 0:
            # FastEmit: penalize blank emissions to encourage faster emission
            # The regularization is the average probability of blank across time
            
            # log_probs is (T, N, C) and contains log probabilities
            # Extract log probability of blank token
            log_blank_probs = log_probs[:, :, self.blank]  # (T, N)
            
            # Compute FastEmit regularization averaged over batch
            # Average blank probability over the actual sequence length
            fastemit_reg = 0.0
            batch_size = log_probs.shape[1]
            for i in range(batch_size):
                seq_len = input_lengths[i].item()
                # Average blank probability over sequence
                blank_probs_seq = torch.exp(log_blank_probs[:seq_len, i])
                fastemit_reg += blank_probs_seq.mean()
            
            fastemit_reg = fastemit_reg / batch_size
            
            # Combine losses (both are scalars now)
            loss = ctc_loss + self.fastemit_lambda * fastemit_reg
        else:
            loss = ctc_loss
        return loss
    
    def extra_repr(self) -> str:
        return (
            f'blank={self.blank}, '
            f'reduction={self.reduction}, '
            f'zero_infinity={self.zero_infinity}, '
            f'fastemit_lambda={self.fastemit_lambda}'
        )


def fastemit_regularization(
    log_probs: torch.Tensor,
    input_lengths: torch.Tensor,
    blank: int = 0,
) -> torch.Tensor:

    # Ensure (T, N, C) format
    if log_probs.shape[0] < log_probs.shape[1]:
        log_probs = log_probs.permute(1, 0, 2)
    
    probs = torch.exp(log_probs)
    blank_probs = probs[:, :, blank]  # (T, N)
    
    # Average blank probability weighted by sequence lengths
    reg = 0
    for i in range(log_probs.shape[1]):
        seq_len = input_lengths[i].item()
        reg += blank_probs[:seq_len, i].mean()
    
    return reg / log_probs.shape[1]


class CTCLossWithOptions(nn.Module):

    
    def __init__(
        self,
        blank: int = 0,
        reduction: str = 'mean',
        zero_infinity: bool = True,
        fastemit_lambda: float = 0.0,
        label_smoothing: float = 0.0,
    ):

        super(CTCLossWithOptions, self).__init__()
        self.blank = blank
        self.reduction = reduction
        self.zero_infinity = zero_infinity
        self.fastemit_lambda = fastemit_lambda
        self.label_smoothing = label_smoothing
        
        self.ctc_loss = nn.CTCLoss(
            blank=blank,
            reduction=reduction,
            zero_infinity=zero_infinity,
        )
    
    def forward(
        self,
        log_probs: torch.Tensor,
        targets: torch.Tensor,
        input_lengths: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> torch.Tensor:

        
        # Ensure correct format
        if log_probs.dim() == 3 and log_probs.shape[0] != input_lengths.max():
            log_probs_ctc = log_probs.permute(1, 0, 2)
        else:
            log_probs_ctc = log_probs
        
        # Standard CTC loss
        loss = self.ctc_loss(log_probs_ctc, targets, input_lengths, target_lengths)
        
        # Add FastEmit regularization
        if self.fastemit_lambda > 0:
            fastemit_reg = fastemit_regularization(log_probs_ctc, input_lengths, self.blank)
            loss = loss + self.fastemit_lambda * fastemit_reg
        
        return loss
