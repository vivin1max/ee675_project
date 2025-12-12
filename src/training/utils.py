import torch
import torch.nn as nn


class WhiteNoiseAugmentation:
    
    def __init__(self, std: float = 0.8):

        self.std = std
    
    def __call__(self, X: torch.Tensor, device: str = 'cuda') -> torch.Tensor:

        if self.std > 0:
            return X + torch.randn(X.shape, device=device) * self.std
        return X


class ConstantOffsetAugmentation:
    
    def __init__(self, std: float = 0.2):

        self.std = std
    
    def __call__(self, X: torch.Tensor, device: str = 'cuda') -> torch.Tensor:

        if self.std > 0:
            offset = torch.randn([X.shape[0], 1, X.shape[2]], device=device) * self.std
            return X + offset
        return X


def set_seed(seed: int):

    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def count_parameters(model: nn.Module) -> int:

    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device(prefer_cuda: bool = True) -> str:

    if prefer_cuda and torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
    batch: int,
    best_cer: float,
    save_path: str,
):

    torch.save({
        'epoch': epoch,
        'batch': batch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_cer': best_cer,
        'model_config': model.get_model_config() if hasattr(model, 'get_model_config') else {},
    }, save_path)


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer = None,
    scheduler: torch.optim.lr_scheduler._LRScheduler = None,
) -> dict:

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return {
        'epoch': checkpoint.get('epoch', 0),
        'batch': checkpoint.get('batch', 0),
        'best_cer': checkpoint.get('best_cer', float('inf')),
        'model_config': checkpoint.get('model_config', {}),
    }
