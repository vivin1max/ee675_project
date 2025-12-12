import torch
from torch.optim.lr_scheduler import _LRScheduler


class LinearLRScheduler(_LRScheduler):

    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        start_factor: float = 1.0,
        end_factor: float = 1.0,
        total_iters: int = 10000,
        last_epoch: int = -1,
    ):

        self.start_factor = start_factor
        self.end_factor = end_factor
        self.total_iters = total_iters
        super(LinearLRScheduler, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch == 0:
            return [base_lr * self.start_factor for base_lr in self.base_lrs]
        
        if self.last_epoch > self.total_iters:
            return [base_lr * self.end_factor for base_lr in self.base_lrs]
        
        # Linear interpolation
        factor = self.start_factor + (self.end_factor - self.start_factor) * (
            self.last_epoch / self.total_iters
        )
        return [base_lr * factor for base_lr in self.base_lrs]


class StepLRScheduler(_LRScheduler):

    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        step_size: int = 2000,
        gamma: float = 0.5,
        last_epoch: int = -1,
    ):

        self.step_size = step_size
        self.gamma = gamma
        super(StepLRScheduler, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch == 0:
            return self.base_lrs
        
        if self.last_epoch % self.step_size == 0:
            return [lr * self.gamma for lr in self.optimizer.param_groups[0]['lr']]
        
        return [group['lr'] for group in self.optimizer.param_groups]


def get_scheduler(
    scheduler_type: str,
    optimizer: torch.optim.Optimizer,
    **kwargs
):

    if scheduler_type == 'linear':
        return torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=kwargs.get('start_factor', 1.0),
            end_factor=kwargs.get('end_factor', 1.0),
            total_iters=kwargs.get('total_iters', 10000),
        )
    
    elif scheduler_type == 'step':
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=kwargs.get('step_size', 2000),
            gamma=kwargs.get('gamma', 0.5),
        )
    
    elif scheduler_type == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=kwargs.get('T_max', 10000),
            eta_min=kwargs.get('eta_min', 0.0),
        )
    
    elif scheduler_type == 'exponential':
        return torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=kwargs.get('gamma', 0.95),
        )
    
    elif scheduler_type == 'constant' or scheduler_type == 'none' or scheduler_type is None:
        # No scheduler - constant learning rate
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)
    
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


def get_current_lr(optimizer: torch.optim.Optimizer) -> float:

    return optimizer.param_groups[0]['lr']
