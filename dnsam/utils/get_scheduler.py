from ..scheduler import *
import torch

def get_scheduler(optimizer, cfg):
    if cfg['trainer']['sch'] == 'step_lr':
        return StepLR(
            optimizer,
            learning_rate=cfg['model']['lr'],
            total_epochs=200
        )
    elif cfg['trainer']['sch'] == "cosine_annealing_lr":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg['trainer']['epochs']
    )
    elif cfg['trainer']['sch'] == "linear_warmup_cosine_annealing_lr":
        sch = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=cfg['trainer']['warmup'],
            max_epochs=cfg['trainer']['epochs']
        )
        sch.step()
        return sch
    elif cfg['trainer']['sch'] == "constant_lr":
        return torch.optim.lr_scheduler.ConstantLR(
        optimizer,
        factor=cfg['trainer']['factor'],
        total_iters=cfg['trainer']['total_iters']
    )
    elif cfg['trainer']['sch'] == "torch_step_lr":
        return torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=cfg['trainer']['step_size'],
        gamma=cfg['trainer']['gamma']
    )
    elif cfg['trainer']['sch'] == "lambda1":
        return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda epoch: 1/(epoch+1)
    )
    elif cfg['trainer']['sch'] == "lambda2":
        return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda epoch: 1/((epoch+1)**1/2)
    )
    elif cfg['trainer']['sch'] == "exponential":
        return torch.optim.lr_scheduler.ExponentialLR(
        optimizer,
        gamma=0.97
    )
    else:
        raise ValueError("Invalid scheduler!!!")