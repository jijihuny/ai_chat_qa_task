from torch.optim.lr_scheduler import LambdaLR
from torch.optim import Optimizer
from trl import SFTTrainer
import math
from functools import partial
from typing import Self, Any
from overrides import overrides


def _get_cosine_with_hard_restarts_and_decreasing_schedule_with_warmup_lr_lambda(
    current_step: int,
    *,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: int,
    gamma: float
):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    progress = float(current_step - num_warmup_steps) / float(
        max(1, num_training_steps - num_warmup_steps)
    )
    if progress >= 1.0:
        return 0.0
    cycle = (float(num_cycles) * progress) % 1.0
    decay = float(gamma) ** cycle
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * cycle))) * decay


def get_cosine_with_hard_restarts__and_decreasing_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: int = 1,
    last_epoch: int = -1,
    gamma: float = 0.5,
):
    lr_lambda = partial(
        _get_cosine_with_hard_restarts_and_decreasing_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
        gamma=gamma,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)


class CosineScheduleTrainer(SFTTrainer):
    def __init__(self: Self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

    @overrides
    def create_scheduler(self, num_training_steps: int, optimizer: Optimizer = None):
        if self.lr_scheduler is None:
            self.lr_scheduler = (
                get_cosine_with_hard_restarts__and_decreasing_schedule_with_warmup(
                    optimizer=self.optimizer if optimizer is None else optimizer,
                    num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                    num_training_steps=num_training_steps,
                    **self.args.lr_scheduler_kwargs
                )
            )
            self._created_lr_scheduler = True

        return self.lr_scheduler
