"""Learning-rate schedules used by InstrAct pretraining."""

import math

from torch.optim.lr_scheduler import LambdaLR


def get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps,
    num_training_steps,
    num_cycles=0.5,
    last_epoch=-1,
):
    """Linear warmup followed by cosine decay to zero."""
    if num_training_steps <= 0:
        raise ValueError("num_training_steps must be positive")
    if num_warmup_steps < 0:
        raise ValueError("num_warmup_steps must be non-negative")

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return current_step / max(1, num_warmup_steps)
        progress = (
            (current_step - num_warmup_steps)
            / max(1, num_training_steps - num_warmup_steps)
        )
        progress = min(max(progress, 0.0), 1.0)
        return max(
            0.0,
            0.5 * (1.0 + math.cos(math.pi * 2.0 * num_cycles * progress)),
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)
