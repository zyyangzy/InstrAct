"""Checkpoint save and resume helpers."""

from __future__ import annotations

import os
import re
from pathlib import Path

import torch


EPOCH_PATTERN = re.compile(r"epoch(\d+)\.pth$")


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def latest_checkpoint(output_dir):
    output_dir = Path(output_dir)
    checkpoints = []
    for path in output_dir.glob("epoch*.pth"):
        match = EPOCH_PATTERN.fullmatch(path.name)
        if match:
            checkpoints.append((int(match.group(1)), path))
    return max(checkpoints, default=(None, None))[1]


def resolve_resume_path(resume, output_dir):
    if not resume:
        return None
    if resume == "auto":
        path = latest_checkpoint(output_dir)
        if path is None:
            raise FileNotFoundError(
                f"no epoch checkpoint found in {Path(output_dir)}"
            )
        return path
    path = Path(resume).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"resume checkpoint not found: {path}")
    return path


def save_checkpoint(
    output_dir,
    epoch,
    global_step,
    model,
    optimizer,
    scheduler,
    scaler,
    args,
    config,
    keep_last=5,
):
    """Atomically save one complete training state on rank zero."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"epoch{epoch:04d}.pth"
    temporary = output_dir / f".{destination.name}.tmp"
    state = {
        "epoch": int(epoch),
        "global_step": int(global_step),
        "model": unwrap_model(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "args": vars(args),
        "config": config,
    }
    torch.save(state, temporary)
    os.replace(temporary, destination)

    if keep_last > 0:
        checkpoints = []
        for path in output_dir.glob("epoch*.pth"):
            match = EPOCH_PATTERN.fullmatch(path.name)
            if match:
                checkpoints.append((int(match.group(1)), path))
        for _, old_path in sorted(checkpoints)[:-keep_last]:
            old_path.unlink()
    return destination


def load_training_checkpoint(
    checkpoint_path,
    model,
    optimizer=None,
    scheduler=None,
    scaler=None,
):
    """Restore a complete training state and return ``(epoch, global_step)``."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "model" not in checkpoint:
        raise KeyError("training checkpoint does not contain 'model'")
    unwrap_model(model).load_state_dict(checkpoint["model"], strict=True)

    if optimizer is not None:
        if "optimizer" not in checkpoint:
            raise KeyError("training checkpoint does not contain 'optimizer'")
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None:
        if "scheduler" not in checkpoint:
            raise KeyError("training checkpoint does not contain 'scheduler'")
        scheduler.load_state_dict(checkpoint["scheduler"])
    if (
        scaler is not None
        and checkpoint.get("scaler") is not None
    ):
        scaler.load_state_dict(checkpoint["scaler"])

    epoch = int(checkpoint.get("epoch", -1)) + 1
    global_step = int(checkpoint.get("global_step", 0))
    return epoch, global_step
