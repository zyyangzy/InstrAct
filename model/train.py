"""InstrAct training steps."""

from __future__ import annotations

import time

import torch
import torch.distributed as dist
from tqdm import tqdm


def _distributed():
    return dist.is_available() and dist.is_initialized()


def _move_batch(batch, device):
    return {
        key: (
            value.to(device=device, non_blocking=True)
            if torch.is_tensor(value) else value
        )
        for key, value in batch.items()
    }


def _mean_across_ranks(value):
    value = value.detach()
    if not _distributed():
        return value
    value = value.clone()
    dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return value / dist.get_world_size()


def train_one_epoch(
    model,
    data_loader,
    optimizer,
    scheduler,
    scaler,
    device,
    epoch,
    global_step,
    amp_dtype,
    gradient_clip_norm=None,
    log_every=20,
    max_steps=None,
    rank=0,
    wandb_run=None,
):
    """Train one epoch and return ``(mean_loss, new_global_step)``."""
    model.train()
    if hasattr(data_loader.sampler, "set_epoch"):
        data_loader.sampler.set_epoch(epoch)

    running_loss = 0.0
    running_terms = {}
    progress = tqdm(
        data_loader,
        desc=f"Train {epoch + 1}",
        disable=rank != 0,
    )
    for step, batch in enumerate(progress):
        if max_steps is not None and step >= max_steps:
            break
        started = time.time()
        batch = _move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=amp_dtype is not None,
        ):
            loss_output = model(
                batch["video"].float(),
                batch["text"],
                hard_negative_mask=batch["text_mask"],
                num_verbs=batch["num_verbs"],
                action_mask=batch["action_mask"],
                idx=batch["idx"],
                return_loss_dict=True,
            )
            loss = loss_output["loss"]

        finite = torch.isfinite(loss.detach()).to(dtype=torch.int32)
        if _distributed():
            dist.all_reduce(finite, op=dist.ReduceOp.MIN)
        if not finite.item():
            raise FloatingPointError(
                f"non-finite loss at epoch {epoch}, step {step}"
            )

        optimizer_stepped = True
        if scaler is not None and scaler.is_enabled():
            scale_before = scaler.get_scale()
            scaler.scale(loss).backward()
            if gradient_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), gradient_clip_norm
                )
            scaler.step(optimizer)
            scaler.update()
            optimizer_stepped = scaler.get_scale() >= scale_before
        else:
            loss.backward()
            if gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), gradient_clip_norm
                )
            optimizer.step()
        if optimizer_stepped:
            scheduler.step()
            global_step += 1

        reduced_loss = float(_mean_across_ranks(loss).cpu())
        running_loss += reduced_loss
        reduced_terms = {
            name: float(_mean_across_ranks(value).cpu())
            for name, value in loss_output.items()
            if name != "loss"
        }
        for name, value in reduced_terms.items():
            running_terms[name] = running_terms.get(name, 0.0) + value
        if rank == 0:
            postfix = {
                "loss": f"{reduced_loss:.4f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
            }
            postfix.update(
                (name, f"{value:.4f}")
                for name, value in reduced_terms.items()
            )
            progress.set_postfix(postfix)
            if (
                wandb_run is not None
                and optimizer_stepped
                and global_step % log_every == 0
            ):
                wandb_run.log(
                    {
                        "train/loss": reduced_loss,
                        "train/lr": optimizer.param_groups[0]["lr"],
                        "train/step_seconds": time.time() - started,
                        "train/epoch": epoch,
                        **{
                            f"train/{name}_loss": value
                            for name, value in reduced_terms.items()
                        },
                    },
                    step=global_step,
                )

    completed_steps = min(
        len(data_loader),
        max_steps if max_steps is not None else len(data_loader),
    )
    mean_terms = {
        name: value / max(1, completed_steps)
        for name, value in running_terms.items()
    }
    return running_loss / max(1, completed_steps), global_step, mean_terms
