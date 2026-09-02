"""InstrAct pretraining entry point using the original mp.spawn DDP pattern."""

from __future__ import annotations

import json
import logging
import os
import random
import socket
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from args import get_args
from data.eval_loader import HT100MEvalLoader
from data.train_loader import HT100MDataLoader
from validation import evaluate
from models import build_instract
from models.adapters.internvideo import InternVideoAdapter
from train import train_one_epoch
from utils.checkpoint import (
    load_training_checkpoint,
    resolve_resume_path,
    save_checkpoint,
)
from utils.scheduler import get_cosine_schedule_with_warmup


def _find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def _setup_spawn_environment():
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(_find_free_port()))


def _seed_everything(seed, rank):
    seed = int(seed) + int(rank)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _amp_components(amp_mode, device):
    if amp_mode == "off":
        return None, torch.amp.GradScaler(
            "cuda", enabled=False
        )
    if amp_mode == "fp16":
        if device.type != "cuda":
            raise ValueError("fp16 training requires CUDA; use --amp off or bf16")
        return torch.float16, torch.amp.GradScaler("cuda", enabled=True)
    dtype = torch.bfloat16
    return dtype, torch.amp.GradScaler("cuda", enabled=False)


def _create_logger(output_dir, rank):
    logger = logging.getLogger(f"instract.rank{rank}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    logger.addHandler(stream)
    if rank == 0:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(
            Path(output_dir) / "train.log",
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


def _create_datasets(args, config):
    common = {
        "video_root": args.video_root,
        "min_time": config.data.min_time,
        "fps": config.data.fps,
        "num_frames": config.data.num_frames,
        "size": config.backbone.input_resolution,
        "crop_only": config.data.crop_only,
        "max_words": config.data.max_text_length,
        "uniform_window": config.data.uniform_window,
    }
    train_dataset = None
    if args.train_annotations:
        train_dataset = HT100MDataLoader(
            caption_json=args.train_annotations,
            num_hn=config.data.max_negatives,
            num_vp=config.data.max_verbs,
            hard_negative_type=args.hard_negative_type,
            **common,
        )
    eval_dataset = None
    if args.val_annotations:
        eval_dataset = HT100MEvalLoader(
            annotation_json=args.val_annotations,
            max_negatives=(
                args.eval_max_negatives
                if args.eval_max_negatives is not None
                else config.data.get("eval_max_negatives", 9)
            ),
            max_order_negatives=config.data.max_order_negatives,
            benchmark=args.eval_benchmark,
            **common,
        )
        if args.max_eval_samples is not None:
            if args.max_eval_samples <= 0:
                raise ValueError("--max-eval-samples must be positive")
            eval_dataset = torch.utils.data.Subset(
                eval_dataset,
                range(min(args.max_eval_samples, len(eval_dataset))),
            )
    return train_dataset, eval_dataset


def _create_loaders(
    args,
    train_dataset,
    eval_dataset,
    distributed,
    rank,
    world_size,
):
    train_sampler = (
        DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True,
            seed=args.seed,
        )
        if distributed and train_dataset is not None else None
    )
    eval_sampler = (
        DistributedSampler(
            eval_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False,
        )
        if distributed and eval_dataset is not None else None
    )

    local_batch_size = max(
        1, args.batch_size // world_size
    ) if distributed else args.batch_size
    local_val_batch_size = max(
        1, args.val_batch_size // world_size
    ) if distributed else args.val_batch_size
    local_workers = max(
        1, args.workers // world_size
    ) if distributed and args.workers > 0 else args.workers

    train_loader = None
    if train_dataset is not None:
        train_loader = DataLoader(
            train_dataset,
            batch_size=local_batch_size,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            num_workers=local_workers,
            pin_memory=args.pin_memory,
            drop_last=True,
            persistent_workers=local_workers > 0,
        )
    eval_loader = None
    if eval_dataset is not None:
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=local_val_batch_size,
            shuffle=False,
            sampler=eval_sampler,
            num_workers=local_workers,
            pin_memory=args.pin_memory,
            drop_last=False,
            persistent_workers=local_workers > 0,
        )
    return train_loader, eval_loader, local_batch_size


def _barrier(gpu):
    if dist.is_available() and dist.is_initialized():
        dist.barrier(device_ids=[gpu])


def _init_wandb(args, config, rank):
    if not args.wandb or rank != 0:
        return None
    try:
        import wandb
    except ImportError as error:
        raise ImportError("--wandb requires the wandb package") from error
    return wandb.init(
        project=args.wandb_project,
        name=args.run_name,
        config={
            "args": vars(args),
            "model": OmegaConf.to_container(config, resolve=True),
        },
    )


def main():
    args = get_args()
    if args.multiprocessing_distributed:
        gpu_count = torch.cuda.device_count()
        if gpu_count < 1:
            raise RuntimeError(
                "--multiprocessing-distributed requires at least one CUDA GPU"
            )
        _setup_spawn_environment()
        mp.spawn(
            main_worker,
            nprocs=gpu_count,
            args=(gpu_count, args),
            join=True,
        )
    else:
        main_worker(0 if torch.cuda.is_available() else None, 1, args)


def main_worker(gpu, processes_per_node, args):
    # Each rank also launches ffmpeg DataLoader workers. Large default BLAS
    # pools otherwise oversubscribe the host when two four-GPU jobs coexist.
    torch.set_num_threads(1)
    distributed = bool(args.multiprocessing_distributed)
    rank = int(gpu) if distributed else 0
    world_size = int(processes_per_node) if distributed else 1
    args.rank = rank
    args.world_size = world_size
    args.gpu = gpu
    args.distributed = distributed

    if distributed:
        torch.cuda.set_device(gpu)
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=world_size,
            rank=rank,
        )

    device = (
        torch.device("cuda", gpu)
        if torch.cuda.is_available() and args.device.startswith("cuda")
        else torch.device("cpu")
    )
    output_dir = Path(args.output_dir) / args.run_name
    logger = _create_logger(output_dir, rank)
    _seed_everything(args.seed, rank)

    try:
        config_path = Path(args.model_config)
        if not config_path.is_file():
            config_path = Path(__file__).resolve().parent / args.model_config
        if not config_path.is_file():
            raise FileNotFoundError(
                f"model config not found: {args.model_config}"
            )
        config = OmegaConf.load(config_path)
        OmegaConf.resolve(config)
        model = build_instract(config, args).to(device)
        if distributed:
            model = DistributedDataParallel(
                model,
                device_ids=[gpu],
                output_device=gpu,
                find_unused_parameters=args.find_unused_parameters,
                broadcast_buffers=False,
            )

        train_dataset, eval_dataset = _create_datasets(args, config)
        if not args.eval_only and train_dataset is None:
            raise ValueError("training requires --train-annotations")
        train_loader, eval_loader, local_batch_size = _create_loaders(
            args,
            train_dataset,
            eval_dataset,
            distributed,
            rank,
            world_size,
        )

        amp_dtype, scaler = _amp_components(args.amp, device)
        optimizer = None
        scheduler = None
        if not args.eval_only:
            core_model = model.module if hasattr(model, "module") else model
            parameter_groups = InternVideoAdapter.parameter_groups(
                core_model,
                config,
                args.learning_rate,
                args.weight_decay,
            )
            optimizer = torch.optim.AdamW(
                parameter_groups,
                betas=(0.9, 0.98),
                eps=1e-6,
            )
            steps_per_epoch = (
                min(len(train_loader), args.max_steps_per_epoch)
                if args.max_steps_per_epoch is not None
                else len(train_loader)
            )
            if steps_per_epoch <= 0:
                raise ValueError("training requires at least one batch per epoch")
            schedule_epochs = (
                args.scheduler_total_epochs
                if args.scheduler_total_epochs is not None else args.epochs
            )
            if schedule_epochs <= 0:
                raise ValueError("--scheduler-total-epochs must be positive")
            total_steps = steps_per_epoch * schedule_epochs
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                args.warmup_steps,
                total_steps,
            )

        start_epoch = 0
        global_step = 0
        resume_path = resolve_resume_path(args.resume, output_dir)
        if resume_path is not None:
            start_epoch, global_step = load_training_checkpoint(
                resume_path,
                model,
                optimizer,
                scheduler,
                scaler if not args.eval_only else None,
            )
            logger.info(
                "Resumed %s at epoch=%d global_step=%d",
                resume_path,
                start_epoch,
                global_step,
            )
        if distributed:
            _barrier(gpu)

        wandb_run = _init_wandb(args, config, rank)
        if rank == 0:
            output_dir.mkdir(parents=True, exist_ok=True)
            OmegaConf.save(config, output_dir / "config.yaml")
            with (output_dir / "args.json").open("w", encoding="utf-8") as handle:
                json.dump(vars(args), handle, indent=2)
            logger.info(
                "world_size=%d local_batch=%d effective_batch=%d train_samples=%d",
                world_size,
                local_batch_size,
                local_batch_size * world_size,
                len(train_dataset) if train_dataset is not None else 0,
            )

        if args.eval_only:
            if eval_loader is None:
                raise ValueError("--eval-only requires --val-annotations")
            metrics = evaluate(
                model,
                eval_loader,
                device,
                amp_dtype=amp_dtype,
                rank=rank,
                benchmark=args.eval_benchmark,
            )
            if rank == 0:
                logger.info("Evaluation: %s", metrics)
            if wandb_run is not None:
                wandb_run.finish()
            return

        history = []
        best_loss = float("inf")
        best_epoch = -1
        plateau_reference = float("inf")
        epochs_without_significant_improvement = 0
        for epoch in range(start_epoch, args.epochs):
            mean_loss, global_step, mean_terms = train_one_epoch(
                model,
                train_loader,
                optimizer,
                scheduler,
                scaler,
                device,
                epoch,
                global_step,
                amp_dtype,
                gradient_clip_norm=args.gradient_clip_norm,
                log_every=args.log_every,
                max_steps=args.max_steps_per_epoch,
                rank=rank,
                wandb_run=wandb_run,
            )
            metrics = None
            if (
                eval_loader is not None
                and args.eval_every > 0
                and (epoch + 1) % args.eval_every == 0
            ):
                metrics = evaluate(
                    model,
                    eval_loader,
                    device,
                    amp_dtype=amp_dtype,
                    rank=rank,
                    benchmark=args.eval_benchmark,
                )

            should_stop = False
            if rank == 0:
                improved = mean_loss < best_loss
                if improved:
                    best_loss = mean_loss
                    best_epoch = epoch
                if plateau_reference == float("inf"):
                    significant = True
                else:
                    relative_improvement = (
                        plateau_reference - mean_loss
                    ) / max(abs(plateau_reference), 1e-12)
                    significant = (
                        relative_improvement
                        >= args.plateau_min_relative_improvement
                    )
                if significant:
                    plateau_reference = mean_loss
                    epochs_without_significant_improvement = 0
                else:
                    epochs_without_significant_improvement += 1

                checkpoint_path = None
                if args.save_all_checkpoints or improved:
                    checkpoint_path = save_checkpoint(
                        output_dir,
                        epoch,
                        global_step,
                        model,
                        optimizer,
                        scheduler,
                        scaler,
                        args,
                        OmegaConf.to_container(config, resolve=True),
                        keep_last=(0 if args.save_all_checkpoints else 1),
                    )
                history.append({
                    "epoch": epoch,
                    "loss": mean_loss,
                    "loss_terms": mean_terms,
                    "checkpoint": str(checkpoint_path) if checkpoint_path else None,
                })
                should_stop = bool(
                    args.early_stop_training_loss
                    and epoch + 1 >= args.plateau_min_epochs
                    and epochs_without_significant_improvement
                    >= args.plateau_patience
                )
                summary = {
                    "run_name": args.run_name,
                    "epochs_completed": epoch + 1,
                    "best_epoch": best_epoch,
                    "best_loss": best_loss,
                    "plateau_detected": should_stop,
                    "history": history,
                }
                with (output_dir / "training_summary.json").open(
                    "w", encoding="utf-8"
                ) as handle:
                    json.dump(summary, handle, indent=2)
                logger.info(
                    "epoch=%d loss=%.6f loss_terms=%s peak_memory_gib=%.2f metrics=%s checkpoint=%s",
                    epoch,
                    mean_loss,
                    mean_terms,
                    (
                        torch.cuda.max_memory_allocated(device)
                        / (1024 ** 3)
                        if device.type == "cuda" else 0.0
                    ),
                    metrics,
                    checkpoint_path,
                )
                if wandb_run is not None:
                    payload = {
                        "train/epoch_loss": mean_loss,
                        "epoch": epoch,
                    }
                    if metrics is not None:
                        payload.update(
                            {
                                f"eval/{key}": value
                                for key, value in metrics.items()
                            }
                        )
                    wandb_run.log(payload, step=global_step)
            if distributed:
                stop_tensor = torch.tensor(
                    int(should_stop), device=device, dtype=torch.int32
                )
                dist.broadcast(stop_tensor, src=0)
                should_stop = bool(stop_tensor.item())
                _barrier(gpu)
            if should_stop:
                if rank == 0:
                    logger.info(
                        "training-loss plateau after %d epochs; best epoch=%d loss=%.6f",
                        epoch + 1,
                        best_epoch,
                        best_loss,
                    )
                break

        if wandb_run is not None:
            wandb_run.finish()
    finally:
        if distributed and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
