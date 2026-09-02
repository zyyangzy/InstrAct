#!/usr/bin/env python3
"""Standard evaluation for the three InstrAct Bench tasks.

This evaluator computes and reports benchmark metrics in memory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from data.eval_loader import HT100MEvalLoader
from data.train_loader import HT100MDataLoader
from metrics import grouped_retrieval_metrics
from models import build_instract
from models.third_party.internvideo_compat.simple_tokenizer import SimpleTokenizer
from utils.checkpoint import load_training_checkpoint


ROOT = Path(__file__).resolve().parent
DEFAULT_ANNOTATIONS = {
    "semantic": ROOT / "benchmarks/InstrAct-Semantic/annotations.json",
    "logic": ROOT / "benchmarks/InstrAct-Logic/annotations.json",
    "dynamics": ROOT / "benchmarks/InstrAct-Dynamics",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate InstrAct Bench")
    parser.add_argument(
        "--benchmark",
        choices=("semantic", "logic", "dynamics", "all"),
        default="all",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "models/config.yaml"
    )
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--semantic-annotations", type=Path, default=DEFAULT_ANNOTATIONS["semantic"])
    parser.add_argument("--logic-annotations", type=Path, default=DEFAULT_ANNOTATIONS["logic"])
    parser.add_argument("--dynamics-annotations", type=Path, default=DEFAULT_ANNOTATIONS["dynamics"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", choices=("off", "fp16", "bf16"), default="fp16")
    parser.add_argument("--output", type=Path, help="optional JSON metrics file")
    parser.add_argument("--max-samples", type=int, help="smoke-test limit per task/pool")
    return parser.parse_args()


def model_args():
    return SimpleNamespace(
        use_instract=True,
        use_hard_negatives=False,
        use_action_perceiver=False,
        use_dtw=False,
        use_mam=False,
        freeze_backbone=False,
    )


def load_model(cli, config, device):
    model = build_instract(config, model_args())
    completed_epochs, _ = load_training_checkpoint(cli.checkpoint, model)
    print(f"Loaded {cli.checkpoint} (completed epochs: {completed_epochs})")
    return model.to(device).eval()


def amp_dtype(name, device):
    if name == "off":
        return None
    if name == "fp16":
        if device.type != "cuda":
            raise ValueError("--amp fp16 requires CUDA; use --amp off on CPU")
        return torch.float16
    return torch.bfloat16


@torch.inference_mode()
def evaluate_mcq(cli, config, model, device, dtype, task):
    is_semantic = task == "semantic"
    dataset = HT100MEvalLoader(
        video_root=cli.video_root,
        annotation_json=getattr(cli, f"{task}_annotations"),
        min_time=config.data.min_time,
        fps=config.data.fps,
        num_frames=config.data.num_frames,
        size=config.backbone.input_resolution,
        crop_only=config.data.crop_only,
        max_words=config.data.max_text_length,
        max_negatives=9,
        max_order_negatives=2,
        uniform_window=config.data.uniform_window,
        benchmark="hard_negative" if is_semantic else "order",
    )
    if cli.max_samples:
        dataset = torch.utils.data.Subset(dataset, range(min(cli.max_samples, len(dataset))))
    if not len(dataset):
        raise RuntimeError(f"{task}: no videos found under {cli.video_root}")
    loader = DataLoader(dataset, batch_size=cli.batch_size, shuffle=False, num_workers=cli.workers)
    video_rows, text_rows, mask_rows = [], [], []
    prefix = "hard_negative" if is_semantic else "order"
    for batch in tqdm(loader, desc=f"InstrAct-{task.title()}"):
        video = batch["video"].to(device)
        text = batch[f"{prefix}_text"].to(device)
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype is not None):
            video_features, text_features = model.encode(
                video, text.flatten(0, 1)
            )
        video_rows.append(video_features.float().cpu())
        text_rows.append(text_features.reshape(video.shape[0], text.shape[1], -1).float().cpu())
        mask_rows.append(batch[f"{prefix}_mask"].bool())
    metrics = grouped_retrieval_metrics(
        torch.cat(video_rows), torch.cat(text_rows), torch.cat(mask_rows)
    )
    if is_semantic:
        return {"samples": len(dataset), "R@1": 100 * metrics["R1"], "R@5": 100 * metrics["R5"], "MR": metrics["MR"]}
    return {"samples": len(dataset), "ACC": 100 * metrics["R1"]}


class DynamicsDataset(Dataset):
    _decode_video = HT100MDataLoader._decode_video
    tokenize = HT100MDataLoader.tokenize

    def __init__(self, annotation_path, video_root, config, limit=None):
        rows = json.loads(annotation_path.read_text(encoding="utf-8"))
        if isinstance(rows, dict):
            rows = [{"caption": caption, **info} for caption, info in rows.items()]
        self.samples = []
        for row in rows:
            path = video_root / f"{row['video_id']}.mp4"
            if path.is_file():
                self.samples.append((path, row["caption"], float(row["start"]), float(row["end"])))
        if limit:
            self.samples = self.samples[:limit]
        self.num_frames = int(config.data.num_frames)
        self.fps = float(config.data.fps)
        self.size = int(config.backbone.input_resolution)
        self.crop_only = bool(config.data.crop_only)
        self.max_words = int(config.data.max_text_length)
        self.uniform_window = float(config.data.uniform_window)
        self.tokenizer = SimpleTokenizer()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, caption, start, end = self.samples[index]
        return self._decode_video(str(path), start, end), self.tokenize(caption)[0]


def recalls(ranks):
    return {f"R@{k}": 100 * ranks.lt(k).float().mean().item() for k in (1, 5, 10)}


@torch.inference_mode()
def evaluate_dynamics(cli, config, model, device, dtype):
    totals = {direction: {k: 0.0 for k in (1, 5, 10)} for direction in ("T2V", "V2T")}
    total_samples = 0
    pool_count = 0
    paths = sorted(cli.dynamics_annotations.rglob("*.json"))
    for path in tqdm(paths, desc="InstrAct-Dynamics pools"):
        dataset = DynamicsDataset(path, cli.video_root, config, cli.max_samples)
        if not len(dataset):
            continue
        loader = DataLoader(dataset, batch_size=cli.batch_size, shuffle=False, num_workers=cli.workers)
        videos, texts = [], []
        for video, text in loader:
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype is not None):
                video_features, text_features = model.encode(video.to(device), text.to(device))
            videos.append(F.normalize(video_features.float().cpu(), dim=-1))
            texts.append(F.normalize(text_features.float().cpu(), dim=-1))
        video_features, text_features = torch.cat(videos), torch.cat(texts)
        similarity = text_features @ video_features.T
        target = torch.arange(len(dataset))[:, None]
        t2v_ranks = similarity.argsort(1, descending=True).eq(target).float().argmax(1)
        v2t_ranks = similarity.T.argsort(1, descending=True).eq(target).float().argmax(1)
        for direction, ranks in (("T2V", t2v_ranks), ("V2T", v2t_ranks)):
            for k in (1, 5, 10):
                totals[direction][k] += ranks.lt(k).sum().item()
        total_samples += len(dataset)
        pool_count += 1
    if not total_samples:
        raise RuntimeError(f"dynamics: no videos found under {cli.video_root}")
    result = {"samples": total_samples, "pools": pool_count}
    for k in (1, 5, 10):
        t2v = 100 * totals["T2V"][k] / total_samples
        v2t = 100 * totals["V2T"][k] / total_samples
        result[f"R@{k}"] = {"T2V": t2v, "V2T": v2t, "Mean": (t2v + v2t) / 2}
    return result


def main():
    cli = parse_args()
    if cli.batch_size <= 0 or cli.workers < 0:
        raise ValueError("--batch-size must be positive and --workers non-negative")
    device = torch.device(cli.device if cli.device != "cuda" or torch.cuda.is_available() else "cpu")
    dtype = amp_dtype(cli.amp, device)
    config = OmegaConf.load(cli.config)
    model = load_model(cli, config, device)
    tasks = ("semantic", "logic", "dynamics") if cli.benchmark == "all" else (cli.benchmark,)
    results = {}
    for task in tasks:
        results[f"InstrAct-{task.title()}"] = (
            evaluate_dynamics(cli, config, model, device, dtype)
            if task == "dynamics"
            else evaluate_mcq(cli, config, model, device, dtype, task)
        )
    rendered = json.dumps(results, indent=2)
    print(rendered)
    if cli.output:
        cli.output.parent.mkdir(parents=True, exist_ok=True)
        cli.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
