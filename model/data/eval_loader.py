"""Evaluation dataset for action hard-negative retrieval."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from data.train_loader import HT100MDataLoader
from models.third_party.internvideo_compat.simple_tokenizer import SimpleTokenizer


class HT100MEvalLoader(Dataset):
    """Return fixed-size verb-HN and order-swapped candidate groups."""

    def __init__(
        self,
        video_root,
        annotation_json,
        min_time=3.2,
        fps=10,
        num_frames=16,
        size=224,
        crop_only=False,
        max_words=32,
        max_negatives=9,
        max_order_negatives=2,
        uniform_window=5.0,
        benchmark="both",
    ):
        self.video_root = Path(video_root)
        self.min_time = float(min_time)
        self.fps = float(fps)
        self.num_frames = int(num_frames)
        self.size = int(size)
        self.crop_only = bool(crop_only)
        self.max_words = int(max_words)
        self.max_negatives = int(max_negatives)
        self.max_order_negatives = int(max_order_negatives)
        self.uniform_window = float(uniform_window)
        if benchmark not in {"both", "hard_negative", "order"}:
            raise ValueError(f"unsupported evaluation benchmark: {benchmark}")
        self.benchmark = benchmark
        self.tokenizer = SimpleTokenizer()

        with open(annotation_json, "r", encoding="utf-8") as handle:
            annotations = json.load(handle)
        if isinstance(annotations, dict):
            annotation_rows = (
                {"caption": caption, **info}
                for caption, info in annotations.items()
                if isinstance(info, dict)
            )
        elif isinstance(annotations, list):
            annotation_rows = annotations
        else:
            raise ValueError("annotation_json must contain an object or a list")

        self.samples = []
        for info in annotation_rows:
            if not isinstance(info, dict):
                continue
            caption = info.get("caption")
            if not isinstance(caption, str) or not caption.strip():
                continue
            video_id = info.get("video_id")
            if not isinstance(video_id, str) or not video_id:
                continue
            video_path = self.video_root / f"{video_id}.mp4"
            if not video_path.is_file():
                continue
            try:
                start = float(info.get("start", 0.0))
                end = float(info.get("end", 0.0))
            except (TypeError, ValueError):
                continue
            if end - start < self.min_time:
                difference = self.min_time - (end - start)
                start = max(0.0, start - difference / 2.0)
                end = start + self.min_time

            self.samples.append(
                {
                    "caption": caption,
                    "video_id": video_id,
                    "video_path": str(video_path),
                    "start": start,
                    "end": end,
                    "hard_negatives": HT100MDataLoader._clean_strings(
                        info.get("hard_negatives") or []
                    ),
                    "order_swapped_hn": HT100MDataLoader._clean_strings(
                        info.get("order_swapped_hn") or []
                    ),
                }
            )

        print(
            f"[HT100MEvalLoader] Loaded {len(self.samples)} "
            "caption-video pairs."
        )

    def __len__(self):
        return len(self.samples)

    # Reuse the release training loader's tested video and text processing.
    _decode_video = HT100MDataLoader._decode_video
    tokenize = HT100MDataLoader.tokenize

    def _candidate_group(self, caption, negatives, maximum):
        negatives = negatives[:maximum]
        text = self.tokenize([caption, *negatives])
        valid = torch.zeros(1 + maximum, dtype=torch.bool)
        valid[: 1 + len(negatives)] = True
        if text.shape[0] < 1 + maximum:
            text = torch.cat(
                (
                    text,
                    torch.zeros(
                        (1 + maximum - text.shape[0], self.max_words),
                        dtype=torch.long,
                    ),
                )
            )
        return text, valid

    def __getitem__(self, index):
        sample = self.samples[index]
        video = self._decode_video(
            sample["video_path"],
            sample["start"],
            sample["end"],
        )
        result = {
            "index": index,
            "video": video,
        }
        if self.benchmark in {"both", "hard_negative"}:
            text, mask = self._candidate_group(
                sample["caption"], sample["hard_negatives"], self.max_negatives
            )
            result["hard_negative_text"] = text
            result["hard_negative_mask"] = mask
        if self.benchmark in {"both", "order"}:
            text, mask = self._candidate_group(
                sample["caption"],
                sample["order_swapped_hn"],
                self.max_order_negatives,
            )
            result["order_text"] = text
            result["order_mask"] = mask
        return result
