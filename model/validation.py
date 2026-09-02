"""Distributed-safe evaluation for InstrAct."""

from __future__ import annotations

import torch
import torch.distributed as dist
from tqdm import tqdm

from metrics import grouped_retrieval_metrics


def _distributed():
    return dist.is_available() and dist.is_initialized()


def _unwrap(model):
    return model.module if hasattr(model, "module") else model


def _collect_records(local_records):
    if not _distributed():
        return local_records
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, local_records)
    merged = []
    for records in gathered:
        merged.extend(records)
    # DistributedSampler pads to equal lengths. Keep one result per dataset row.
    return list({record["index"]: record for record in merged}.values())


@torch.no_grad()
def evaluate(model, data_loader, device, amp_dtype=None, rank=0, benchmark="both"):
    """Evaluate verb substitutions and order-swapped negatives."""
    model.eval()
    core_model = _unwrap(model)
    records = []
    progress = tqdm(
        data_loader,
        desc="Evaluation",
        disable=rank != 0,
    )

    for batch in progress:
        video = batch["video"].to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        )
        batch_size = video.shape[0]

        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=amp_dtype is not None,
        ):
            encoded = {}
            for name, key in (
                ("hard", "hard_negative_text"),
                ("order", "order_text"),
            ):
                if key not in batch:
                    continue
                text = batch[key].to(device=device, non_blocking=True)
                video_features, text_features = core_model.encode(
                    video, text.reshape(-1, text.shape[-1])
                )
                encoded[name] = (
                    video_features,
                    text_features.reshape(batch_size, text.shape[1], -1),
                )
        for row in range(batch_size):
            record = {"index": int(batch["index"][row])}
            for name, mask_key in (
                ("hard", "hard_negative_mask"),
                ("order", "order_mask"),
            ):
                if name in encoded:
                    record[f"{name}_video"] = encoded[name][0][row].float().cpu()
                    record[f"{name}_text"] = encoded[name][1][row].float().cpu()
                    record[f"{name}_mask"] = batch[mask_key][row].bool().cpu()
            records.append(record)

    records = _collect_records(records)
    metrics = None
    if rank == 0:
        if not records:
            raise RuntimeError("evaluation dataset produced no records")
        records.sort(key=lambda record: record["index"])
        metrics = {}
        if "hard_video" in records[0]:
            hard_metrics = grouped_retrieval_metrics(
                torch.stack([record["hard_video"] for record in records]),
                torch.stack([record["hard_text"] for record in records]),
                torch.stack([record["hard_mask"] for record in records]),
            )
            metrics.update(HN_R1=hard_metrics["R1"], HN_R5=hard_metrics["R5"], HN_MR=hard_metrics["MR"])
        if "order_video" in records[0]:
            order_metrics = grouped_retrieval_metrics(
                torch.stack([record["order_video"] for record in records]),
                torch.stack([record["order_text"] for record in records]),
                torch.stack([record["order_mask"] for record in records]),
            )
            metrics["Order_R1"] = order_metrics["R1"]

    if _distributed():
        payload = [metrics]
        dist.broadcast_object_list(payload, src=0)
        metrics = payload[0]
    model.train()
    return metrics
