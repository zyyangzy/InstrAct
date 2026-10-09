#!/usr/bin/env python3
"""Merge the final pending generated fields into a new copy of exist.json."""

import argparse
import json
import os
import tempfile
from collections import Counter

from merge_order_swapped_hn import iter_concatenated_json


def collect(path, field):
    values = {}
    duplicates = 0
    for group in iter_concatenated_json(path):
        if not isinstance(group, dict):
            continue
        for sentence, record in group.items():
            if not isinstance(record, dict):
                continue
            destination = values.setdefault(sentence, [])
            seen = set(destination)
            for candidate in record.get(field) or []:
                if not isinstance(candidate, str):
                    continue
                if candidate in seen:
                    duplicates += 1
                else:
                    destination.append(candidate)
                    seen.add(candidate)
    return values, duplicates


def merge_field(data, source, field, stats):
    for sentence, candidates in source.items():
        if sentence not in data:
            stats[f"{field}_missing_sentences"] += 1
            stats[f"{field}_skipped_for_missing_sentences"] += len(candidates)
            continue
        destination = data[sentence].setdefault(field, [])
        seen = set(destination)
        for candidate in candidates:
            if candidate in seen:
                stats[f"{field}_already_present"] += 1
            else:
                destination.append(candidate)
                seen.add(candidate)
                stats[f"{field}_added"] += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="exist.json")
    parser.add_argument("--hard-source", default="generated_hard_negatives.jsonl")
    parser.add_argument("--order-source", default="order_swap/vp=7_augmented.json")
    parser.add_argument("--output", default="exist_enriched.next.json")
    args = parser.parse_args()
    if os.path.abspath(args.input) == os.path.abspath(args.output):
        parser.error("output must differ from input")
    if os.path.exists(args.output):
        parser.error(f"refusing to overwrite existing output: {args.output}")

    hard_values, hard_source_duplicates = collect(args.hard_source, "hard_negatives")
    order_values, order_source_duplicates = collect(args.order_source, "order_swapped_hn")
    with open(args.input, encoding="utf-8") as handle:
        data = json.load(handle)

    stats = Counter()
    stats["hard_source_duplicates"] = hard_source_duplicates
    stats["order_source_duplicates"] = order_source_duplicates
    stats["input_sentences"] = len(data)
    merge_field(data, hard_values, "hard_negatives", stats)
    merge_field(data, order_values, "order_swapped_hn", stats)
    stats["final_hard_negatives"] = sum(
        len(record.get("hard_negatives") or []) for record in data.values()
    )
    stats["final_order_swapped_hn"] = sum(
        len(record.get("order_swapped_hn") or []) for record in data.values()
    )

    output_path = os.path.abspath(args.output)
    fd, temporary_path = tempfile.mkstemp(
        prefix=f".{os.path.basename(output_path)}.",
        suffix=".tmp",
        dir=os.path.dirname(output_path),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary_path, os.stat(args.input).st_mode)
        os.replace(temporary_path, output_path)
    except BaseException:
        try:
            os.unlink(temporary_path)
        except FileNotFoundError:
            pass
        raise

    for key in sorted(stats):
        print(f"{key}={stats[key]}")
    print(f"output={output_path}")


if __name__ == "__main__":
    main()
