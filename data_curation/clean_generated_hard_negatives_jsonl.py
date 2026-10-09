#!/usr/bin/env python3
"""
Demo (input and output may be the same; replacement is atomic):

  python3 clean_generated_hard_negatives_jsonl.py \
    --source_json all_merged.json \
    --input_jsonl generated_hard_negatives.jsonl \
    --output_jsonl generated_hard_negatives.jsonl \
    --max_difference 0.55

Filters generated hard negatives, aggregates repeated rolling-pass lines, and writes
generator-v2 state that generate_missing_hard_negatives.py can resume directly.
"""

import argparse
import json
import os
import tempfile
from collections import Counter
from difflib import SequenceMatcher

import generate_missing_hard_negatives as generator


def difference_ratio(original, candidate):
    a = generator.normalized(original).split()
    b = generator.normalized(candidate).split()
    return 1.0 - SequenceMatcher(None, a, b, autojunk=False).ratio()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_json", required=True)
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--max_difference", type=float, default=0.55)
    parser.add_argument(
        "--skip_background_check", action="store_true",
        help="Use only edit-distance filtering (not recommended)",
    )
    args = parser.parse_args()
    if not 0 <= args.max_difference <= 1:
        parser.error("max_difference must be between 0 and 1")

    with open(args.source_json, encoding="utf-8") as f:
        source = json.load(f)

    aggregated = {}
    stats = Counter()
    with open(args.input_jsonl, encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                wrapper = json.loads(line)
            except json.JSONDecodeError:
                stats["invalid_jsonl_lines"] += 1
                continue
            if not isinstance(wrapper, dict):
                stats["invalid_wrappers"] += 1
                continue
            for sentence, record in wrapper.items():
                if not isinstance(record, dict):
                    stats["invalid_records"] += 1
                    continue
                aggregated.setdefault(sentence, []).extend(
                    record.get("hard_negatives") or []
                )

    output_dir = os.path.dirname(os.path.abspath(args.output_jsonl)) or "."
    fd, temporary_path = tempfile.mkstemp(
        prefix=".clean_hn_", suffix=".jsonl.tmp", dir=output_dir, text=True
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fout:
            for sentence, candidates in aggregated.items():
                source_record = source.get(sentence)
                if source_record is None:
                    stats["unknown_sentences"] += 1
                    continue
                original_key = generator.normalized(sentence)
                seen = set()
                kept = []
                for candidate in candidates:
                    stats["input_hard_negatives"] += 1
                    if not isinstance(candidate, str):
                        stats["removed_non_string"] += 1
                        continue
                    candidate = candidate.strip()
                    key = generator.normalized(candidate)
                    if not key:
                        stats["removed_empty"] += 1
                        continue
                    if key == original_key:
                        stats["removed_same_as_original"] += 1
                        continue
                    if key in seen:
                        stats["removed_duplicate"] += 1
                        continue
                    if difference_ratio(sentence, candidate) > args.max_difference:
                        stats["removed_over_difference_threshold"] += 1
                        continue
                    if (not args.skip_background_check
                            and not generator.background_preserved(
                                sentence, candidate,
                                source_record.get("verb_phrases") or [],
                            )):
                        stats["removed_background_mismatch"] += 1
                        continue
                    seen.add(key)
                    kept.append(candidate)
                    stats["kept_hard_negatives"] += 1

                if kept:
                    request_id = generator.request_id_for(sentence)
                    wrapper = {
                        sentence: {
                            "hard_negatives": kept,
                            "generator_version": generator.GENERATOR_VERSION,
                            "request_id": request_id,
                        }
                    }
                    fout.write(json.dumps(wrapper, ensure_ascii=False) + "\n")
                    stats["sentences_with_kept_output"] += 1
            fout.flush()
            os.fsync(fout.fileno())
        os.replace(temporary_path, args.output_jsonl)
    except BaseException:
        try:
            os.unlink(temporary_path)
        except FileNotFoundError:
            pass
        raise

    for key in sorted(stats):
        print(f"{key}: {stats[key]}")


if __name__ == "__main__":
    main()
