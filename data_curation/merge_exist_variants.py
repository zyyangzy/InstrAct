#!/usr/bin/env python3
"""Union two sentence-keyed dataset JSON files without overwriting either input."""

import argparse
import json
import os
import tempfile
from collections import Counter


UNION_LIST_FIELDS = ("hard_negatives", "order_swapped_hn", "verb_phrases")


def append_unique(destination, incoming):
    if not isinstance(destination, list):
        destination = []
    seen = set()
    for value in destination:
        try:
            seen.add(json.dumps(value, ensure_ascii=False, sort_keys=True))
        except TypeError:
            pass
    added = 0
    for value in incoming if isinstance(incoming, list) else []:
        key = json.dumps(value, ensure_ascii=False, sort_keys=True)
        if key not in seen:
            destination.append(value)
            seen.add(key)
            added += 1
    return destination, added


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary", required=True)
    parser.add_argument("--secondary", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    input_paths = {os.path.abspath(args.primary), os.path.abspath(args.secondary)}
    output_path = os.path.abspath(args.output)
    if output_path in input_paths:
        parser.error("output must differ from both inputs")
    if os.path.exists(output_path):
        parser.error(f"refusing to overwrite existing output: {output_path}")

    with open(args.primary, encoding="utf-8") as handle:
        merged = json.load(handle)
    with open(args.secondary, encoding="utf-8") as handle:
        secondary = json.load(handle)

    stats = Counter()
    for sentence, incoming_record in secondary.items():
        if sentence not in merged:
            merged[sentence] = incoming_record
            stats["secondary_only_sentences"] += 1
            continue
        stats["shared_sentences"] += 1
        record = merged[sentence]
        for field in UNION_LIST_FIELDS:
            combined, added = append_unique(record.get(field, []), incoming_record.get(field, []))
            if combined or field in record or field in incoming_record:
                record[field] = combined
            stats[f"added_{field}_from_secondary"] += added
        for field, value in incoming_record.items():
            if field in UNION_LIST_FIELDS:
                continue
            if field not in record:
                record[field] = value
                stats["missing_metadata_fields_filled"] += 1
            elif record[field] != value:
                # The primary is the newer/preferred record for non-list metadata.
                stats["metadata_conflicts_kept_primary"] += 1

    stats["primary_only_sentences"] = len(merged) - stats["shared_sentences"] - stats["secondary_only_sentences"]
    stats["final_sentences"] = len(merged)
    stats["final_hard_negatives"] = sum(
        len(record.get("hard_negatives") or []) for record in merged.values()
    )
    stats["final_order_swapped_hn"] = sum(
        len(record.get("order_swapped_hn") or []) for record in merged.values()
    )

    output_dir = os.path.dirname(output_path)
    fd, temporary_path = tempfile.mkstemp(
        prefix=f".{os.path.basename(output_path)}.", suffix=".tmp", dir=output_dir
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(merged, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary_path, os.stat(args.primary).st_mode)
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
