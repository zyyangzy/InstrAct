#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
python nljson_to_dict.py filtered_captions_cleaned.json -o captions.json


nljson_to_dict.py
Input

{"6zPPafYg4Os": {"text": [...], "start": [...], "end": [...]}}
{"6za3ueaq7JU": {"text": [], "start": [], "end": []}}

Output
{
  "6zPPafYg4Os": {...},
  "6za3ueaq7JU": {...}
}
"""

#!/usr/bin/env python3
# nljson_to_dict.py — merge per-line JSON objects into one JSON dict

#!/usr/bin/env python3
# nljson_to_dict.py — merge newline-delimited / streamed JSON into one JSON dict

import argparse, json, sys
from typing import Any, Dict, Iterator, Tuple

def as_list(x): return [] if x is None else (x if isinstance(x, list) else [x])

def to_float_list(arr):
    out=[]
    for v in arr:
        try: out.append(float(v))
        except: out.append(v)
    return out

def normalize(entry: Dict[str, Any], vid: str, strict: bool):
    t = [str(x) for x in as_list(entry.get("text", []))]
    s = as_list(entry.get("start", []))
    e = as_list(entry.get("end", []))
    if not (len(t) == len(s) == len(e)):
        if strict:
            raise ValueError(f"Length mismatch for {vid}: text={len(t)}, start={len(s)}, end={len(e)}")
        m = min(len(t), len(s), len(e))
        t, s, e = t[:m], s[:m], e[:m]
    return {"text": t, "start": to_float_list(s), "end": to_float_list(e)}

def decode_stream(line: str) -> Iterator[Dict[str, Any]]:
    # Parse multiple top-level JSON objects from a single string.
    dec = json.JSONDecoder()
    i, n = 0, len(line)
    while i < n:
        while i < n and line[i].isspace(): i += 1
        if i >= n: break
        obj, j = dec.raw_decode(line, i)
        yield obj
        i = j

def parse_args():
    ap = argparse.ArgumentParser(description="Merge JSONL / JSON stream into a single JSON dict.")
    ap.add_argument("input", nargs="?", default="-", help="Input file (or - for stdin)")
    ap.add_argument("-o","--output", default="-", help="Output file (or - for stdout)")
    ap.add_argument("-s","--strict", action="store_true", help="Fail if text/start/end lengths differ")
    ap.add_argument("-d","--drop-empty", action="store_true", help="Drop entries with all empty lists")
    ap.add_argument("-i","--indent", type=int, default=2, help="Pretty-print indent")
    ap.add_argument("--ignore-errors", action="store_true", help="Skip malformed chunks instead of failing")
    return ap.parse_args()

def main():
    args = parse_args()
    fh = sys.stdin if args.input == "-" else open(args.input, "r", encoding="utf-8")
    merged: Dict[str, Dict[str, Any]] = {}
    try:
        for ln, raw in enumerate(fh, 1):
            line = raw.strip().lstrip("\ufeff")
            if not line: continue
            try:
                objs = list(decode_stream(line))
            except Exception as e:
                if args.ignore_errors:
                    print(f"[warn] line {ln}: {e}", file=sys.stderr)
                    continue
                raise ValueError(f"Line {ln} not valid JSON stream: {e}") from e
            for obj in objs:
                if not isinstance(obj, dict): continue
                for vid, entry in obj.items():
                    if not isinstance(entry, dict): continue
                    try:
                        fixed = normalize(entry, str(vid), args.strict)
                    except Exception as e:
                        if args.ignore_errors:
                            print(f"[warn] vid {vid} on line {ln}: {e}", file=sys.stderr)
                            continue
                        raise
                    if args.drop_empty and not any(len(fixed[k]) for k in ("text","start","end")):
                        continue
                    merged[str(vid)] = fixed  # later overrides earlier
    finally:
        if fh is not sys.stdin: fh.close()
    out = json.dumps(merged, ensure_ascii=False, indent=args.indent)
    if args.output == "-": sys.stdout.write(out + "\n")
    else:
        with open(args.output, "w", encoding="utf-8") as fo:
            fo.write(out + "\n")

if __name__ == "__main__":
    main()
