#!/usr/bin/env python3
"""Generate only missing order-swapped hard negatives, with safe resume support.

Example:
  python3 generate_order_hn.py \
    --input_json exist_updated.json \
    --output_jsonl order_swap/missing_order_swapped_hn.jsonl \
    --prompt_file prompt/order_swap_prompt_over4.txt \
    --model_path Qwen/Qwen3-32B \
    --model_url http://127.0.0.1:18000/v1/completions \
    --req_batch_size 16 --parallel_workers 4
    
  python3 generate_order_hn.py \
    --input_json exist.json \
    --output_jsonl missing_order_swapped_hn.jsonl \
    --prompt_file prompt/order_swap_prompt_over4.txt \
    --model_path Qwen/Qwen2.5-72B-Instruct \
    --model_url http://localhost:8000/v1/completions \
    --req_batch_size 16 \
    --parallel_workers 4

The input JSON is never rewritten. One sentence-keyed object is appended per
successful sample. Re-running the same command resumes from valid prior output.
"""

import argparse
import ast
import hashlib
import json
import os
import re
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

import requests

try:
    from tqdm import tqdm
except ImportError:
    class tqdm:
        def __init__(self, total, desc="", unit="item"):
            self.total, self.desc, self.unit, self.count = total, desc, unit, 0
        def __enter__(self):
            print(f"{self.desc}: 0/{self.total} {self.unit}")
            return self
        def __exit__(self, exc_type, exc, traceback):
            print(f"{self.desc}: {self.count}/{self.total} {self.unit}")
        def update(self, amount=1):
            self.count += amount


TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)*|[^\w\s]", re.UNICODE)
NUMBERED_RE = re.compile(r"^\s*(?:\d+\s*[\)\.:-]|[-*])\s*(.+?)\s*$")
REQUEST_ID_RE = re.compile(r"^\s*REQUEST_ID\s*:\s*([a-f0-9]{16})\s*$", re.I)
BAD_AUX_GERUND_RE = re.compile(
    r"\b(?:(?:i|you|we|they|he|she|it)'ll|"
    r"(?:i|you|we|they|he|she|it)\s+(?:will|would|could|should|can|must))"
    r"\s+[a-z]+ing\b",
    re.I,
)
BAD_TO_GERUND_RE = re.compile(r"\b(?:begin|begins|began|start|starts|started)\s+to\s+[a-z]+ing\b", re.I)
DANGLING_END_RE = re.compile(r"\b(?:and|or|to|the|a|an|then|and\s+add)\s*[.!?]*$", re.I)
GENERATOR_VERSION = 1
_THREAD_LOCAL = threading.local()


class ModelUnavailableError(RuntimeError):
    """Raised after the model endpoint remains unavailable across all retries."""


def normalized(text):
    tokens = TOKEN_RE.findall(str(text).strip())
    return " ".join(
        token.lower().replace("’", "'") for token in tokens if re.search(r"\w", token)
    )


def unique_strings(values):
    kept, seen = [], set()
    for value in values or []:
        if not isinstance(value, str):
            continue
        value = value.strip().strip('"').strip()
        key = normalized(value)
        if key and key not in seen:
            seen.add(key)
            kept.append(value)
    return kept


def load_prior_generations(path):
    generated = defaultdict(list)
    if not os.path.exists(path):
        return generated
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                wrapper = json.loads(line)
            except json.JSONDecodeError:
                print(f"Warning: ignoring invalid JSONL line {line_number}")
                continue
            if not isinstance(wrapper, dict):
                continue
            for sentence, record in wrapper.items():
                if isinstance(record, dict):
                    generated[sentence].extend(record.get("order_swapped_hn") or [])
    for sentence in list(generated):
        generated[sentence] = unique_strings(generated[sentence])
    return generated


def request_id_for(sentence):
    return hashlib.sha256(sentence.encode("utf-8")).hexdigest()[:16]


def count_limits(verb_phrase_count):
    if verb_phrase_count == 2:
        return 1, 1
    if verb_phrase_count >= 3:
        return 3, 3
    return 0, 0


def build_prompt(base_prompt, job):
    caption = job["sentence"]
    verb_phrases = job["verb_phrases"]
    needed = job["needed"]
    existing = json.dumps(job["existing"], ensure_ascii=False)
    request_id = job["request_id"]
    input_text = (
        f'caption: {json.dumps(caption, ensure_ascii=False)} '
        f'"verb_phrases": {json.dumps(verb_phrases, ensure_ascii=False)}'
    )
    # Keep the requested prompt as the task/example source, then explicitly
    # override its old 4/5-phrase and 12-output constraints for this new run.
    return f"""/no_think

{base_prompt}

---

IMPORTANT: For the NEW input below, ignore the example's constraints that the
input has 4 or 5 verb phrases and that 12 outputs are required. The NEW input may
have 2 or more verb phrases.

Generate EXACTLY {needed} new, distinct order-swapped caption(s). Reorder the
listed verb phrases chronologically while preserving the caption's other content.
Do not output the original caption or anything in the existing list. Use only
words from the CURRENT input caption; never copy words from the examples.

Existing order-swapped captions (DO NOT repeat):
{existing}

Input: {input_text}
Request ID: {request_id}

The first output line MUST be exactly:
REQUEST_ID: {request_id}
Then output only {needed} numbered captions. Do not include analysis, headings,
examples, or a second input. End immediately with <END_OF_OUTPUT>.

Outputs:"""


def parse_generation(raw_text, expected_request_id):
    text = (raw_text or "").strip()
    if not text:
        return [], "empty"
    # Qwen3 may emit a thinking block before the requested answer.
    text = re.sub(r"^\s*<think>.*?</think>\s*", "", text, flags=re.S | re.I)
    lines = text.splitlines()
    marker_index = next(
        (i for i, line in enumerate(lines) if REQUEST_ID_RE.match(line)), None
    )
    if marker_index is None:
        return [], "request_id_missing"
    match = REQUEST_ID_RE.match(lines[marker_index])
    if match.group(1).lower() != expected_request_id.lower():
        return [], "request_id_mismatch"
    text = "\n".join(lines[marker_index + 1:]).split("<END_OF_OUTPUT>", 1)[0].strip()
    repeated = re.search(r"\b(?:REQUEST_ID|Input)\s*:", text, flags=re.I)
    if repeated:
        text = text[:repeated.start()].strip()
    if text.startswith("["):
        end = text.rfind("]")
        if end >= 0:
            try:
                parsed = ast.literal_eval(text[:end + 1])
                if isinstance(parsed, list):
                    return unique_strings(parsed), "ok"
            except (ValueError, SyntaxError):
                pass
    outputs = []
    for line in text.splitlines():
        match = NUMBERED_RE.match(line)
        if match:
            outputs.append(match.group(1).strip())
    parsed = unique_strings(outputs)
    return parsed, "ok" if parsed else "parse_failed"


def words_are_from_caption(sentence, candidate):
    allowed = Counter(normalized(sentence).split())
    produced = Counter(normalized(candidate).split())
    # Reordering may repeat a connector/pronoun, but must not introduce vocabulary.
    return all(word in allowed for word in produced)


def structurally_valid(candidate):
    """Reject a few high-confidence malformed patterns; avoid subjective scoring."""
    text = candidate.strip()
    if BAD_AUX_GERUND_RE.search(text) or BAD_TO_GERUND_RE.search(text):
        return False
    if DANGLING_END_RE.search(text):
        return False
    if re.match(r"^\s*until\b", text, flags=re.I):
        return False
    return True


def get_session():
    session = getattr(_THREAD_LOCAL, "session", None)
    if session is None:
        session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=1, pool_maxsize=1)
        session.mount("http://", adapter)
        _THREAD_LOCAL.session = session
    return session


def max_tokens_for(jobs, cap):
    estimate = max(256, max(64 + job["needed"] * (len(job["sentence"].split()) * 2 + 20)
                            for job in jobs))
    return min(cap, estimate)


def post_batch(url, model_path, jobs, base_prompt, timeout, token_cap, retries, verbose):
    payload = {
        "model": model_path,
        "prompt": [build_prompt(base_prompt, job) for job in jobs],
        "max_tokens": max_tokens_for(jobs, token_cap),
        "temperature": 0.5,
        "top_p": 0.9,
        "stop": ["<END_OF_OUTPUT>"],
    }
    last_error = None
    for attempt in range(retries + 1):
        try:
            response = get_session().post(url, json=payload, timeout=timeout)
            response.raise_for_status()
            choices = response.json().get("choices", [])
            mapped = [""] * len(jobs)
            seen = set()
            for choice in choices:
                index = choice.get("index")
                if not isinstance(index, int) or not 0 <= index < len(jobs) or index in seen:
                    raise RuntimeError(f"invalid or duplicate choice.index: {index!r}")
                seen.add(index)
                mapped[index] = choice.get("text", "")
            if len(seen) != len(jobs):
                raise RuntimeError(f"expected {len(jobs)} choices, received {len(seen)}")
            return mapped
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(min(2 ** attempt, 8))
    if verbose:
        print(f"Request failed at {url}: {last_error}", flush=True)
    raise ModelUnavailableError(
        f"model endpoint unavailable after {retries + 1} attempts: "
        f"{url}: {last_error}"
    )


def make_jobs(data, prior, max_samples, reverse):
    groups = defaultdict(list)
    items = reversed(data.items()) if reverse else data.items()
    selected = 0
    for sentence, record in items:
        # Only fill absent/empty input values. Never alter samples already populated.
        if unique_strings(record.get("order_swapped_hn") or []):
            continue
        verb_phrases = record.get("verb_phrases") or []
        minimum, maximum = count_limits(len(verb_phrases))
        if maximum == 0:
            continue
        existing = unique_strings(prior.get(sentence, []))[:maximum]
        if len(existing) >= minimum:
            continue
        needed = maximum - len(existing)
        groups[needed].append({
            "sentence": sentence,
            "verb_phrases": verb_phrases,
            "existing": existing,
            "needed": needed,
            "minimum": minimum,
            "maximum": maximum,
            "request_id": request_id_for(sentence),
        })
        selected += 1
        if max_samples is not None and selected >= max_samples:
            break
    return groups


def main():
    parser = argparse.ArgumentParser(description="Fill missing order-swapped hard negatives.")
    parser.add_argument("--input_json", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--prompt_file", required=True)
    parser.add_argument("--model_path", default="Qwen/Qwen3-32B")
    parser.add_argument("--model_url", default="http://127.0.0.1:18000/v1/completions")
    parser.add_argument("--req_batch_size", type=int, default=16)
    parser.add_argument("--parallel_workers", type=int, default=4)
    parser.add_argument("--request_timeout", type=int, default=300)
    parser.add_argument("--request_retries", type=int, default=2)
    parser.add_argument("--max_tokens_cap", type=int, default=1024)
    parser.add_argument("--flush_every", type=int, default=100)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if (args.req_batch_size <= 0 or args.parallel_workers <= 0 or args.flush_every <= 0
            or args.request_retries < 0):
        parser.error("batch/worker/flush must be positive and retries nonnegative")

    with open(args.prompt_file, encoding="utf-8") as handle:
        base_prompt = handle.read().strip()
    with open(args.input_json, encoding="utf-8") as handle:
        data = json.load(handle)
    prior = load_prior_generations(args.output_jsonl)
    groups = make_jobs(data, prior, args.max_samples, args.reverse)
    total = sum(map(len, groups.values()))
    print(f"Eligible samples this pass: {total}", flush=True)
    for needed in sorted(groups, reverse=True):
        print(f"  requesting {needed}: {len(groups[needed])}", flush=True)
    if total == 0:
        print("Nothing to generate.")
        return
    urls = [url.strip() for url in args.model_url.split(",") if url.strip()]
    if not urls:
        parser.error("No valid model_url")

    stats = Counter()
    written_since_flush = 0
    os.makedirs(os.path.dirname(os.path.abspath(args.output_jsonl)), exist_ok=True)
    with open(args.output_jsonl, "a", encoding="utf-8") as output:
        with tqdm(total=total, desc="Generating", unit="sample") as progress:
            for needed in sorted(groups, reverse=True):
                jobs = groups[needed]
                batch_count = (len(jobs) + args.req_batch_size - 1) // args.req_batch_size
                with ThreadPoolExecutor(max_workers=args.parallel_workers) as pool:
                    pending, next_batch = {}, 0

                    def submit_one(batch_index):
                        start = batch_index * args.req_batch_size
                        batch = jobs[start:start + args.req_batch_size]
                        future = pool.submit(
                            post_batch, urls[batch_index % len(urls)], args.model_path,
                            batch, base_prompt, args.request_timeout,
                            args.max_tokens_cap, args.request_retries, args.verbose,
                        )
                        pending[future] = batch

                    initial = min(batch_count, args.parallel_workers * 2)
                    for next_batch in range(initial):
                        submit_one(next_batch)
                    next_batch = initial
                    while pending:
                        completed, _ = wait(pending, return_when=FIRST_COMPLETED)
                        for future in completed:
                            batch = pending.pop(future)
                            try:
                                raw_outputs = future.result()
                            except ModelUnavailableError as exc:
                                # Nothing from this failed batch has been written.
                                # Flush all earlier completed batches, cancel work
                                # not yet started, then end the screen process so a
                                # dead endpoint cannot create millions of empty runs.
                                output.flush()
                                for unfinished in pending:
                                    unfinished.cancel()
                                print(f"FATAL: {exc}", flush=True)
                                print(
                                    "Stopping without recording the failed batch. "
                                    "Re-run the same command to resume.",
                                    flush=True,
                                )
                                raise SystemExit(2)
                            for job, raw in zip(batch, raw_outputs):
                                forbidden = {normalized(job["sentence"])}
                                forbidden.update(normalized(x) for x in job["existing"])
                                accepted = []
                                candidates, status = parse_generation(raw, job["request_id"])
                                if status != "ok":
                                    stats["rejected_" + status] += 1
                                for candidate in candidates:
                                    key = normalized(candidate)
                                    if not key or key in forbidden:
                                        stats["rejected_duplicate_or_original"] += 1
                                        continue
                                    if not words_are_from_caption(job["sentence"], candidate):
                                        stats["rejected_new_vocabulary"] += 1
                                        continue
                                    if not structurally_valid(candidate):
                                        stats["rejected_malformed_structure"] += 1
                                        continue
                                    forbidden.add(key)
                                    accepted.append(candidate)
                                    if len(accepted) >= job["needed"]:
                                        break
                                if accepted:
                                    result = {job["sentence"]: {
                                        "order_swapped_hn": accepted,
                                        "generator_version": GENERATOR_VERSION,
                                        "request_id": job["request_id"],
                                    }}
                                    output.write(json.dumps(result, ensure_ascii=False) + "\n")
                                    written_since_flush += 1
                                    stats["samples_with_new_output"] += 1
                                    stats["new_order_swapped_hn"] += len(accepted)
                                    if written_since_flush >= args.flush_every:
                                        output.flush()
                                        written_since_flush = 0
                                else:
                                    stats["samples_without_new_output"] += 1
                                if len(job["existing"]) + len(accepted) < job["minimum"]:
                                    stats["samples_still_incomplete"] += 1
                                progress.update(1)
                            if next_batch < batch_count:
                                submit_one(next_batch)
                                next_batch += 1
            output.flush()

    print("Pass complete. Re-run the same command to fill remaining samples.")
    for key in sorted(stats):
        print(f"{key}: {stats[key]}")


if __name__ == "__main__":
    main()
