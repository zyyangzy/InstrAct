#!/usr/bin/env python3
"""
Run demo (safe to stop and run repeatedly):

  python3 generate_missing_hard_negatives.py \
    --input_json exist.json \
    --output_jsonl generated_hard_negatives.jsonl \
    --prompt_file prompt/hard_negative_prompt.txt \
    --model_path Qwen/Qwen2.5-72B-Instruct \
    --model_url http://localhost:8000/v1/completions \
    --req_batch_size 16 \
    --parallel_workers 4

Optional small test (only sends at most 100 eligible samples):

  python3 generate_missing_hard_negatives.py \
    --input_json all_merged.json \
    --output_jsonl generated_hard_negatives_v2_test.jsonl \
    --prompt_file prompt/hard_negative_prompt.txt \
    --model_path Qwen/Qwen2.5-72B-Instruct \
    --model_url http://localhost:8000/v1/completions \
    --max_samples 100 --verbose

This script ONLY generates missing hard negatives. It does not rewrite the input
JSON and does not run linguistic post-processing. Results are appended as JSONL,
one sentence-keyed object per successful sample. Re-running the same command reads
all earlier JSONL lines, recalculates each sample's remaining deficit, and continues
until every eligible sample has TARGET_COUNT unique hard negatives.

Legacy output must first be upgraded with clean_generated_hard_negatives_jsonl.py;
unversioned legacy lines are intentionally ignored to prevent poisoned resume state.
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
    class tqdm:  # Minimal fallback so tqdm is optional.
        def __init__(self, total, desc="", unit="item"):
            self.total, self.desc, self.unit, self.count = total, desc, unit, 0

        def __enter__(self):
            print(f"{self.desc}: 0/{self.total} {self.unit}")
            return self

        def __exit__(self, exc_type, exc, traceback):
            print(f"{self.desc}: {self.count}/{self.total} {self.unit}")

        def update(self, amount=1):
            self.count += amount
            if self.count == self.total or self.count % 1000 == 0:
                print(f"{self.desc}: {self.count}/{self.total} {self.unit}")


TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)*|[^\w\s]", re.UNICODE)
NUMBERED_RE = re.compile(r"^\s*(?:\d+\s*[\)\.:-]|[-*])\s*(.+?)\s*$")
REQUEST_ID_RE = re.compile(r"^\s*REQUEST_ID\s*:\s*([a-f0-9]{16})\s*$", re.I)
GENERATOR_VERSION = 2
_THREAD_LOCAL = threading.local()
ACTION_PARTICLES = {
    "about", "across", "apart", "around", "away", "back", "down", "in",
    "into", "off", "on", "onto", "out", "over", "through", "together", "up",
}


def normalized(text):
    """Comparison key that ignores case, whitespace, and superficial punctuation."""
    tokens = TOKEN_RE.findall(str(text).strip())
    words = [t.lower().replace("’", "'") for t in tokens if re.search(r"\w", t)]
    return " ".join(words)


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


def simple_verb_stem(word):
    word = word.lower().replace("’", "'")
    if len(word) > 5 and word.endswith("ing"):
        stem = word[:-3]
        if len(stem) >= 2 and stem[-1] == stem[-2]:
            stem = stem[:-1]
        return stem
    if len(word) > 4 and word.endswith("ied"):
        return word[:-3] + "y"
    if len(word) > 4 and word.endswith("ed"):
        stem = word[:-2]
        if len(stem) >= 2 and stem[-1] == stem[-2]:
            stem = stem[:-1]
        return stem
    return word


def load_prior_generations(path, accept_legacy=False):
    """Aggregate every prior append, including multiple rolling passes per sentence."""
    generated = defaultdict(list)
    if not os.path.exists(path):
        return generated
    with open(path, encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                wrapper = json.loads(line)
            except json.JSONDecodeError:
                # A killed process can leave only its final line incomplete.
                print(f"Warning: ignoring invalid JSONL line {line_number}")
                continue
            if not isinstance(wrapper, dict):
                continue
            for sentence, record in wrapper.items():
                if isinstance(record, dict):
                    if (record.get("generator_version") != GENERATOR_VERSION
                            and not accept_legacy):
                        continue
                    generated[sentence].extend(record.get("hard_negatives") or [])
    for sentence in list(generated):
        generated[sentence] = unique_strings(generated[sentence])
    return generated


def request_id_for(sentence):
    return hashlib.sha256(sentence.encode("utf-8")).hexdigest()[:16]


def build_prompt(base_prompt, sentence, existing, needed, request_id):
    forbidden = json.dumps(existing, ensure_ascii=False)
    return f"""{base_prompt}

---

Now process this NEW input caption.

Generate EXACTLY {needed} new hard-negative sentence(s), numbered from 1 to {needed}.
Every output MUST be different from the original and from every sentence in the
existing list below. Do not paraphrase, repeat, or make only punctuation/case changes
to an existing sentence. Only change action verbs or their verb phrases; preserve
the original nouns, ingredients, tools, quantities, locations, and background.

Existing hard negatives (DO NOT output any of these):
{forbidden}

Input: {sentence}
Request ID: {request_id}

The first output line MUST be exactly:
REQUEST_ID: {request_id}
Then output only the {needed} numbered sentences. Do not output examples, another
Input section, explanations, headings, or a second REQUEST_ID.
Immediately after the final numbered sentence, output <END_OF_OUTPUT>.

Outputs:"""


def parse_generation(raw_text, expected_request_id):
    text = (raw_text or "").strip()
    if not text:
        return [], "empty"

    lines = text.splitlines()
    first_nonempty = next((line for line in lines if line.strip()), "")
    match = REQUEST_ID_RE.match(first_nonempty)
    if not match or match.group(1).lower() != expected_request_id.lower():
        return [], "request_id_mismatch"
    text = "\n".join(lines[lines.index(first_nonempty) + 1:]).strip()
    # Qwen may start a second copy of the answer block, occasionally on the same
    # line as the last candidate. Never let that marker leak into a candidate.
    repeated_marker = re.search(r"\bREQUEST_ID\s*:", text, flags=re.I)
    if repeated_marker:
        text = text[:repeated_marker.start()].strip()

    # Accept a model-produced JSON/Python list as a fallback.
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
            candidate = match.group(1).strip()
            if re.search(r"\b(?:REQUEST_ID|Input|Outputs?)\s*:", candidate, re.I):
                continue
            outputs.append(candidate)
    parsed = unique_strings(outputs)
    return parsed, "ok" if parsed else "parse_failed"


def background_preserved(sentence, candidate, verb_phrases):
    """Reject obvious cross-sample/example leakage before it reaches JSONL.

    Words named in verb_phrases are allowed to change. Nearly all other original
    words must remain, which is consistent with the hard-negative task definition.
    """
    original = normalized(sentence).split()
    generated = Counter(normalized(candidate).split())
    action_stems = set()
    action_particles = set()
    for phrase in verb_phrases or []:
        words = normalized(phrase).split()
        if words:
            action_stems.add(simple_verb_stem(words[0]))
            action_particles.update(word for word in words[1:] if word in ACTION_PARTICLES)
    background = [
        word for word in original
        if simple_verb_stem(word) not in action_stems and word not in action_particles
    ]
    if not background:
        return True
    needed = Counter(background)
    matched = sum(min(count, generated[word]) for word, count in needed.items())
    ratio = matched / len(background)
    threshold = 1.0 if len(background) <= 4 else 0.8
    return ratio >= threshold


def max_tokens_for(sentence, needed, cap):
    # Long captions need proportionally more output tokens. Group-specific `needed`
    # keeps requests for deficits of 1 or 2 much cheaper than requests for 10.
    input_words = max(8, len(sentence.split()))
    estimate = 48 + needed * (input_words * 2 + 16)
    return min(cap, max(128, estimate))


def get_session():
    session = getattr(_THREAD_LOCAL, "session", None)
    if session is None:
        session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=1, pool_maxsize=1, max_retries=0
        )
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        _THREAD_LOCAL.session = session
    return session


def post_batch(url, model_path, jobs, base_prompt, timeout, token_cap, retries, verbose):
    prompts = [build_prompt(base_prompt, j["sentence"], j["existing"], j["needed"],
                            j["request_id"])
               for j in jobs]
    needed = jobs[0]["needed"]
    payload = {
        "model": model_path,
        "prompt": prompts,
        "max_tokens": max(max_tokens_for(j["sentence"], needed, token_cap) for j in jobs),
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
            # OpenAI/vLLM batch completions identify the corresponding input prompt
            # with choice.index. Response list order is not a mapping guarantee.
            mapped = [""] * len(jobs)
            seen_indices = set()
            for choice in choices:
                index = choice.get("index")
                if not isinstance(index, int) or not 0 <= index < len(jobs):
                    raise RuntimeError(f"invalid choice.index: {index!r}")
                if index in seen_indices:
                    raise RuntimeError(f"duplicate choice.index: {index}")
                seen_indices.add(index)
                mapped[index] = choice.get("text", "")
            if len(seen_indices) != len(jobs):
                raise RuntimeError(
                    f"expected indices 0..{len(jobs)-1}, received {sorted(seen_indices)}"
                )
            return mapped
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(min(2 ** attempt, 8))
    if verbose:
        print(f"Request failed at {url} after {retries + 1} attempts: {last_error}")
    return [""] * len(jobs)


def make_jobs(data, prior, target_count, max_samples, reverse):
    groups = defaultdict(list)
    items = reversed(data.items()) if reverse else data.items()
    selected = 0
    for sentence, record in items:
        existing = unique_strings(
            (record.get("hard_negatives") or []) + prior.get(sentence, [])
        )
        needed = target_count - len(existing)
        if needed <= 0:
            continue
        groups[needed].append({
            "sentence": sentence,
            "existing": existing,
            "needed": needed,
            "verb_phrases": record.get("verb_phrases") or [],
            "request_id": request_id_for(sentence),
        })
        selected += 1
        if max_samples is not None and selected >= max_samples:
            break
    return groups


def main():
    parser = argparse.ArgumentParser(
        description="Rollingly fill sentence-keyed samples to N hard negatives."
    )
    parser.add_argument("--input_json", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--prompt_file", required=True)
    parser.add_argument("--model_path", default="Qwen/Qwen2.5-72B-Instruct")
    parser.add_argument("--model_url", default="http://localhost:8000/v1/completions")
    parser.add_argument("--target_count", type=int, default=10)
    parser.add_argument("--req_batch_size", type=int, default=16)
    parser.add_argument("--parallel_workers", type=int, default=4)
    parser.add_argument("--request_timeout", type=int, default=300)
    parser.add_argument("--request_retries", type=int, default=2)
    parser.add_argument("--max_tokens_cap", type=int, default=1024)
    parser.add_argument("--flush_every", type=int, default=100)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument(
        "--accept_legacy_output", action="store_true",
        help="Count pre-v2 JSONL output as prior results (unsafe if it contains misalignment)",
    )
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if (args.target_count <= 0 or args.req_batch_size <= 0
            or args.parallel_workers <= 0 or args.flush_every <= 0
            or args.request_retries < 0):
        parser.error("count/batch/worker/flush values must be positive; retries cannot be negative")

    with open(args.prompt_file, encoding="utf-8") as f:
        base_prompt = f.read().strip()
    with open(args.input_json, encoding="utf-8") as f:
        data = json.load(f)
    prior = load_prior_generations(args.output_jsonl, args.accept_legacy_output)
    groups = make_jobs(data, prior, args.target_count, args.max_samples, args.reverse)

    total = sum(len(jobs) for jobs in groups.values())
    print(f"Eligible samples this pass: {total}")
    for needed in sorted(groups, reverse=True):
        print(f"  missing {needed}: {len(groups[needed])}")
    if total == 0:
        print("Nothing to generate.")
        return

    urls = [u.strip() for u in args.model_url.split(",") if u.strip()]
    if not urls:
        parser.error("No valid model_url")

    stats = Counter()
    written_since_flush = 0
    with open(args.output_jsonl, "a", encoding="utf-8") as fout:
        with tqdm(total=total, desc="Generating", unit="sample") as progress:
            # Separate deficit groups mean each API batch shares a smaller exact
            # requested count and an appropriately reduced generation token budget.
            for needed in sorted(groups, reverse=True):
                jobs = groups[needed]
                batch_count = (len(jobs) + args.req_batch_size - 1) // args.req_batch_size
                with ThreadPoolExecutor(max_workers=args.parallel_workers) as pool:
                    # Keep only a bounded number of futures alive. With millions of
                    # samples, submitting every batch up front wastes many GB of RAM.
                    pending = {}
                    next_batch = 0

                    def submit_one(batch_index):
                        start = batch_index * args.req_batch_size
                        batch = jobs[start:start + args.req_batch_size]
                        url = urls[batch_index % len(urls)]
                        future = pool.submit(
                            post_batch, url, args.model_path, batch, base_prompt,
                            args.request_timeout, args.max_tokens_cap,
                            args.request_retries, args.verbose,
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
                            raw_outputs = future.result()
                            for job, raw in zip(batch, raw_outputs):
                                forbidden = {normalized(x) for x in job["existing"]}
                                forbidden.add(normalized(job["sentence"]))
                                accepted = []
                                candidates, parse_status = parse_generation(
                                    raw, job["request_id"]
                                )
                                if parse_status != "ok":
                                    stats["rejected_" + parse_status] += 1
                                for candidate in candidates:
                                    key = normalized(candidate)
                                    if not key or key in forbidden:
                                        stats["rejected_duplicate_or_original"] += 1
                                        continue
                                    if not background_preserved(
                                            job["sentence"], candidate, job["verb_phrases"]
                                    ):
                                        stats["rejected_background_mismatch"] += 1
                                        continue
                                    forbidden.add(key)
                                    accepted.append(candidate)
                                    if len(accepted) >= job["needed"]:
                                        break
                                if accepted:
                                    result = {
                                        job["sentence"]: {
                                            "hard_negatives": accepted,
                                            "generator_version": GENERATOR_VERSION,
                                            "request_id": job["request_id"],
                                        }
                                    }
                                    fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                                    written_since_flush += 1
                                    if written_since_flush >= args.flush_every:
                                        fout.flush()
                                        written_since_flush = 0
                                    stats["samples_with_new_output"] += 1
                                    stats["new_hard_negatives"] += len(accepted)
                                else:
                                    stats["samples_without_new_output"] += 1
                                if len(accepted) < job["needed"]:
                                    stats["samples_still_incomplete"] += 1
                                progress.update(1)
                            if next_batch < batch_count:
                                submit_one(next_batch)
                                next_batch += 1
            fout.flush()

    print("Pass complete. Re-run the same command to fill remaining deficits.")
    for key in sorted(stats):
        print(f"{key}: {stats[key]}")


if __name__ == "__main__":
    main()
