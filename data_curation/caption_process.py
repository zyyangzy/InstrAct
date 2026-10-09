"""Filter instructional captions and extract action verb phrases.

The model endpoint must implement the OpenAI-compatible ``/v1/completions`` API.
Detailed end-to-end usage instructions will be added in a later update.
"""
import argparse
import requests
import json
import os
import ast
from concurrent.futures import ThreadPoolExecutor

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **_kwargs):
        """Keep the pipeline usable when the optional progress bar is absent."""
        return iterable


def _build_prompts(captions, base_prompt, mode):
    if mode == "filter":
        prompts = [
            f'{base_prompt}\n\nNow evaluate the following sentence:\n\nSentence: "{caption}"\nAnswer:'
            for caption in captions
        ]

    elif mode == "extract":
        prompts = [
            f"""{base_prompt}

---

Now extract action verb phrases for the following NEW sentence.
Only output a Python list of strings.
Do not copy examples.
Only use information from this sentence.

Input: {caption}
Output:"""
            for caption in captions
        ]

    elif mode == "extend":
        prompts = [
            f"{base_prompt.replace('input caption', caption)}"
            for caption in captions
        ]

    else:
        raise ValueError(f"Unknown mode: {mode}")

    return prompts


def _decode_choices(choices, mode, verbose=False):
    if mode == "filter":
        out = []
        for choice in choices:
            txt = choice.get("text", "").strip().lower()
            if "yes" in txt:
                out.append("Yes")
            elif "no" in txt:
                out.append("No")
            else:
                out.append("Unknown")
        return out

    elif mode == "extract":
        verb_lists = []

        for choice in choices:
            raw_text = choice.get("text", "")
            text = raw_text.strip()

            if verbose:
                print("\nRAW RESPONSE:", repr(raw_text))

            if "Output:" in text:
                text = text.split("Output:", 1)[1].strip()

            start = text.find("[")
            end = text.find("]", start)

            if start != -1 and end != -1:
                text = text[start:end + 1]

            try:
                parsed = ast.literal_eval(text)
                if isinstance(parsed, list):
                    parsed = [str(x).strip() for x in parsed if str(x).strip()]
                    verb_lists.append(parsed)
                else:
                    verb_lists.append([])
            except Exception:
                if verbose:
                    print("Parse failed:", repr(raw_text))
                verb_lists.append([])

        return verb_lists

    elif mode == "extend":
        neg_lists = []
        for choice in choices:
            text = choice.get("text", "").strip()
            lines = [l.strip() for l in text.split("\n") if l.strip()]
            numbered = [l for l in lines if len(l) > 0 and l[0].isdigit() and ")" in l]
            numbered = [l.split(")", 1)[1].strip() for l in numbered]
            neg_lists.append(numbered)
        return neg_lists

    else:
        raise ValueError(f"Unknown mode: {mode}")


def _post_once(url, model_path, prompts, mode, verbose=False):
    try:
        payload = {
            "model": model_path,
            "prompt": prompts,
            "max_tokens": 128 if mode in ("extract", "extend") else 5,
            "temperature": 0.0,
        }

        if mode == "extract":
            payload["stop"] = ["\nInput:", "\n\nInput:", "\n---"]

        resp = requests.post(
            url,
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()

        choices = resp.json()["choices"]

        if verbose:
            print(f"[{url}] batch {len(prompts)} ok")

        return choices

    except Exception as e:
        if verbose:
            print(f"[{url}] request error: {e}")
        return [{"text": ""}] * len(prompts)


def classify_or_extract_batch(
    captions,
    base_prompt,
    model_urls,
    model_path,
    mode="filter",
    verbose=False,
    req_batch_size=32,
    parallel_workers=8,
):
    if isinstance(model_urls, str):
        model_urls = [u.strip() for u in model_urls.split(",") if u.strip()]

    assert len(model_urls) >= 1, "No valid model_url provided."

    prompts = _build_prompts(captions, base_prompt, mode)
    n = len(prompts)

    if n == 0:
        return []

    indexed = list(enumerate(prompts))

    if req_batch_size <= 0:
        req_batch_size = n

    batches = [
        indexed[i:i + req_batch_size]
        for i in range(0, n, req_batch_size)
    ]

    results_buffer = [None] * n
    futures = []

    with ThreadPoolExecutor(max_workers=parallel_workers) as ex:
        for i, batch in enumerate(batches):
            url = model_urls[i % len(model_urls)]
            batch_indices = [idx for idx, _ in batch]
            batch_prompts = [p for _, p in batch]

            futures.append(
                (
                    batch_indices,
                    ex.submit(
                        _post_once,
                        url,
                        model_path,
                        batch_prompts,
                        mode,
                        verbose,
                    ),
                )
            )

        for batch_indices, fut in futures:
            choices = fut.result()

            if len(choices) != len(batch_indices) and verbose:
                print("Warning: mismatched choices length, padding/truncating.")

            m = min(len(choices), len(batch_indices))

            for j in range(m):
                results_buffer[batch_indices[j]] = choices[j]

            for j in range(m, len(batch_indices)):
                results_buffer[batch_indices[j]] = {"text": ""}

    return _decode_choices(results_buffer, mode, verbose)


def process_video(
    video_id,
    data,
    base_prompt,
    model_urls,
    model_path,
    mode="filter",
    verbose=False,
    req_batch_size=32,
    parallel_workers=8,
):
    captions = data["text"]
    starts = data["start"]
    ends = data["end"]

    results = classify_or_extract_batch(
        captions,
        base_prompt,
        model_urls,
        model_path,
        mode,
        verbose,
        req_batch_size=req_batch_size,
        parallel_workers=parallel_workers,
    )

    if mode == "filter":
        filtered_text, filtered_start, filtered_end = [], [], []

        for caption, start, end, label in zip(captions, starts, ends, results):
            if label == "Yes":
                filtered_text.append(caption)
                filtered_start.append(start)
                filtered_end.append(end)

        return {
            video_id: {
                "text": filtered_text,
                "start": filtered_start,
                "end": filtered_end,
            }
        }

    elif mode == "extract":
        return {
            video_id: {
                "verb_phrases": results,
                "start": starts,
                "end": ends,
            }
        }

    elif mode == "extend":
        return {
            video_id: {
                "hard_negatives": results,
                "start": starts,
                "end": ends,
            }
        }


def load_completed_ids(output_jsonl_path):
    completed = set()

    if os.path.exists(output_jsonl_path):
        with open(output_jsonl_path, "r") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    completed.update(obj.keys())
                except json.JSONDecodeError:
                    continue

    return completed


def main():
    parser = argparse.ArgumentParser(
        description="Filter captions or extract verb phrases."
    )

    parser.add_argument("--input_json", type=str, required=True)
    parser.add_argument("--output_jsonl", type=str, required=True)
    parser.add_argument("--prompt_file", type=str, required=True)

    parser.add_argument(
        "--model_url",
        type=str,
        default="http://localhost:8000/v1/completions",
    )

    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--verbose", action="store_true")

    parser.add_argument("--req_batch_size", type=int, default=32)
    parser.add_argument("--parallel_workers", type=int, default=8)

    args = parser.parse_args()

    with open(args.prompt_file, "r") as f:
        base_prompt = f.read().strip()

    prompt_filename = os.path.basename(args.prompt_file).lower()

    if "filter" in prompt_filename:
        mode = "filter"
    elif "extract" in prompt_filename:
        mode = "extract"
    elif "hard" in prompt_filename:
        mode = "extend"
    else:
        raise ValueError(
            "Unable to detect mode from prompt filename. "
            "Filename must contain 'filter', 'extract', or 'hard'."
        )

    print(f"Mode: {mode}")

    with open(args.input_json, "r") as f:
        input_json = json.load(f)

    completed_ids = load_completed_ids(args.output_jsonl)
    print(f"{len(completed_ids)} video_ids already processed.")

    model_urls = [u.strip() for u in args.model_url.split(",") if u.strip()]

    with open(args.output_jsonl, "a") as fout:
        items = list(input_json.items())

        if args.reverse:
            items = list(reversed(items))

        for video_id, data in tqdm(items, desc="Processing"):
            if video_id in completed_ids:
                continue

            result = process_video(
                video_id,
                data,
                base_prompt,
                model_urls,
                args.model_path,
                mode,
                args.verbose,
                req_batch_size=args.req_batch_size,
                parallel_workers=args.parallel_workers,
            )

            fout.write(json.dumps(result) + "\n")
            fout.flush()

    print(f"Done. Results saved to {args.output_jsonl}")


if __name__ == "__main__":
    main()
