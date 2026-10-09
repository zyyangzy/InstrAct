#!/usr/bin/env python3
"""
Demo:
  python3 post_process/post_process_hard_negatives.py \
    --input data_by_sentence.json \
    --output data_by_sentence_post_processed.json

Deduplicate an already merged/processed file without re-running other repairs:
  python3 post_process/post_process_hard_negatives.py \
    --input all_merged.json \
    --output all_merged_deduplicated.json \
    --dedupe-only

Post-process semantic hard negatives without third-party NLP dependencies.

The processor is intentionally conservative:
1. Rejects truncated generations (especially candidates missing an original suffix).
2. Uses verb_phrases to locate action-token slots in the original sentence.
3. Restores edits outside those slots from the original sentence.
4. Rejects candidates that become identical to the original after restoration.
5. Deduplicates the surviving candidates.

Input and output use the data_by_sentence.json/all.json-style dictionary schema.
"""

import argparse
import json
import re
from collections import Counter
from difflib import SequenceMatcher


TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)*|[^\w\s]", re.UNICODE)
WORD_RE = re.compile(r"^[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)*$")
PARTICLES = {
    "about", "across", "apart", "around", "away", "back", "down", "in",
    "into", "off", "on", "onto", "out", "over", "through", "together",
    "up", "with",
}
NEGATIONS = {"no", "not", "never", "neither", "nor", "without"}
META_RE = re.compile(r"(?:REQUEST_ID|END_OF_OUTPUT|<think>|</think>|\b(?:input|outputs?|analysis)\s*:)", re.I)
BAD_AUX_GERUND_RE = re.compile(
    r"\b(?:(?:i|you|we|they|he|she|it)'ll|"
    r"(?:i|you|we|they|he|she|it)\s+(?:will|would|could|should|can|must))"
    r"\s+[a-z]+ing\b",
    re.I,
)
BAD_TO_GERUND_RE = re.compile(
    r"\b(?:begin|begins|began|start|starts|started)\s+to\s+[a-z]+ing\b", re.I
)
DANGLING_END_RE = re.compile(r"\b(?:and|or|to|the|a|an|then)\s*[.!?]*$", re.I)


def tokenize(text):
    return TOKEN_RE.findall(text)


def norm(token):
    return token.lower().replace("’", "'")


def simple_stem(word):
    """Small dependency-free stemmer used only to match action inflections."""
    word = norm(word)
    if len(word) > 5 and word.endswith("ing"):
        stem = word[:-3]
        if len(stem) > 3 and stem[-1] == stem[-2]:
            stem = stem[:-1]
        return stem
    if len(word) > 4 and word.endswith("ied"):
        return word[:-3] + "y"
    if len(word) > 4 and word.endswith("ed"):
        stem = word[:-2]
        if len(stem) > 3 and stem[-1] == stem[-2]:
            stem = stem[:-1]
        return stem
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 4 and word.endswith("es"):
        return word[:-2]
    if len(word) > 3 and word.endswith("s"):
        return word[:-1]
    return word


def comparison_key(text):
    """Ignore case, whitespace, and punctuation for identity/deduplication."""
    return " ".join(norm(token) for token in tokenize(text) if WORD_RE.match(token))


def protected_counter(text, vocabulary):
    return Counter(
        norm(token) for token in tokenize(text)
        if WORD_RE.match(token) and norm(token) in vocabulary
    )


def high_confidence_rejection(original, candidate, max_length_ratio):
    """Return a reason for failures that cannot be repaired safely."""
    if META_RE.search(candidate):
        return "meta_leakage"
    candidate_bad_structure = (
        BAD_AUX_GERUND_RE.search(candidate) or BAD_TO_GERUND_RE.search(candidate)
    )
    original_bad_structure = (
        BAD_AUX_GERUND_RE.search(original) or BAD_TO_GERUND_RE.search(original)
    )
    if candidate_bad_structure and not original_bad_structure:
        return "malformed_structure"
    if DANGLING_END_RE.search(candidate) and not DANGLING_END_RE.search(original):
        return "dangling_fragment"
    original_words = [token for token in tokenize(original) if WORD_RE.match(token)]
    candidate_words = [token for token in tokenize(candidate) if WORD_RE.match(token)]
    if not candidate_words:
        return "empty"
    if len(candidate_words) / max(1, len(original_words)) > max_length_ratio:
        return "overlong"
    if protected_counter(original, NEGATIONS) != protected_counter(candidate, NEGATIONS):
        return "negation_changed"
    # Quantities are background for this task and must never be changed.
    original_numbers = Counter(norm(t) for t in original_words if any(c.isdigit() for c in t))
    candidate_numbers = Counter(norm(t) for t in candidate_words if any(c.isdigit() for c in t))
    if original_numbers != candidate_numbers:
        return "number_changed"
    return None


def detokenize(tokens):
    """Readable English detokenization; exact casing comes from original/candidate tokens."""
    if not tokens:
        return ""
    text = " ".join(tokens)
    text = re.sub(r"\s+([,.;:!?%\)\]\}])", r"\1", text)
    text = re.sub(r"([\(\[\{])\s+", r"\1", text)
    text = re.sub(r"\s+(['’])\s+", r"\1", text)
    return text.strip()


def action_slots(original_tokens, verb_phrases):
    """Map compact verb phrases back to original token indices.

    The first phrase word is treated as the verb. Phrase particles are also allowed
    edit slots. Other phrase words (usually objects/background nouns) remain fixed.
    """
    original_norm = [norm(t) for t in original_tokens]
    original_stems = [simple_stem(t) if WORD_RE.match(t) else norm(t) for t in original_tokens]
    verb_slots = set()
    particle_slots = set()
    cursor = 0
    for phrase in verb_phrases or []:
        words = [norm(t) for t in tokenize(str(phrase)) if WORD_RE.match(t)]
        if not words:
            continue
        verb = words[0]
        verb_stem = simple_stem(verb)
        positions = [i for i in range(cursor, len(original_stems))
                     if original_stems[i] == verb_stem]
        if not positions:
            positions = [i for i, token in enumerate(original_stems) if token == verb_stem]
        if not positions:
            continue
        verb_i = positions[0]
        verb_slots.add(verb_i)
        last = verb_i
        # Locate only verb particles from the compact phrase; nouns stay background.
        for word in words[1:]:
            if word not in PARTICLES:
                continue
            found = next((i for i in range(last + 1, min(len(original_norm), verb_i + 9))
                          if original_norm[i] == word), None)
            if found is not None:
                particle_slots.add(found)
                last = found
        cursor = verb_i + 1
    return verb_slots, particle_slots


def has_truncated_suffix(opcodes, original_tokens, candidate_tokens, min_ratio):
    original_words = sum(WORD_RE.match(t) is not None for t in original_tokens)
    candidate_words = sum(WORD_RE.match(t) is not None for t in candidate_tokens)
    if original_words and candidate_words / original_words < min_ratio:
        return True
    # A generation ending while a meaningful original suffix is still missing.
    if opcodes:
        tag, i1, i2, j1, j2 = opcodes[-1]
        if tag == "delete" and j2 == len(candidate_tokens):
            missing_words = [t for t in original_tokens[i1:i2] if WORD_RE.match(t)]
            if missing_words:
                return True
    return False


def repair_candidate(original, candidate, verb_phrases, min_length_ratio):
    original_tokens = tokenize(original)
    candidate_tokens = tokenize(candidate)
    if not original_tokens or not candidate_tokens:
        return None, "empty"
    verb_slots, particle_slots = action_slots(original_tokens, verb_phrases)
    slots = verb_slots | particle_slots
    if not slots:
        return None, "no_action_slots"

    a = [norm(t) for t in original_tokens]
    b = [norm(t) for t in candidate_tokens]
    matcher = SequenceMatcher(None, a, b, autojunk=False)
    opcodes = matcher.get_opcodes()
    if has_truncated_suffix(opcodes, original_tokens, candidate_tokens, min_length_ratio):
        return None, "truncated"

    rebuilt = []
    action_changed = False
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "equal":
            rebuilt.extend(original_tokens[i1:i2])
            continue
        touched = slots.intersection(range(i1, i2))
        if tag == "insert":
            # Only action particles may be inserted beside an action slot. Arbitrary
            # inserted nouns/adverbs are background changes and are discarded.
            if i1 in slots or i1 - 1 in slots:
                additions = [t for t in candidate_tokens[j1:j2] if norm(t) in PARTICLES]
                rebuilt.extend(additions)
                action_changed = action_changed or bool(additions)
            continue
        if tag == "replace" and touched:
            candidate_words = [t for t in candidate_tokens[j1:j2] if WORD_RE.match(t)]
            used_replacement = False
            for original_i in range(i1, i2):
                if original_i in verb_slots and candidate_words and not used_replacement:
                    # Keep one replacement action word, while preserving every noun
                    # or adverb that shared this SequenceMatcher replacement block.
                    rebuilt.append(candidate_words[0])
                    used_replacement = True
                    action_changed = action_changed or norm(candidate_words[0]) != norm(original_tokens[original_i])
                elif original_i in particle_slots:
                    replacement = next((t for t in candidate_words if norm(t) in PARTICLES), None)
                    if replacement is not None:
                        rebuilt.append(replacement)
                        action_changed = action_changed or norm(replacement) != norm(original_tokens[original_i])
                    else:
                        rebuilt.append(original_tokens[original_i])
                else:
                    rebuilt.append(original_tokens[original_i])
        else:
            # Restore noun/adverb/background replacements and all deletions.
            rebuilt.extend(original_tokens[i1:i2])

    repaired = detokenize(rebuilt)
    if not action_changed or [norm(t) for t in tokenize(repaired)] == a:
        return None, "no_verb_change"
    return repaired, "kept"


def process_record(sentence, record, args, stats):
    hard_negatives = record.get("hard_negatives") or []
    verb_phrases = record.get("verb_phrases") or []
    kept = []
    seen = set()
    for candidate in hard_negatives:
        stats["input_hard_negatives"] += 1
        if not isinstance(candidate, str):
            stats["removed_non_string"] += 1
            continue
        candidate = candidate.strip()
        reason = high_confidence_rejection(sentence, candidate, args.max_length_ratio)
        if reason:
            stats["removed_" + reason] += 1
            continue
        repaired, reason = repair_candidate(
            sentence, candidate, verb_phrases, args.min_length_ratio
        )
        if repaired is None:
            stats["removed_" + reason] += 1
            continue
        key = comparison_key(repaired)
        if key in seen:
            stats["removed_duplicate"] += 1
            continue
        seen.add(key)
        if repaired != candidate:
            stats["repaired_background"] += 1
        kept.append(repaired)
        stats["kept_hard_negatives"] += 1
    result = dict(record)
    result["hard_negatives"] = kept
    return result


def deduplicate_record(record, stats):
    """Order-preserving, case/punctuation-normalized hard-negative deduplication."""
    kept = []
    seen = set()
    for candidate in record.get("hard_negatives") or []:
        stats["input_hard_negatives"] += 1
        if not isinstance(candidate, str):
            stats["removed_non_string"] += 1
            continue
        candidate = candidate.strip()
        key = comparison_key(candidate)
        if not key:
            stats["removed_empty"] += 1
            continue
        if key in seen:
            stats["removed_duplicate"] += 1
            continue
        seen.add(key)
        kept.append(candidate)
        stats["kept_hard_negatives"] += 1
    result = dict(record)
    result["hard_negatives"] = kept
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input sentence-keyed JSON")
    parser.add_argument("--output", required=True, help="Output processed JSON")
    parser.add_argument(
        "--min-length-ratio", type=float, default=0.72,
        help="Reject candidates shorter than this word-count ratio (default: 0.72)",
    )
    parser.add_argument(
        "--max-length-ratio", type=float, default=1.35,
        help="Reject likely appended hallucinations above this ratio (default: 1.35)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process only the first N sentences (useful for inspection)",
    )
    parser.add_argument(
        "--dedupe-only", action="store_true",
        help="Only deduplicate hard_negatives; do not run repair/filter rules",
    )
    args = parser.parse_args()
    if not 0 < args.min_length_ratio <= args.max_length_ratio:
        parser.error("require 0 < min-length-ratio <= max-length-ratio")

    with open(args.input, encoding="utf-8") as f:
        data = json.load(f)

    stats = Counter()
    output = {}
    for index, (sentence, record) in enumerate(data.items()):
        if args.limit is not None and index >= args.limit:
            break
        if args.dedupe_only:
            output[sentence] = deduplicate_record(record, stats)
        else:
            output[sentence] = process_record(sentence, record, args, stats)
        stats["sentences"] += 1

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
        f.write("\n")

    for key in sorted(stats):
        print(f"{key}: {stats[key]}")


if __name__ == "__main__":
    main()
