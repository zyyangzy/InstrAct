#!/usr/bin/env python3
"""
Post-process order-swapped hard negatives with conservative, dependency-free rules.

Example:
  python3 post_process/post_process_order_swapped_hn.py \
    --input exist.json \
    --output exist_order_post_processed.json

The processor:
1. Removes originals and normalized duplicates.
2. Protects quantities and negation.
3. Rejects truncation, appended hallucinations, metadata, and clear grammar fragments.
4. Allows only original vocabulary or inflections of listed action verbs.
5. Prevents unexplained repetition of content nouns.
6. Uses verb-phrase object/action anchors to reject candidates whose action order
   did not actually change when the order is mechanically verifiable.

It filters but does not attempt to rewrite malformed order swaps.
"""

import argparse
import json
import re
from collections import Counter


TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)*|[^\w\s]", re.UNICODE)
WORD_RE = re.compile(r"^[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)*$")
NEGATIONS = {"no", "not", "never", "neither", "nor", "without"}
PARTICLES = {
    "about", "across", "apart", "around", "away", "back", "down", "in",
    "into", "off", "on", "onto", "out", "over", "through", "together",
    "up", "with",
}
FUNCTION_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "but", "by",
    "for", "from", "he", "her", "hers", "him", "his", "i", "if", "in", "into",
    "is", "it", "its", "me", "my", "of", "on", "or", "our", "ours", "she",
    "so", "than", "that", "the", "their", "theirs", "them", "then", "they",
    "this", "those", "through", "to", "up", "us", "we", "when", "while", "with",
    "you", "your", "yours",
    "i'm", "i'll", "i'd", "i've", "you're", "you'll", "you'd", "you've",
    "we're", "we'll", "we'd", "we've", "they're", "they'll", "they'd",
    "they've", "he's", "he'll", "he'd", "she's", "she'll", "she'd",
    "it's", "it'll", "it'd", "that's", "there's", "here's",
    "gonna", "wanna",
}
META_RE = re.compile(
    r"(?:REQUEST_ID|END_OF_OUTPUT|<think>|</think>|\b(?:input|outputs?|analysis)\s*:)",
    re.I,
)
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
    return TOKEN_RE.findall(str(text))


def norm(token):
    return token.lower().replace("’", "'")


def words(text):
    return [norm(token) for token in tokenize(text) if WORD_RE.match(token)]


def simple_stem(word):
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
    return " ".join(words(text))


def number_counter(text):
    return Counter(token for token in words(text) if any(char.isdigit() for char in token))


def negation_counter(text):
    return Counter(token for token in words(text) if token in NEGATIONS)


def phrase_words(verb_phrases):
    return [words(phrase) for phrase in verb_phrases or [] if words(phrase)]


def action_verb_stems(verb_phrases):
    return {simple_stem(tokens[0]) for tokens in phrase_words(verb_phrases)}


def select_action_anchors(original, verb_phrases):
    """Select distinct, phrase-specific anchors that really occur in the original."""
    original_stems = [simple_stem(token) for token in words(original)]
    phrases = phrase_words(verb_phrases)
    verb_counts = Counter(simple_stem(tokens[0]) for tokens in phrases)
    anchors = []
    seen = set()
    for tokens in phrases:
        verb = simple_stem(tokens[0])
        # Distinct verbs track action movement directly. When actions share a
        # verb ("add salt", "add pepper"), their objects disambiguate order.
        candidates = (
            tokens[:1] + list(reversed(tokens[1:]))
            if verb_counts[verb] == 1
            else list(reversed(tokens[1:])) + tokens[:1]
        )
        anchor = next(
            (
                simple_stem(token) for token in candidates
                if token not in PARTICLES
                and simple_stem(token) in original_stems
                and simple_stem(token) not in seen
            ),
            None,
        )
        if anchor is not None:
            anchors.append(anchor)
            seen.add(anchor)
    return anchors


def anchor_positions(text, anchors):
    stems = [simple_stem(token) for token in words(text)]
    positions = []
    for anchor in anchors:
        try:
            positions.append(stems.index(anchor))
        except ValueError:
            return None
    return positions


def order_signature(positions):
    """Relative rank of each phrase anchor, independent of absolute token offsets."""
    return tuple(sorted(range(len(positions)), key=lambda index: positions[index]))


def candidate_failure(sentence, candidate, verb_phrases, args):
    if len(phrase_words(verb_phrases)) < 2:
        return "insufficient_action_phrases"
    original_words = words(sentence)
    candidate_words = words(candidate)
    if not candidate_words:
        return "empty"
    if META_RE.search(candidate):
        return "meta_leakage"
    if comparison_key(candidate) == comparison_key(sentence):
        return "same_as_original"
    candidate_bad_structure = (
        BAD_AUX_GERUND_RE.search(candidate) or BAD_TO_GERUND_RE.search(candidate)
    )
    original_bad_structure = (
        BAD_AUX_GERUND_RE.search(sentence) or BAD_TO_GERUND_RE.search(sentence)
    )
    if candidate_bad_structure and not original_bad_structure:
        return "malformed_structure"
    candidate_dangling = DANGLING_END_RE.search(candidate) or re.match(
        r"^\s*until\b", candidate, re.I
    )
    original_dangling = DANGLING_END_RE.search(sentence) or re.match(
        r"^\s*until\b", sentence, re.I
    )
    if candidate_dangling and not original_dangling:
        return "dangling_fragment"
    ratio = len(candidate_words) / max(1, len(original_words))
    if ratio < args.min_length_ratio:
        return "too_short"
    if ratio > args.max_length_ratio:
        return "too_long"
    if number_counter(sentence) != number_counter(candidate):
        return "number_changed"
    if negation_counter(sentence) != negation_counter(candidate):
        return "negation_changed"

    original_stems = [simple_stem(token) for token in original_words]
    candidate_stems = [simple_stem(token) for token in candidate_words]
    original_counts = Counter(original_stems)
    candidate_counts = Counter(candidate_stems)
    action_stems = action_verb_stems(verb_phrases)

    # Inflected action forms are allowed, but genuinely new vocabulary is not.
    unseen = {
        stem for stem in candidate_counts
        if stem not in original_counts
        and stem not in action_stems
        and stem not in FUNCTION_WORDS
    }
    if unseen:
        return "new_vocabulary"

    overlap = sum((original_counts & candidate_counts).values())
    if overlap / max(1, sum(original_counts.values())) < args.min_word_coverage:
        return "low_original_coverage"

    anchors = select_action_anchors(sentence, verb_phrases)
    if len(anchors) >= 2:
        original_positions = anchor_positions(sentence, anchors)
        candidate_positions = anchor_positions(candidate, anchors)
        if candidate_positions is None:
            return "missing_action_anchor"
        if order_signature(original_positions) == order_signature(candidate_positions):
            return "no_action_order_change"
    return None


def deduplicate(values, stats, field):
    kept, seen = [], set()
    for candidate in values or []:
        stats[f"input_{field}"] += 1
        if not isinstance(candidate, str):
            stats["removed_non_string"] += 1
            continue
        candidate = candidate.strip()
        key = comparison_key(candidate)
        if not key:
            stats["removed_empty"] += 1
        elif key in seen:
            stats["removed_duplicate"] += 1
        else:
            seen.add(key)
            kept.append(candidate)
            stats[f"kept_{field}"] += 1
    return kept


def process_record(sentence, record, args, stats):
    field = "order_swapped_hn"
    if args.dedupe_only:
        kept = deduplicate(record.get(field), stats, field)
    else:
        kept, seen = [], set()
        verb_phrases = record.get("verb_phrases") or []
        for candidate in record.get(field) or []:
            stats[f"input_{field}"] += 1
            if not isinstance(candidate, str):
                stats["removed_non_string"] += 1
                continue
            candidate = candidate.strip()
            failure = candidate_failure(sentence, candidate, verb_phrases, args)
            if failure:
                stats["removed_" + failure] += 1
                continue
            key = comparison_key(candidate)
            if key in seen:
                stats["removed_duplicate"] += 1
                continue
            seen.add(key)
            kept.append(candidate)
            stats[f"kept_{field}"] += 1
    result = dict(record)
    result[field] = kept
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-length-ratio", type=float, default=0.72)
    parser.add_argument("--max-length-ratio", type=float, default=1.35)
    parser.add_argument("--min-word-coverage", type=float, default=0.80)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dedupe-only", action="store_true")
    args = parser.parse_args()
    if not 0 < args.min_length_ratio <= args.max_length_ratio:
        parser.error("require 0 < min-length-ratio <= max-length-ratio")
    if not 0 < args.min_word_coverage <= 1:
        parser.error("min-word-coverage must be in (0, 1]")

    with open(args.input, encoding="utf-8") as handle:
        data = json.load(handle)
    stats = Counter()
    output = {}
    for index, (sentence, record) in enumerate(data.items()):
        if args.limit is not None and index >= args.limit:
            break
        output[sentence] = process_record(sentence, record, args, stats)
        stats["sentences"] += 1
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(output, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    for key in sorted(stats):
        print(f"{key}: {stats[key]}")


if __name__ == "__main__":
    main()
