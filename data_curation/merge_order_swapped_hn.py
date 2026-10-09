#!/usr/bin/env python3
import argparse
import json
import mmap
import os
import tempfile


def iter_concatenated_json(path):
    decoder = json.JSONDecoder()
    content = open(path, "r", encoding="utf-8").read()
    position = 0
    while position < len(content):
        while position < len(content) and content[position].isspace():
            position += 1
        if position == len(content):
            return
        try:
            value, position = decoder.raw_decode(content, position)
        except json.JSONDecodeError as exc:
            # A terminated append-only producer can leave only its final JSONL
            # record incomplete. Preserve the source and ignore that trailing
            # fragment; malformed data before EOF remains a hard failure.
            if "\n" not in content[exc.pos:]:
                print(
                    f"Warning: ignoring incomplete trailing JSON in {path} "
                    f"at line {exc.lineno}, column {exc.colno}",
                    flush=True,
                )
                return
            raise
        yield value


def source_updates(source_path):
    updates = {}
    source_duplicates = 0
    for group in iter_concatenated_json(source_path):
        for sentence, record in group.items():
            values = updates.setdefault(sentence, [])
            seen = set(values)
            for candidate in record.get("order_swapped_hn", []):
                if candidate in seen:
                    source_duplicates += 1
                else:
                    values.append(candidate)
                    seen.add(candidate)
    return updates, source_duplicates


def skip_space(data, position):
    while position < len(data) and data[position] in b" \t\r\n":
        position += 1
    return position


def string_end(data, position):
    assert data[position] == ord('"')
    position += 1
    while position < len(data):
        if data[position] == ord('\\'):
            position += 2
        elif data[position] == ord('"'):
            return position + 1
        else:
            position += 1
    raise ValueError("unterminated JSON string")


def compound_end(data, position):
    opening = data[position]
    if opening not in (ord('{'), ord('[')):
        raise ValueError(f"expected object or array at byte {position}")
    depth = 0
    in_string = False
    while position < len(data):
        byte = data[position]
        if in_string:
            if byte == ord('\\'):
                position += 2
                continue
            if byte == ord('"'):
                in_string = False
        elif byte == ord('"'):
            in_string = True
        elif byte in (ord('{'), ord('[')):
            depth += 1
        elif byte in (ord('}'), ord(']')):
            depth -= 1
            if depth == 0:
                return position + 1
        position += 1
    raise ValueError("unterminated JSON value")


def main():
    parser = argparse.ArgumentParser(
        description="Merge generated order-swapped HNs into a new JSON file."
    )
    parser.add_argument("--input_json", default="exist.json")
    parser.add_argument(
        "--source_jsonl", default="order_swap/missing_order_swapped_hn.jsonl"
    )
    parser.add_argument("--output_json", default="exist_updated.json")
    args = parser.parse_args()
    input_path = os.path.abspath(args.input_json)
    output_path = os.path.abspath(args.output_json)
    if input_path == output_path:
        parser.error("output_json must differ from input_json")
    if os.path.exists(output_path):
        parser.error(f"refusing to overwrite existing output: {output_path}")

    updates, source_duplicates = source_updates(args.source_jsonl)
    generated_sentences = len(updates)
    generated_hn = sum(len(values) for values in updates.values())
    matched = added = existing_duplicates = records = 0

    target_dir = os.path.dirname(output_path)
    output_name = os.path.basename(output_path)
    fd, temporary_path = tempfile.mkstemp(
        prefix=f".{output_name}.", suffix=".tmp", dir=target_dir
    )
    try:
        with open(input_path, "rb") as source, os.fdopen(fd, "w", encoding="utf-8") as output:
            data = mmap.mmap(source.fileno(), 0, access=mmap.ACCESS_READ)
            position = skip_space(data, 0)
            if data[position] != ord('{'):
                raise ValueError(f"{input_path} is not a JSON object")
            position += 1
            output.write("{\n")
            first = True
            while True:
                position = skip_space(data, position)
                if data[position] == ord('}'):
                    break
                key_start = position
                key_finish = string_end(data, key_start)
                sentence = json.loads(data[key_start:key_finish])
                position = skip_space(data, key_finish)
                if data[position] != ord(':'):
                    raise ValueError(f"expected colon at byte {position}")
                position = skip_space(data, position + 1)
                value_finish = compound_end(data, position)
                record = json.loads(data[position:value_finish])
                records += 1

                candidates = updates.pop(sentence, None)
                if candidates is not None:
                    matched += 1
                    destination = record.setdefault("order_swapped_hn", [])
                    seen = set(destination)
                    for candidate in candidates:
                        if candidate in seen:
                            existing_duplicates += 1
                        else:
                            destination.append(candidate)
                            seen.add(candidate)
                            added += 1

                if not first:
                    output.write(",\n")
                first = False
                output.write("  ")
                output.write(json.dumps(sentence, ensure_ascii=False))
                output.write(": ")
                output.write(json.dumps(record, ensure_ascii=False, indent=2).replace("\n", "\n  "))

                position = skip_space(data, value_finish)
                if data[position] == ord(','):
                    position += 1
                elif data[position] != ord('}'):
                    raise ValueError(f"expected comma or closing brace at byte {position}")
            data.close()
            output.write("\n}\n")
            output.flush()
            os.fsync(output.fileno())

        os.chmod(temporary_path, os.stat(input_path).st_mode)
        os.replace(temporary_path, output_path)
    except BaseException:
        try:
            os.unlink(temporary_path)
        except FileNotFoundError:
            pass
        raise

    print(f"target_records={records}")
    print(f"generated_sentences={generated_sentences}")
    print(f"generated_order_swapped_hn={generated_hn}")
    print(f"matched_sentences={matched}")
    print(f"missing_sentences={len(updates)}")
    print(f"added_order_swapped_hn={added}")
    print(f"skipped_existing_duplicates={existing_duplicates}")
    print(f"skipped_source_duplicates={source_duplicates}")
    print(f"output={output_path}")


if __name__ == "__main__":
    main()
