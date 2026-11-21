"""Convert selected BLINK validation configs to the MMSI-compatible parquet schema."""
import argparse
import io
import re
from pathlib import Path


DATASET = "BLINK-Benchmark/BLINK"
SUBSETS = (
    "Spatial_Relation",
    "Relative_Depth",
    "Multi-view_Reasoning",
    "Object_Localization",
)
IMAGE_COLUMNS = tuple(f"image_{index}" for index in range(1, 5))
LETTERS = "ABCD"
CHOICE_PREFIX = re.compile(r"^\s*(?:\(([A-D])\)|([A-D])[.):])\s*")
ANSWER = re.compile(r"^\s*\(?([A-D])\)?\s*$")


def _image_bytes(image):
    if image is None:
        return None
    if isinstance(image, (bytes, bytearray, memoryview)):
        return bytes(image)
    if isinstance(image, dict):
        if image.get("bytes") is not None:
            return bytes(image["bytes"])
        if image.get("path"):
            return Path(image["path"]).read_bytes()
    if hasattr(image, "save"):
        buffer = io.BytesIO()
        image.save(buffer, format=image.format or "PNG")
        return buffer.getvalue()
    raise TypeError(f"unsupported BLINK image value: {type(image).__name__}")


def _choice_text(choice, expected_letter):
    text = str(choice).strip()
    match = CHOICE_PREFIX.match(text)
    if match:
        label = match.group(1) or match.group(2)
        if label != expected_letter:
            raise ValueError(f"choice {expected_letter} is labeled {label}")
        text = text[match.end():].strip()
    if not text:
        raise ValueError(f"choice {expected_letter} is empty")
    return text


def convert_blink_row(row, subset, row_id):
    choices = list(row["choices"])
    if not 2 <= len(choices) <= len(LETTERS):
        raise ValueError(f"{subset} row has {len(choices)} choices; expected 2 to 4")
    choices = [_choice_text(choice, LETTERS[index])
               for index, choice in enumerate(choices)]

    match = ANSWER.match(str(row["answer"]))
    if not match or match.group(1) not in LETTERS[:len(choices)]:
        raise ValueError(f"invalid answer {row['answer']!r} for {len(choices)} choices")

    images = [_image_bytes(row.get(column)) for column in IMAGE_COLUMNS]
    images = [image for image in images if image is not None]
    if not images:
        raise ValueError(f"{subset} row has no images")

    stem = str(row["question"]).strip()
    options = ", ".join(f"{LETTERS[index]}: {choice}"
                        for index, choice in enumerate(choices))
    return {
        "id": int(row_id),
        "question": f"{stem}\nOptions: {options}",
        "answer": match.group(1),
        "images": images,
        "question_type": subset,
    }


def convert_blink_rows(rows, subset, id_offset):
    return [convert_blink_row(row, subset, id_offset + index)
            for index, row in enumerate(rows)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument("--split", default="val")
    parser.add_argument("--subsets", nargs="+", choices=SUBSETS, default=list(SUBSETS))
    args = parser.parse_args()

    import pandas as pd
    from datasets import Image, load_dataset

    converted = []
    for subset in args.subsets:
        data = load_dataset(args.dataset, subset, split=args.split)
        for column in IMAGE_COLUMNS:
            if column in data.column_names:
                data = data.cast_column(column, Image(decode=False))
        offset = SUBSETS.index(subset) * 1_000_000
        rows = convert_blink_rows(data, subset, offset)
        converted.extend(rows)
        print(f"{subset}: {len(rows)} rows")

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(converted).to_parquet(output, index=False)
    print(f"wrote {len(converted)} rows to {output}")


if __name__ == "__main__":
    main()
