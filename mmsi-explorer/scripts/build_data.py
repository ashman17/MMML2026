from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image, ImageOps


def main() -> None:
    source = Path(sys.argv[1]).resolve()
    dist = Path(__file__).resolve().parents[1] / "dist"
    image_dir = dist / "assets" / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    table = pq.read_table(source)
    rows = table.to_pylist()
    records = []

    for row in rows:
        question_id = int(row["id"])
        image_paths = []
        for index, raw in enumerate(row["images"] or []):
            output = image_dir / f"{question_id}_{index}.webp"
            with Image.open(io.BytesIO(raw)) as image:
                image = ImageOps.exif_transpose(image).convert("RGB")
                image.thumbnail((720, 720), Image.Resampling.LANCZOS)
                image.save(output, "WEBP", quality=78, method=4)
            image_paths.append(f"assets/images/{output.name}")

        difficulty_raw = str(row.get("difficulty") or "unknown").strip().lower()
        difficulty = difficulty_raw.split()[-1]
        if difficulty not in {"easy", "medium", "hard"}:
            difficulty = "unknown"

        records.append(
            {
                "id": question_id,
                "category": row["question_type"],
                "question": row["question"],
                "answer": row["answer"],
                "thought": row["thought"],
                "difficulty": difficulty,
                "humanTime": row.get("mean_normed_duration_seconds"),
                "images": image_paths,
            }
        )

    (dist / "records.json").write_text(
        json.dumps(records, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
    )
    print(f"Prepared {len(records)} questions and {sum(len(r['images']) for r in records)} images")


if __name__ == "__main__":
    main()
