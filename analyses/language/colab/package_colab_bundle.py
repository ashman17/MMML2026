"""Build the small zip uploaded by MMSI_blind_vlm.ipynb.

The bundle contains code plus the text cache and Step A tags. It deliberately
excludes every Step E raw file so a Colab run cannot resume the Mac
disk-offloaded smoke test.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = Path(__file__).resolve().parent / "mmsi_blind_vlm_colab.zip"
PREFIX = "mmsi_language"
FILES = [
    "common.py",
    "blind_vlm.py",
    "05_blind_vlm.py",
    "requirements-blind-vlm.txt",
    "cache/mmsi_text.parquet",
    "outputs/01_tags.csv",
    "colab/MMSI_blind_vlm.ipynb",
]


def main() -> None:
    missing = [name for name in FILES if not (ROOT / name).exists()]
    if missing:
        raise FileNotFoundError("Missing bundle inputs: " + ", ".join(missing))
    DEST.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(DEST, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in FILES:
            archive.write(ROOT / name, f"{PREFIX}/{name}")
    print(f"Wrote {DEST} ({DEST.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
