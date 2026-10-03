from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time

import numpy as np
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fvpg import measure_patch_correspondence


MEAN = torch.tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
STD = torch.tensor((0.229, 0.224, 0.225)).view(3, 1, 1)


def prepare_py39_repo(source: Path, destination: Path) -> Path:
    """Create a temporary Python-3.9-compatible view of the cached DINOv2 repo."""
    if sys.version_info >= (3, 10):
        return source
    shutil.copytree(source, destination)
    for path in destination.rglob("*.py"):
        content = path.read_text(encoding="utf-8")
        if not content.startswith("from __future__ import annotations"):
            path.write_text("from __future__ import annotations\n" + content, encoding="utf-8")
    return destination


def load_model(repo: Path, weights: Path, device: torch.device) -> torch.nn.Module:
    model = torch.hub.load(str(repo), "dinov2_vits14", source="local", pretrained=False)
    state = torch.load(weights, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    return model.eval().to(device)


def load_image(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        image = image.convert("RGB").resize((224, 224), Image.Resampling.BICUBIC)
        array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    return (tensor - MEAN) / STD


def select_records(records: list[dict[str, object]], limit: int) -> list[dict[str, object]]:
    groups: dict[str, list[dict[str, object]]] = {}
    for record in records:
        if "Positional Relationship" not in str(record["category"]):
            continue
        if len(record.get("images", [])) < 2:
            continue
        groups.setdefault(str(record["category"]), []).append(record)
    selected: list[dict[str, object]] = []
    for index in range(max(map(len, groups.values()))):
        for category in sorted(groups):
            if index < len(groups[category]):
                selected.append(groups[category][index])
                if len(selected) == limit:
                    return selected
    return selected


def pairwise_auc(positive: np.ndarray, negative: np.ndarray) -> float:
    """Probability that a within-question score exceeds a mismatched score."""
    comparisons = positive[:, None] - negative[None, :]
    return float(((comparisons > 0).sum() + 0.5 * (comparisons == 0).sum()) / comparisons.size)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("records", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    args = parser.parse_args()

    records_path = args.records.resolve()
    image_root = records_path.parent
    records = json.loads(records_path.read_text(encoding="utf-8"))
    selected = select_records(records, args.limit)
    device = torch.device(args.device)
    started = time.perf_counter()

    with tempfile.TemporaryDirectory(prefix="fvpg-dinov2-") as temporary:
        repo = prepare_py39_repo(args.repo.resolve(), Path(temporary) / "repo")
        model = load_model(repo, args.weights.resolve(), device)
        image_paths = []
        for record in selected:
            image_paths.extend([image_root / p for p in record["images"][:2]])
        batches: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, len(image_paths), 8):
                batch = torch.stack([load_image(path) for path in image_paths[start:start + 8]]).to(device)
                features = model.forward_features(batch)["x_norm_patchtokens"]
                batches.append(features.cpu().numpy())
        descriptors = np.concatenate(batches, axis=0)

    measurements = []
    negative_controls = []
    for index, record in enumerate(selected):
        measurement = measure_patch_correspondence(
            descriptors[2 * index], descriptors[2 * index + 1], (16, 16)
        )
        measurements.append({
            "id": record["id"],
            "category": record["category"],
            "difficulty": record["difficulty"],
            "image_pair": record["images"][:2],
            **measurement.to_dict(),
        })
        negative_index = (index + max(1, len(selected) // 2)) % len(selected)
        negative = measure_patch_correspondence(
            descriptors[2 * index], descriptors[2 * negative_index + 1], (16, 16)
        )
        negative_controls.append({
            "source_id": record["id"],
            "target_id": selected[negative_index]["id"],
            **negative.to_dict(),
        })

    categories: dict[str, dict[str, int]] = {}
    for measurement in measurements:
        bucket = categories.setdefault(str(measurement["category"]), {"tested": 0, "supported": 0})
        bucket["tested"] += 1
        bucket["supported"] += measurement["provisional_status"] == "supported"
    elapsed = time.perf_counter() - started
    within_scores = np.asarray([m["geometric_score"] for m in measurements])
    negative_scores = np.asarray([m["geometric_score"] for m in negative_controls])
    negative_p95 = float(np.quantile(negative_scores, 0.95, method="higher"))
    auc = pairwise_auc(within_scores, negative_scores)
    for measurement in measurements:
        measurement["passes_negative_control_p95"] = measurement["geometric_score"] > negative_p95
    for measurement in negative_controls:
        measurement["passes_negative_control_p95"] = measurement["geometric_score"] > negative_p95
    payload = {
        "method": "DINOv2 ViT-S/14 patch MNN + deterministic affine consistency",
        "status_warning": (
            "The original provisional rule has excessive negative-control passes and must not be "
            "used as evidence. The negative-control p95 field is diagnostic, not calibrated proof."
        ),
        "scope_warning": (
            "Only the first two images are tested. Mismatched-question pairs are assumed unrelated, "
            "but the dataset does not guarantee that every such pair is a true negative."
        ),
        "summary": {
            "questions_tested": len(measurements),
            "provisional_support": sum(m["provisional_status"] == "supported" for m in measurements),
            "negative_control_provisional_support": sum(
                m["provisional_status"] == "supported" for m in negative_controls
            ),
            "geometric_score_auc_vs_mismatched_pairs": auc,
            "negative_control_p95_score": negative_p95,
            "within_pairs_above_negative_p95": int((within_scores > negative_p95).sum()),
            "negative_pairs_above_negative_p95": int((negative_scores > negative_p95).sum()),
            "elapsed_seconds": elapsed,
            "device": args.device,
            "by_category": categories,
        },
        "measurements": measurements,
        "negative_controls": negative_controls,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
