from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import cv2
import numpy as np
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from measure_dino_overlap import pairwise_auc, select_records


def image_tensor(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


def coverage(points: np.ndarray, image_size: tuple[float, float]) -> float:
    if len(points) < 3:
        return 0.0
    hull = cv2.convexHull(points.astype(np.float32))
    width, height = image_size
    return float(cv2.contourArea(hull) / max(width * height, 1.0))


def sparse_geometry(
    points0: np.ndarray,
    points1: np.ndarray,
    scores: np.ndarray,
    image_size0: tuple[float, float],
    image_size1: tuple[float, float],
) -> dict[str, object]:
    count = len(points0)
    if count < 4:
        return {
            "matches": count,
            "median_match_confidence": float(np.median(scores)) if count else 0.0,
            "best_model": "none",
            "geometric_inliers": 0,
            "geometric_inlier_ratio": 0.0,
            "source_coverage": 0.0,
            "target_coverage": 0.0,
            "geometric_score": 0.0,
        }

    _, homography_mask = cv2.findHomography(
        points0, points1, cv2.RANSAC, 3.0, maxIters=10_000, confidence=0.999
    )
    masks: list[tuple[str, np.ndarray]] = []
    if homography_mask is not None:
        masks.append(("homography", homography_mask.ravel().astype(bool)))
    if count >= 8:
        _, fundamental_mask = cv2.findFundamentalMat(
            points0, points1, cv2.FM_RANSAC, 1.5, 0.999, 10_000
        )
        if fundamental_mask is not None:
            masks.append(("fundamental", fundamental_mask.ravel().astype(bool)))
    if not masks:
        return {
            "matches": count,
            "median_match_confidence": float(np.median(scores)),
            "best_model": "none",
            "geometric_inliers": 0,
            "geometric_inlier_ratio": 0.0,
            "source_coverage": 0.0,
            "target_coverage": 0.0,
            "geometric_score": 0.0,
        }

    model, inliers = max(masks, key=lambda item: int(item[1].sum()))
    inlier_count = int(inliers.sum())
    ratio = inlier_count / count
    source_coverage = coverage(points0[inliers], image_size0)
    target_coverage = coverage(points1[inliers], image_size1)
    median_confidence = float(np.median(scores[inliers])) if inlier_count else 0.0
    # Coverage suppresses locally repeated patterns. This continuous score is
    # compared with controls; it is not directly converted into proof evidence.
    score = inlier_count * ratio * median_confidence * np.sqrt(
        max(min(source_coverage, target_coverage), 0.0)
    )
    return {
        "matches": count,
        "median_match_confidence": median_confidence,
        "best_model": model,
        "geometric_inliers": inlier_count,
        "geometric_inlier_ratio": ratio,
        "source_coverage": source_coverage,
        "target_coverage": target_coverage,
        "geometric_score": float(score),
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("records", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--lightglue-repo", type=Path, required=True)
    parser.add_argument("--max-keypoints", type=int, default=512)
    parser.add_argument("--resize", type=int, default=512)
    args = parser.parse_args()

    repo = args.lightglue_repo.resolve()
    sys.path.insert(0, str(repo))
    from lightglue import ALIKED, LightGlue
    from lightglue.utils import rbd

    records_path = args.records.resolve()
    image_root = records_path.parent
    records = json.loads(records_path.read_text(encoding="utf-8"))
    selected = select_records(records, args.limit)
    started = time.perf_counter()

    extractor = ALIKED(max_num_keypoints=args.max_keypoints).eval()
    matcher = LightGlue(
        features="aliked",
        depth_confidence=0.9,
        width_confidence=0.95,
        filter_threshold=0.1,
    ).eval()

    paths: list[Path] = []
    for record in selected:
        paths.extend([image_root / p for p in record["images"][:2]])
    features = []
    with torch.inference_mode():
        for path in paths:
            features.append(extractor.extract(image_tensor(path), resize=args.resize))

    def match(source_index: int, target_index: int) -> dict[str, object]:
        with torch.inference_mode():
            output = rbd(matcher({
                "image0": features[source_index],
                "image1": features[target_index],
            }))
        source = rbd(features[source_index])
        target = rbd(features[target_index])
        pairs = output["matches"].cpu().numpy()
        scores = output["scores"].cpu().numpy()
        points0 = source["keypoints"][pairs[:, 0]].cpu().numpy()
        points1 = target["keypoints"][pairs[:, 1]].cpu().numpy()
        size0 = tuple(float(v) for v in source["image_size"].cpu().numpy())
        size1 = tuple(float(v) for v in target["image_size"].cpu().numpy())
        return sparse_geometry(points0, points1, scores, size0, size1)

    within = []
    negatives = []
    offset = max(1, len(selected) // 2)
    for index, record in enumerate(selected):
        within.append({
            "id": record["id"],
            "category": record["category"],
            "difficulty": record["difficulty"],
            "image_pair": record["images"][:2],
            **match(2 * index, 2 * index + 1),
        })
        negative_index = (index + offset) % len(selected)
        negatives.append({
            "source_id": record["id"],
            "target_id": selected[negative_index]["id"],
            **match(2 * index, 2 * negative_index + 1),
        })

    within_scores = np.asarray([item["geometric_score"] for item in within])
    negative_scores = np.asarray([item["geometric_score"] for item in negatives])
    negative_p95 = float(np.quantile(negative_scores, 0.95, method="higher"))
    for item in (*within, *negatives):
        item["passes_negative_control_p95"] = item["geometric_score"] > negative_p95

    commit = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    cache = Path(torch.hub.get_dir()) / "checkpoints"
    weights = {
        name: {
            "path": str(cache / name),
            "sha256": file_sha256(cache / name),
        }
        for name in ("aliked-n16.pth", "aliked_lightglue_v0-1_arxiv.pth")
    }
    payload = {
        "method": "Official ALIKED + LightGlue; homography/fundamental RANSAC",
        "status_warning": "Negative-control threshold is a feasibility diagnostic, not calibrated proof evidence.",
        "scope_warning": "First two images only; mismatched questions are assumed unrelated but may occasionally share scenes.",
        "provenance": {
            "lightglue_commit": commit,
            "weights": weights,
            "torch": torch.__version__,
            "opencv": cv2.__version__,
            "max_keypoints": args.max_keypoints,
            "resize": args.resize,
            "device": "cpu",
        },
        "summary": {
            "questions_tested": len(within),
            "geometric_score_auc_vs_mismatched_pairs": pairwise_auc(within_scores, negative_scores),
            "negative_control_p95_score": negative_p95,
            "within_pairs_above_negative_p95": int((within_scores > negative_p95).sum()),
            "negative_pairs_above_negative_p95": int((negative_scores > negative_p95).sum()),
            "elapsed_seconds": time.perf_counter() - started,
        },
        "measurements": within,
        "negative_controls": negatives,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
