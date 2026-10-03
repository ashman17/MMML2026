import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class AnnotationPipelineTest(unittest.TestCase):
    def test_merge_and_leakage_safe_evaluation(self) -> None:
        annotations = []
        for index in range(100):
            overlap = index % 2 == 0
            annotations.append({
                "key": f"pair-{index}",
                "pair_type": "synthetic",
                "source_question_id": index,
                "target_question_id": index,
                "images": [f"{index}_0.webp", f"{index}_1.webp"],
                "category": "test",
                "difficulty": "test",
                "dino_score": 1.0 if overlap else 0.1,
                "lightglue_score": 0.8 if overlap else 0.05,
                "annotation": {
                    "relationship": "overlap" if overlap else "different_scene",
                    "fraction": "medium" if overlap else None,
                },
            })
        export = {"schema_version": 1, "annotations": annotations}

        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            first = directory / "first.json"
            second = directory / "second.json"
            consensus = directory / "consensus.json"
            evaluation = directory / "evaluation.json"
            first.write_text(json.dumps(export), encoding="utf-8")
            second.write_text(json.dumps(export), encoding="utf-8")

            subprocess.run([
                sys.executable,
                str(ROOT / "scripts/merge_overlap_annotations.py"),
                str(first),
                str(second),
                "--output",
                str(consensus),
            ], check=True, capture_output=True, text=True)
            subprocess.run([
                sys.executable,
                str(ROOT / "scripts/evaluate_overlap_verifiers.py"),
                str(consensus),
                str(evaluation),
            ], check=True, capture_output=True, text=True)

            merged = json.loads(consensus.read_text(encoding="utf-8"))
            report = json.loads(evaluation.read_text(encoding="utf-8"))
            self.assertEqual(merged["consensus_pairs"], 100)
            self.assertEqual(merged["pairwise_agreement"][0]["binary_overlap_kappa"], 1.0)
            self.assertEqual(report["results"]["dino"]["evaluation"]["auc"], 1.0)
            self.assertEqual(report["results"]["lightglue"]["evaluation"]["auc"], 1.0)


if __name__ == "__main__":
    unittest.main()
