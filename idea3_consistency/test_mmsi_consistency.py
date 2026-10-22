import importlib.util
import pathlib
import sys
import unittest


MODULE_PATH = pathlib.Path(__file__).with_name("mmsi_consistency.py")
SPEC = importlib.util.spec_from_file_location("mmsi_consistency", MODULE_PATH)
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


class ConsistencyTests(unittest.TestCase):
    def test_parse_and_rebuild_options(self):
        q = "Where is it?\nOptions: A: left, B: center, C: right, D: unknown"
        stem, opts = m.parse_question(q)
        self.assertEqual(opts, ["left", "center", "right", "unknown"])
        self.assertEqual(m.parse_question(m.build_question(stem, opts))[1], opts)

    def test_mirror_swaps_direction_words_and_preserves_case(self):
        text = "Left, rightmost, clockwise, and counter-clockwise."
        self.assertEqual(m.mirror_text(text),
                         "Right, leftmost, counterclockwise, and clockwise.")

    def test_mirror_blocks_compass_compounds(self):
        for word in ["north", "Northeast", "northwest", "Southeast", "southwest"]:
            self.assertIsNotNone(m.MIRROR_BLOCK.search(f"Move {word} now"), word)

    def test_shuffle_prediction_maps_to_original_letter(self):
        # Displayed A contains original option C.
        self.assertEqual(m.map_back("A", [2, 0, 3, 1]), "C")
        self.assertIsNone(m.map_back(None, [2, 0, 3, 1]))

    def test_exact_mcnemar(self):
        a = [True, True, False, False]
        b = [False, True, True, False]
        self.assertEqual(m.mcnemar_exact(a, b), (1, 1, 1.0))

    def test_auroc(self):
        self.assertEqual(m.auroc([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]), 1.0)


FA_SPEC = importlib.util.spec_from_file_location(
    "failure_analysis", MODULE_PATH.with_name("failure_analysis.py"))
sys.path.insert(0, str(MODULE_PATH.parent))
fa = importlib.util.module_from_spec(FA_SPEC)
FA_SPEC.loader.exec_module(fa)


def _rec(orig_q, orig_pred, **variants):
    v = {"orig": dict(question=orig_q, pred=orig_pred, pred_raw=orig_pred, perm=None)}
    v.update(variants)
    return {"variants": v, "answer": "A"}


class FailureAnalysisTests(unittest.TestCase):
    Q = "Where is the door?\nOptions: A: on the left, B: on the right, C: behind"

    def test_mirror_lexical_anchoring(self):
        mq = m.mirror_text(self.Q)  # A: on the right, B: on the left
        for pred, want in [("A", "same"), ("B", "lexical"), ("C", "other")]:
            r = _rec(self.Q, "A", mirror=dict(question=mq, pred=pred))
            self.assertEqual(fa.mirror_anchoring(r), want)
        r = _rec(self.Q, "C", mirror=dict(question=mq, pred="C"))
        self.assertIsNone(fa.mirror_anchoring(r))  # "behind" has no swapped twin

    def test_shuffle_position_anchoring(self):
        perm = [2, 0, 1]  # displayed A = original C, displayed B = original A
        for raw, want in [("B", "content"), ("A", "position"), ("C", "other")]:
            r = _rec(self.Q, "A", shuffle=dict(pred_raw=raw, pred=m.map_back(raw, perm), perm=perm))
            self.assertEqual(fa.shuffle_anchoring(r), want)
        r = _rec(self.Q, "A", shuffle=dict(pred_raw="A", pred="A", perm=[0, 2, 1]))
        self.assertIsNone(fa.shuffle_anchoring(r))  # original answer did not move


IP_SPEC = importlib.util.spec_from_file_location(
    "image_probes", MODULE_PATH.with_name("image_probes.py"))
ip = importlib.util.module_from_spec(IP_SPEC)
IP_SPEC.loader.exec_module(ip)


class ImageProbeTests(unittest.TestCase):
    def test_split_crops_overlap(self):
        import io
        from PIL import Image
        buf = io.BytesIO(); Image.new("RGB", (100, 40)).save(buf, "JPEG")
        left, right = (Image.open(io.BytesIO(b)) for b in ip.split_image(buf.getvalue()))
        self.assertEqual((left.width, right.width, left.height), (60, 60, 40))  # 20% shared

    def test_split_skips_counting_and_camera_motion(self):
        self.assertFalse(ip.splittable(dict(question="How many chairs?", question_type="MSR")))
        self.assertFalse(ip.splittable(dict(question="Where?", question_type="Motion (Cam.)")))
        self.assertTrue(ip.splittable(dict(question="Where?", question_type="MSR")))


if __name__ == "__main__":
    unittest.main()
