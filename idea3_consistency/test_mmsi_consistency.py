import contextlib
import importlib.util
import io
import json
import math
import pathlib
import random
import sys
import tempfile
import unittest


MODULE_PATH = pathlib.Path(__file__).with_name("mmsi_consistency.py")
SPEC = importlib.util.spec_from_file_location("mmsi_consistency", MODULE_PATH)
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)

BLINK_SPEC = importlib.util.spec_from_file_location(
    "blink_to_parquet", MODULE_PATH.with_name("blink_to_parquet.py"))
blink = importlib.util.module_from_spec(BLINK_SPEC)
BLINK_SPEC.loader.exec_module(blink)


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

    def test_blink_row_conversion(self):
        source = {
            "question": "Is the cup above the plate?",
            "choices": ["(A) yes", "(B) no"],
            "answer": "(B)",
            "image_1": {"bytes": b"first", "path": None},
            "image_2": bytearray(b"second"),
            "image_3": None,
            "image_4": None,
        }
        row = blink.convert_blink_row(source, "Spatial_Relation", 1_000_007)
        self.assertEqual(row, {
            "id": 1_000_007,
            "question": "Is the cup above the plate?\nOptions: A: yes, B: no",
            "answer": "B",
            "images": [b"first", b"second"],
            "question_type": "Spatial_Relation",
        })
        self.assertEqual(m.parse_question(row["question"])[1], ["yes", "no"])

    def test_single_image_variants_skip_reorder(self):
        safe = {"question": "Which one?\nOptions: A: red, B: blue",
                "images": [b"image"]}
        names = [variant["name"] for variant in m.make_variants(safe, random.Random(0))]
        self.assertEqual(names, ["orig", "mirror", "shuffle"])

        unsafe = {"question": "What text is shown?\nOptions: A: red, B: blue",
                  "images": [b"image"]}
        names = [variant["name"] for variant in m.make_variants(unsafe, random.Random(0))]
        self.assertEqual(names, ["orig", "shuffle"])

    def test_stratified_sample_keeps_all_rows_when_n_is_large(self):
        rows = [{"id": index, "question_type": "odd" if index % 2 else "even"}
                for index in range(7)]
        sampled = m.stratified_sample(rows, 20, 3)
        self.assertEqual(len(sampled), len(rows))
        self.assertEqual({row["id"] for row in sampled}, {row["id"] for row in rows})


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

    def test_grid_tiles_overlap(self):
        import io
        from PIL import Image
        buf = io.BytesIO(); Image.new("RGB", (100, 40)).save(buf, "JPEG")
        tiles = [Image.open(io.BytesIO(b)).size for b in ip.grid_image(buf.getvalue())]
        self.assertEqual(tiles, [(60, 24)] * 4)

    def test_grid_shuffle_keeps_labels_and_changes_order(self):
        import io
        from PIL import Image
        buf = io.BytesIO(); Image.new("RGB", (100, 40)).save(buf, "JPEG")
        row = dict(question="Where is the chair?", question_type="MSR", images=[buf.getvalue()] * 2)
        convs = {v: c for _, v, c in ip.probe_messages([row], 768, ["grid", "grid_shuffle"])}
        labels = lambda v: [p["text"] for p in convs[v][0]["content"][:-1] if p["type"] == "text"]
        self.assertEqual(len(labels("grid")), 8)
        self.assertEqual(labels("grid")[0], "Image 1 (top-left part):")
        self.assertEqual(sorted(labels("grid")), sorted(labels("grid_shuffle")))
        self.assertNotEqual(labels("grid"), labels("grid_shuffle"))
        self.assertIn("not shown in order", convs["grid_shuffle"][0]["content"][-1]["text"])


TGCD_SPEC = importlib.util.spec_from_file_location("tgcd", MODULE_PATH.with_name("tgcd.py"))
tgcd = importlib.util.module_from_spec(TGCD_SPEC)
TGCD_SPEC.loader.exec_module(tgcd)


class TGCDTests(unittest.TestCase):
    def test_letter_score_extraction_fills_missing_letters(self):
        values = {
            1: {"decoded_token": " A", "logprob": -0.2},
            2: {"decoded_token": "C", "logprob": -1.5},
            3: {"decoded_token": "word", "logprob": -0.1},
        }
        self.assertEqual(m.extract_letter_logprobs(values),
                         {"A": -0.2, "B": -1.5, "C": -1.5, "D": -1.5})

    def test_letter_score_extraction_uses_uniform_fallback(self):
        values = {1: {"decoded_token": "word", "logprob": -0.1}}
        scores, missing = m.extract_letter_logprobs(values, return_missing=True)
        self.assertTrue(missing)
        for letter in m.LETTERS:
            self.assertAlmostEqual(scores[letter], math.log(.25))

    def test_two_option_letter_scoring(self):
        values = {
            1: {"decoded_token": " A", "logprob": -0.2},
            2: {"decoded_token": "B", "logprob": -1.5},
            3: {"decoded_token": "C", "logprob": -0.01},
        }
        scores, missing = m.extract_letter_logprobs(
            values, return_missing=True, letters="AB")
        self.assertFalse(missing)
        self.assertEqual(scores, {"A": -0.2, "B": -1.5})

        scores, missing = m.extract_letter_logprobs(
            {3: values[3]}, return_missing=True, letters="AB")
        self.assertTrue(missing)
        self.assertEqual(set(scores), {"A", "B"})
        self.assertAlmostEqual(scores["A"], math.log(.5))
        self.assertAlmostEqual(scores["B"], math.log(.5))

        mapped = tgcd.map_back_scores({"A": -0.1, "B": -0.9}, [1, 0])
        self.assertEqual(mapped, {"B": -0.1, "A": -0.9})
        aggregate = tgcd.aggregate_scores([scores, scores], "prob_mean")
        self.assertEqual(set(aggregate), {"A", "B"})

    def test_shuffle_scores_map_back_to_original_letters(self):
        shown = {"A": -0.1, "B": -0.2, "C": -0.3, "D": -0.4}
        self.assertEqual(tgcd.map_back_scores(shown, [2, 0, 3, 1]),
                         {"A": -0.2, "B": -0.4, "C": -0.1, "D": -0.3})

    def test_logprob_and_probability_aggregation(self):
        first = {"A": math.log(.8), "B": math.log(.1),
                 "C": math.log(.05), "D": math.log(.05)}
        second = {"A": math.log(.2), "B": math.log(.3),
                  "C": math.log(.25), "D": math.log(.25)}
        log_mean = tgcd.aggregate_scores([first, second], "logprob_mean")
        prob_mean = tgcd.aggregate_scores([first, second], "prob_mean")
        self.assertAlmostEqual(log_mean["A"], (math.log(.8) + math.log(.2)) / 2)
        self.assertAlmostEqual(prob_mean["A"], math.log(.5))

    def test_contrast_subtracts_normalized_noimage_logprob(self):
        aggregate = {"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0}
        noimage = {"A": math.log(.5), "B": math.log(.2),
                   "C": math.log(.2), "D": math.log(.1)}
        contrasted = tgcd.contrast_scores(aggregate, noimage, .5)
        self.assertAlmostEqual(contrasted["A"], 1.0 - .5 * math.log(.5))
        self.assertAlmostEqual(contrasted["D"], 4.0 - .5 * math.log(.1))

    def test_two_option_analysis_reports_subset_chance(self):
        record = {
            "id": 1,
            "question_type": "Spatial_Relation",
            "answer": "A",
            "n_images": 1,
            "n_options": 2,
            "variants": {
                "orig": {"scores": {"A": math.log(.7), "B": math.log(.3)},
                         "pred": "A", "missing_letters": False},
                "shuffle": {"scores": {"A": math.log(.6), "B": math.log(.4)},
                            "pred": "A", "missing_letters": False},
            },
            "noimage": {"scores": {"A": math.log(.5), "B": math.log(.5)},
                        "pred": "A", "missing_letters": False},
            "greedy_pred": "A",
            "score_pred": "A",
            "sc_preds": ["A", "B"],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "scores.jsonl"
            path.write_text(json.dumps(record) + "\n")
            with contextlib.redirect_stdout(io.StringIO()):
                tgcd.analyze(str(path), bootstrap=0, make_figure=False)
            report = (path.parent / "tgcd_report.md").read_text()
        self.assertIn("| Spatial_Relation | 1 | 50.0% |", report)


sys.path.insert(0, str(MODULE_PATH.parent))
SAT_SPEC = importlib.util.spec_from_file_location("sat_to_parquet",
                                                  MODULE_PATH.with_name("sat_to_parquet.py"))
sat = importlib.util.module_from_spec(SAT_SPEC)
SAT_SPEC.loader.exec_module(sat)


class SATTests(unittest.TestCase):
    def test_mirror_answer_swaps_left_right_option(self):
        options = ["rotated right and moved forward", "rotated left and moved forward"]
        stem = "How did the camera rotate from the first image to the second image?"
        self.assertEqual(sat.mirror_answer(stem, options, 1), "A")

    def test_mirror_answer_none_when_stem_has_direction_or_no_match(self):
        self.assertIsNone(sat.mirror_answer("If I turn left, is the sofa closer?",
                                            ["yes", "no"], 0))
        self.assertIsNone(sat.mirror_answer(
            "Was any object moved?",
            ["Chair moved left and towards the camera", "Chair moved right and away"], 0))
        self.assertIsNone(sat.mirror_answer("Is the lamp above the bed?", ["above", "below"], 0))

    def test_convert_row_shuffles_options_and_keeps_answer(self):
        source = {"question": "How did the camera rotate?",
                  "answers": ["rotated left", "rotated right"],
                  "correct_answer": "rotated left", "question_type": "action_sequence",
                  "images": [{"bytes": b"a", "path": None}, {"bytes": b"b", "path": None}]}
        for seed in range(5):
            row = sat.convert_sat_row(source, 20_000_001, random.Random(seed))
            stem, options = m.parse_question(row["question"])
            self.assertEqual(options[m.LETTERS.index(row["answer"])], "rotated left")
            self.assertEqual(options[m.LETTERS.index(row["mirror_answer"])], "rotated right")
            self.assertEqual(row["images"], [b"a", b"b"])

    def test_convert_row_skips_more_than_four_options(self):
        source = {"question": "How many chairs?", "answers": ["1", "2", "3", "4", "5"],
                  "correct_answer": "3", "question_type": "other", "images": [b"x"]}
        self.assertIsNone(sat.convert_sat_row(source, 1, random.Random(0)))


TRAIN_BUILD_SPEC = importlib.util.spec_from_file_location(
    "build_train_set", MODULE_PATH.with_name("train") / "build_train_set.py")
train_build = importlib.util.module_from_spec(TRAIN_BUILD_SPEC)
TRAIN_BUILD_SPEC.loader.exec_module(train_build)

TRAIN_DATA_SPEC = importlib.util.spec_from_file_location(
    "train_data", MODULE_PATH.with_name("train") / "data.py")
train_data = importlib.util.module_from_spec(TRAIN_DATA_SPEC)
sys.modules[TRAIN_DATA_SPEC.name] = train_data
TRAIN_DATA_SPEC.loader.exec_module(train_data)

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
PARQUET_AVAILABLE = (importlib.util.find_spec("pyarrow") is not None
                     or importlib.util.find_spec("fastparquet") is not None)


class _FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [{"A": 1, "B": 2, "C": 3, "D": 4}[text]]


class _FakeProcessor:
    def __init__(self):
        self.tokenizer = _FakeTokenizer()
        self.conversations = None

    def apply_chat_template(self, conversations, **kwargs):
        self.conversations = conversations
        images = [part["image"] for conversation in conversations
                  for part in conversation[0]["content"] if part["type"] == "image"]
        return {
            "input_ids": [[1, 1, 1] for _ in conversations],
            "attention_mask": [[1, 1, 1] for _ in conversations],
            "pixel_values": [image.getpixel((0, 0)) for image in images],
            "image_grid_thw": [[1, 1, 1] for _ in images],
        }


class TrainTests(unittest.TestCase):
    @staticmethod
    def _filter_fixture():
        rows = [
            {"id": 1, "question_type": "spatial", "mirror_answer": "B"},
            {"id": 2, "question_type": "spatial", "mirror_answer": None},
            {"id": 3, "question_type": "counting", "mirror_answer": None},
            {"id": 4, "question_type": "counting", "mirror_answer": None},
        ]
        priors = [
            {"id": 1, "noimage": {"scores": {"A": math.log(.5), "B": math.log(.5)},
                                    "missing_letters": False}},
            {"id": 2, "noimage": {"scores": {"A": math.log(.9), "B": math.log(.1)},
                                    "missing_letters": False}},
            {"id": 3, "noimage": {"scores": {"A": math.log(.5), "B": math.log(.5)},
                                    "missing_letters": True}},
            {"id": 4, "noimage": {"scores": {"A": math.log(.69), "B": math.log(.31)},
                                    "missing_letters": False}},
        ]
        return rows, priors

    def test_build_train_set_filter_logic_without_torch(self):
        rows, priors = self._filter_fixture()
        kept, counts = train_build.filter_rows(rows, priors, .7)
        self.assertEqual([row["id"] for row in kept], [1, 4])
        self.assertEqual(counts["spatial"], {"kept": 1, "dropped": 1})
        self.assertEqual(counts["counting"], {"kept": 1, "dropped": 1})
        self.assertAlmostEqual(json.loads(kept[0]["noimage_logprobs"])["A"], math.log(.5))

    @unittest.skipUnless(PARQUET_AVAILABLE, "parquet engine not installed")
    def test_build_train_set_filters_confident_text_prior(self):
        import pandas as pd
        rows, priors = self._filter_fixture()
        for row in rows:
            row.update(question="Question\nOptions: A: one, B: two", answer="A",
                       images=[b"image"])
        with tempfile.TemporaryDirectory() as directory:
            source = pathlib.Path(directory) / "source.parquet"
            prior_path = pathlib.Path(directory) / "priors.jsonl"
            output = pathlib.Path(directory) / "train.parquet"
            pd.DataFrame(rows).to_parquet(source, index=False)
            prior_path.write_text("".join(json.dumps(record) + "\n" for record in priors))
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                train_build.build_train_set(str(source), str(prior_path), .7, str(output))
            result = pd.read_parquet(output)
        self.assertEqual(result["id"].tolist(), [1, 4])
        self.assertIn("spatial: kept 1, dropped 1", stream.getvalue())
        self.assertIn("1 kept rows have mirror_answer", stream.getvalue())

    def test_data_samples_include_mirror_copy_with_fake_processor(self):
        from PIL import Image
        buffer = io.BytesIO()
        image = Image.new("RGB", (2, 1))
        image.putpixel((0, 0), (255, 0, 0))
        image.putpixel((1, 0), (0, 0, 255))
        image.save(buffer, "PNG")
        base = {
            "question": "Which side?\nOptions: A: first, B: second",
            "answer": "A", "images": [buffer.getvalue()],
            "noimage_logprobs": json.dumps({"A": math.log(.6), "B": math.log(.4)}),
        }
        rows = [dict(base, id=1, mirror_answer="B"),
                dict(base, id=2, answer="B", mirror_answer=None)]
        samples = train_data.make_train_samples(rows, mirror_aug=True)
        self.assertEqual(len(samples), 3)
        self.assertTrue(samples[1].flip)
        self.assertEqual(samples[1].label, "B")
        processor = _FakeProcessor()
        batch = train_data.build_inputs(processor, samples, 8)
        mirrored = [part["image"] for part in processor.conversations[1][0]["content"]
                    if part["type"] == "image"][0]
        self.assertEqual(mirrored.getpixel((0, 0)), (0, 0, 255))
        labels = batch["labels"].tolist() if hasattr(batch["labels"], "tolist") else batch["labels"]
        counts = (batch["n_options"].tolist() if hasattr(batch["n_options"], "tolist")
                  else batch["n_options"])
        self.assertEqual(labels, [0, 1, 1])
        self.assertEqual(counts, [2, 2, 2])
        self.assertTrue(math.isinf(float(batch["bias"][0][2])))
        self.assertEqual(train_data.validate_letter_tokens(processor), [1, 2, 3, 4])

    @unittest.skipUnless(TORCH_AVAILABLE, "torch not installed")
    def test_letter_logits_masks_invalid_options(self):
        import torch
        from train.losses import letter_logits
        logits = torch.arange(20, dtype=torch.float32).reshape(2, 10)
        selected = letter_logits(logits, [1, 3, 5, 7], [2, 4])
        self.assertEqual(selected[0, :2].tolist(), [1.0, 3.0])
        self.assertTrue(torch.isneginf(selected[0, 2:]).all())
        self.assertEqual(selected[1].tolist(), [11.0, 13.0, 15.0, 17.0])

    @unittest.skipUnless(TORCH_AVAILABLE, "torch not installed")
    def test_learned_mixin_matches_numpy_and_has_finite_gradients(self):
        import numpy as np
        import torch
        from train.losses import ce_loss, learned_mixin_loss

        z_np = np.array([[2.0, 0.0, -np.inf, -np.inf],
                         [.5, -.5, 1.0, -1.0]], dtype=np.float64)
        bias_np = np.array([[math.log(.7), math.log(.3), -np.inf, -np.inf],
                            [math.log(.1), math.log(.2), math.log(.3), math.log(.4)]])
        labels_np = np.array([0, 2])
        g_np = np.array([.8, 1.2])

        losses = []
        entropies = []
        for row, bias_row, label, scale in zip(z_np, bias_np, labels_np, g_np):
            valid = np.isfinite(row)
            combined = row[valid] + scale * bias_row[valid]
            combined -= np.log(np.exp(combined).sum())
            losses.append(-combined[label])
            scaled_bias = scale * bias_row[valid]
            probabilities = np.exp(scaled_bias - np.log(np.exp(scaled_bias).sum()))
            entropies.append(-(probabilities * np.log(probabilities)).sum())
        expected = np.mean(losses) + .1 * np.mean(entropies)

        z = torch.tensor(z_np, dtype=torch.float64, requires_grad=True)
        bias = torch.tensor(bias_np, dtype=torch.float64)
        labels = torch.tensor(labels_np)
        g = torch.tensor(g_np, dtype=torch.float64, requires_grad=True)
        loss, combined = learned_mixin_loss(z, bias, labels, g, .1)
        self.assertAlmostEqual(float(loss), float(expected), places=5)
        loss.backward()
        self.assertTrue(torch.isfinite(z.grad).all())
        self.assertTrue(torch.isfinite(g.grad).all())
        self.assertTrue(torch.isfinite(combined[torch.isfinite(z)]).all())

        zero_g = torch.zeros(2, dtype=torch.float64)
        lmh_zero, _ = learned_mixin_loss(z.detach(), bias, labels, zero_g, 0.0)
        self.assertAlmostEqual(float(lmh_zero), float(ce_loss(z.detach(), labels)), places=7)


if __name__ == "__main__":
    unittest.main()
