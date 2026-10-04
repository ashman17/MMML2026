import importlib.util
import pathlib
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


if __name__ == "__main__":
    unittest.main()
