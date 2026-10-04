import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import TEXT_CACHE, load_text, parse_question  # noqa: E402


def test_comma_separated_options():
    stem, opts = parse_question(
        "How is the perspective transformed?\nOptions: A: Turn right, B: Turn down, C: Turn up, D: Turn left")
    assert stem == "How is the perspective transformed?"
    assert opts == {"A": "Turn right", "B": "Turn down", "C": "Turn up", "D": "Turn left"}


def test_commas_inside_an_option_are_kept():
    _, opts = parse_question(
        "In which direction is the race car moving?\nOptions: A: Forward, B: Front left, "
        "C: First front left, then front right, D: Front right")
    assert opts["C"] == "First front left, then front right"


def test_sentence_options_with_periods():
    _, opts = parse_question(
        "Which of the following statements is correct?\nOptions: A: There is no door. "
        "B: There are two tables, the larger one is southeast. C: There are 5 vases. D: The door is east.")
    assert opts["B"] == "There are two tables, the larger one is southeast"
    assert opts["D"] == "The door is east"


@pytest.mark.parametrize("bad", [
    "No options here?",
    "Q?\nOptions: A: x, B: y, C: z",
    "Q?\nOptions: A: x, C: y, B: z, D: w",
])
def test_malformed_questions_raise(bad):
    with pytest.raises(ValueError):
        parse_question(bad)


@pytest.mark.skipif(not TEXT_CACHE.exists(), reason="run 00_build_dataset.py first")
def test_known_items_in_cache():
    df = load_text().set_index("id")
    assert df.loc[668, "opt_C"] == "First front left, then front right"
    assert df.loc[17, "stem"].startswith("The camera coordinate system is defined as +Y up")
    assert df.loc[531, "opt_B"] == "Right"
    assert df[[f"opt_{k}" for k in "ABCD"]].map(len).min().min() > 0
