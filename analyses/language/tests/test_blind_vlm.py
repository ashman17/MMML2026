import pandas as pd

from blind_vlm import (
    build_tasks,
    circular_questions,
    make_prompt,
    map_logits_to_original,
    rotate_options,
)
from common import LETTERS


def test_rotation_and_mapping_are_inverses():
    opts = {"A": "alpha", "B": "bravo", "C": "charlie", "D": "delta"}
    displayed, mapping = rotate_options(opts, 1)
    assert displayed == {"A": "bravo", "B": "charlie", "C": "delta", "D": "alpha"}
    assert mapping == {"A": "B", "B": "C", "C": "D", "D": "A"}
    # Displayed D is highest; after shift 1 it maps back to original A.
    assert map_logits_to_original([0, 1, 2, 9], 1).argmax() == 0


def test_options_only_prompt_hides_stem():
    opts = {"A": "Left", "B": "Right", "C": "Front", "D": "Behind"}
    prompt = make_prompt("SECRET QUESTION", opts, options_only=True)
    assert "SECRET QUESTION" not in prompt
    assert "intentionally hidden" in prompt
    assert all(f"{letter}. {opts[letter]}" in prompt for letter in LETTERS)


def test_build_tasks_scores_five_prompts_per_question():
    frame = pd.DataFrame([{
        "id": 7,
        "category_en": "Pos: Cam-Obj",
        "split": "confirm",
        "answer": "B",
        "stem": "Where is the chair?",
        "opt_A": "Left",
        "opt_B": "Right",
        "opt_C": "Front",
        "opt_D": "Behind",
    }])
    tasks = build_tasks(frame)
    assert len(tasks) == 5
    assert {(task["condition"], task["shift"]) for task in tasks} == {
        ("original", 0), ("options_only", 0),
        ("circular", 1), ("circular", 2), ("circular", 3),
    }


def test_circular_consensus_maps_displayed_letters_back():
    rows = []
    gold_index = LETTERS.index("B")
    for shift in range(4):
        displayed_index = (gold_index - shift) % 4
        logits = [0.0] * 4
        logits[displayed_index] = 10.0
        pred_display = LETTERS[displayed_index]
        pred_original = LETTERS[(displayed_index + shift) % 4]
        rows.append({
            "id": 1,
            "category_en": "Pos: Cam-Obj",
            "split": "confirm",
            "gold_original": "B",
            "condition": "original" if shift == 0 else "circular",
            "shift": shift,
            "pred_original": pred_original,
            **{f"logit_{letter}": logits[i] for i, letter in enumerate(LETTERS)},
        })
    result = circular_questions(pd.DataFrame(rows)).iloc[0]
    assert result["consensus_prediction"] == "B"
    assert result["consensus_correct"]
    assert result["all_correct"]
    assert result["consistent"]
    assert result["rotation_accuracy"] == 1.0


def test_circular_rotation_uses_recorded_predictions_on_ties():
    """When displayed logits are tied on a nonzero shift, rotation metrics must
    use the recorded pred_original rather than the logit-recomputed choice."""
    # Tied logits: argmax always picks displayed A (index 0).
    # Recorded pred_original: A, B, C, D for shifts 0, 1, 2, 3.
    gold = "C"
    rows = []
    for shift in range(4):
        pred_display = "A"  # argmax of tied displayed logits is always A
        pred_original = LETTERS[(LETTERS.index(pred_display) + shift) % 4]
        rows.append({
            "id": 99,
            "category_en": "Pos: Cam-Obj",
            "split": "confirm",
            "gold_original": gold,
            "condition": "original" if shift == 0 else "circular",
            "shift": shift,
            "pred_original": pred_original,
            **{f"logit_{letter}": 5.0 for letter in LETTERS},
        })
    result = circular_questions(pd.DataFrame(rows)).iloc[0]
    # Only shift=2 gives pred_original="C" == gold
    assert result["rotation_accuracy"] == 0.25
    assert not result["consistent"]
    assert not result["all_correct"]
    # Mean-logit consensus: all equal → argmax 0 → original A ≠ C
    assert not result["consensus_correct"]
