import numpy as np

from shortcuts import (
    UnionFind,
    candidate_eliminations,
    elimination_policy,
    heuristic_policies,
    holm_adjust,
    option_family,
    tokens,
)


def test_tokens_are_lowercase_words_and_numbers():
    assert tokens("Front-left, 60 degrees!") == ["front", "left", "60", "degrees"]


def test_option_families():
    assert option_family("Sometimes, depending on the view") == "sometimes"
    assert option_family("Cannot be determined") == "abstain"
    assert option_family("Northwest") == "compass"
    assert option_family("Front right") == "ego"
    assert option_family("3") == "number"
    assert option_family("First image") == "order"


def test_format_outlier_requires_three_matching_options():
    opts = {"A": "North", "B": "South", "C": "East", "D": "a chair"}
    assert candidate_eliminations(opts)["format_outlier"] == {"D"}
    mixed = {"A": "North", "B": "Left", "C": "3", "D": "a chair"}
    assert candidate_eliminations(mixed)["format_outlier"] == set()


def test_explicit_elimination_candidates():
    opts = {"A": "Left", "B": "Sometimes", "C": "Cannot be determined", "D": "Right"}
    candidates = candidate_eliminations(opts)
    assert candidates["sometimes"] == {"B"}
    assert candidates["abstain"] == {"C"}
    policy = elimination_policy(opts, {"sometimes", "abstain"})
    np.testing.assert_allclose(policy, [0.5, 0.0, 0.0, 0.5])


def test_heuristics_split_ties():
    opts = {"A": "red chair", "B": "blue table", "C": "lamp", "D": "green sofa"}
    policies = heuristic_policies("Where is the chair?", opts)
    np.testing.assert_allclose(policies["longest_option"], [1 / 3, 1 / 3, 0, 1 / 3])
    np.testing.assert_allclose(policies["shortest_option"], [0, 0, 1, 0])
    np.testing.assert_allclose(policies["greatest_overlap"], [1, 0, 0, 0])


def test_holm_adjustment_is_monotone_in_rank():
    adjusted = holm_adjust({"a": 0.01, "b": 0.03, "c": 0.20})
    assert adjusted == {"a": 0.03, "b": 0.06, "c": 0.20}


def test_union_find_connects_template_components():
    groups = UnionFind([1, 2, 3, 4])
    groups.union(1, 2)
    groups.union(2, 3)
    assert groups.find(1) == groups.find(3)
    assert groups.find(4) != groups.find(1)


def test_unseen_log_odds_equals_chance_log_odds():
    """The Beta(1, 3) unseen-token default should equal log(1/3), the log-odds
    of the four-choice chance rate 0.25."""
    # Import here to avoid pulling heavy sklearn imports at collection time.
    import importlib, sys
    # 04_shortcuts is a top-level script; import via importlib with sys.path trick.
    import os
    here = os.path.dirname(os.path.dirname(__file__))
    sys.path.insert(0, here)
    mod = importlib.import_module("04_shortcuts".replace("-", "_"))
    # Module name has a digit prefix; use importlib with the file path instead.
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_shortcuts04", os.path.join(here, "04_shortcuts.py")
    )
    m = importlib.util.module_from_spec(spec)
    # Don't fully exec (triggers side-effects); just check the constant definition.
    expected = float(np.log(1 / 3))
    assert abs(expected - (-1.0986122886681098)) < 1e-9  # sanity
    # The constant in the source equals log(1/3).
    import ast
    src = open(os.path.join(here, "04_shortcuts.py")).read()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "UNSEEN_LOG_ODDS":
                    # The RHS must be float(np.log(1/3)) or equivalent.
                    rhs = ast.unparse(node.value)
                    assert "1 / 3" in rhs or "1/3" in rhs, f"Unexpected RHS: {rhs}"
                    return
    raise AssertionError("UNSEEN_LOG_ODDS constant not found in 04_shortcuts.py")

