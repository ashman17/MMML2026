import pytest

spacy = pytest.importorskip("spacy")

from additions import (  # noqa: E402
    added_comparatives,
    added_reasoning,
    added_spatial_pairs,
    added_viewpoint,
    spatial_pairs,
)
from traces import STEP_CUES  # noqa: E402


@pytest.fixture(scope="module")
def nlp():
    return spacy.load("en_core_web_sm", disable=["ner"])


def test_step_f_presence_cues_are_unchanged():
    assert STEP_CUES["comparison"].search("between")
    assert STEP_CUES["comparison"].search("closer")


def test_same_landmark_relation_is_not_an_addition(nlp):
    question = nlp("The lamp is behind the chair. Where is the lamp?")
    trace = nlp("The lamp is behind the chair.")
    assert added_spatial_pairs(trace, question) == set()


def test_bare_direction_option_does_not_block_a_landmark(nlp):
    question = nlp("Where is the lamp relative to the chair? Directly behind")
    trace = nlp("The lamp is behind the chair.")
    assert ("behind", "chair") in added_spatial_pairs(trace, question)


def test_behind_you_is_not_a_landmark(nlp):
    trace = nlp("The toilet is behind you to your left.")
    assert spatial_pairs(trace) == set()


def test_between_keeps_both_landmarks(nlp):
    question = nlp("How is the camera rotating?")
    trace = nlp("The camera moves between the robotic arm and the gray bag.")
    pairs = added_spatial_pairs(trace, question)
    assert ("between", "arm") in pairs, pairs
    assert ("between", "bag") in pairs, pairs


def test_in_front_of_and_next_to(nlp):
    question = nlp("Where is the bed?")
    trace = nlp("The stool is next to the sofa, and the table is in front of the bed.")
    pairs = added_spatial_pairs(trace, question)
    assert ("next to", "sofa") in pairs, pairs
    assert ("in front of", "bed") in pairs, pairs


def test_repeated_closer_is_not_an_addition(nlp):
    question = nlp("Which object is closer to the door?")
    trace = nlp("The lamp is closer to the door than the chair.")
    assert added_comparatives(trace.text, question.text) == set()


def test_a_new_comparative_still_counts(nlp):
    question = nlp("Which object is closer to the door?")
    trace = nlp("The lamp is higher than the statue.")
    assert added_comparatives(trace.text, question.text) == {"higher"}


def test_between_is_not_a_comparison(nlp):
    question = nlp("Where is the cup?")
    trace = nlp("The cup is between the box and the bag.")
    assert added_comparatives(trace.text, question.text) == set()
    pairs = added_spatial_pairs(trace, question)
    assert ("between", "box") in pairs and ("between", "bag") in pairs, pairs


def test_figure_mention_is_not_a_viewpoint_change(nlp):
    question = nlp("Where is the chair relative to the table?")
    trace = nlp("The chair in Figure 2 is next to the window.")
    viewpoint = added_viewpoint(trace, question, anchor=None)
    assert not viewpoint["added"]
    assert ("next to", "window") in added_spatial_pairs(trace, question)


def test_comparing_with_another_photo_is_a_viewpoint_change(nlp):
    question = nlp("When you took the second photo, where was the toilet in relation to you?")
    trace = nlp("By comparing with photo 1, the toilet is behind the sink.")
    viewpoint = added_viewpoint(trace, question, anchor=2)
    assert viewpoint["images"] == {1}
    assert viewpoint["added"]


def test_new_facing_complement(nlp):
    question = nlp("When you take photo 2, where is the green plant relative to you?")
    trace = nlp("You are now facing the right armrest, so the plant is behind you.")
    viewpoint = added_viewpoint(trace, question, anchor=2)
    assert "armrest" in viewpoint["complements"]
    assert viewpoint["added"]
    assert added_spatial_pairs(trace, question) == set()


def test_repeating_both_stated_cameras_is_not_a_viewpoint_change(nlp):
    question = nlp("How is the viewpoint in the second image obtained from the viewpoint in the first image?")
    trace = nlp("The viewpoint in the second image is behind the viewpoint in the first image.")
    viewpoint = added_viewpoint(trace, question, anchor=None)
    assert not viewpoint["added"], viewpoint
    assert ("behind", "viewpoint") in added_spatial_pairs(trace, question)


def test_numbered_conclusion_is_not_a_reasoning_step(nlp):
    question = nlp("Where is the pen relative to the girl?")
    trace = nlp("1. The pen is at the chin.\n2. So the pen is therefore behind.")
    assert added_reasoning(trace, question) == set()


def test_identity_claim_is_a_reasoning_step(nlp):
    question = nlp("Where is the cabinet relative to the bathroom door?")
    trace = nlp("The cabinet in Figure 2 is not the same as the one in the bathroom.")
    assert "not the same" in added_reasoning(trace, question)


def test_route_consequence_needs_a_verb_and_a_noun(nlp):
    question = nlp("How should I move toward the cabinet?")
    trace = nlp("You cannot walk straight to the cabinet.")
    cues = added_reasoning(trace, question)
    assert "cannot" in cues and "walk" in cues


def test_walking_area_is_not_a_route(nlp):
    question = nlp("Where is the bed?")
    trace = nlp("The walking area is next to the bed.")
    assert added_reasoning(trace, question) == set()
    assert ("next to", "bed") in added_spatial_pairs(trace, question)
