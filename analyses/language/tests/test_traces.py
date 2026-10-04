import pytest

spacy = pytest.importorskip("spacy")

from traces import bridging_lemmas, content_lemmas, cue_flags  # noqa: E402


@pytest.fixture(scope="module")
def nlp():
    return spacy.load("en_core_web_sm", disable=["ner"])


def test_conclusion_sequence_and_numbered_cues():
    flags = cue_flags("1. The pen is at the chin.\n2. The pen is at the shoulder.\n"
                      "So the pen is moving toward the girl.")
    assert flags["numbered_steps"]
    assert flags["conclusion"]
    assert flags["motion"]
    assert not flags["perspective"]


def test_cross_view_and_perspective_cues():
    flags = cue_flags("When you sit on the chair, Figure 3 shows the lamp further to your left.")
    assert flags["cross_view"]
    assert flags["perspective"]
    assert flags["comparison"]


def test_bridging_noun_is_absent_from_the_question(nlp):
    question = nlp("Where is the chair relative to the table?")
    trace = nlp("The window is behind the chair, so the chair faces the table.")
    assert "window" in bridging_lemmas(trace, question)
    assert "chair" not in bridging_lemmas(trace, question)
    assert "table" not in bridging_lemmas(trace, question)


def test_image_references_and_directions_are_not_content_nouns(nlp):
    lemmas = content_lemmas(nlp("Figure 2 is on your left relative to the door."))
    assert "door" in lemmas
    assert "figure" not in lemmas
    assert "left" not in lemmas
