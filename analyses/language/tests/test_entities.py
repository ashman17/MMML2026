import pytest

spacy = pytest.importorskip("spacy")

from entities import LVIS_PATH, load_lvis_names, mentions_from_doc  # noqa: E402

LVIS = load_lvis_names() if LVIS_PATH.exists() else set()


@pytest.fixture(scope="module")
def nlp():
    return spacy.load("en_core_web_sm", disable=["ner"])


def mentions(nlp, text):
    return {m["text"]: m for m in mentions_from_doc(nlp(text), LVIS)}


def test_image_anchored_object_and_skipped_chunks(nlp):
    got = mentions(nlp, "When you take the photo in Image 2, where is the green chair from Image 1 "
                        "located relative to you?")
    assert set(got) == {"the green chair"}
    chair = got["the green chair"]
    assert chair["type"] == "object" and chair["attribute"] and chair["image_anchored"]
    assert not chair["relational"]


def test_in_image_position_with_figure(nlp):
    got = mentions(nlp, "Which of the following is thicker: the round stone pillar on the right in Figure 2, "
                        "or the round stone pillar slightly to the left in Figure 1?")
    first = [m for m in mentions_from_doc(nlp("the round stone pillar on the right in Figure 2"), LVIS)][0]
    assert first["in_image_position"] and first["image_anchored"] and first["attribute"]
    assert all(m["type"] == "object" for m in got.values())


def test_functional_area(nlp):
    got = mentions(nlp, "When you took the second picture, where was the toothbrushing area in relation to you?")
    assert got["the toothbrushing area"]["type"] == "region_functional"
    assert got["the toothbrushing area"]["bare"]


def test_relational_object_and_coco_alias(nlp):
    got = mentions(nlp, "In what direction is the white ceramic stool next to the sofa in relation to "
                        "the bedside lamp?")
    stool = got["the white ceramic stool"]
    assert stool["relational"] and stool["attribute"] and not stool["image_anchored"]
    assert got["the sofa"]["in_coco"]
    assert "what direction" not in got


def test_quantities_parts_and_objects(nlp):
    got = mentions(nlp, "Which is smaller: the height of the black box in Figure 1, or the gap between "
                        "the top surface of the washbasin and the bottom of the wall cabinet in Figure 2?")
    assert got["the height"]["type"] == "quantity" and got["the gap"]["type"] == "quantity"
    assert got["the top surface"]["type"] == "part"
    assert got["the black box"]["type"] == "object"
    assert got["the wall cabinet"]["type"] == "object"


def test_room_and_area_with_relational_modifier(nlp):
    got = mentions(nlp, "In which direction is the pantry area with the white cabinet located relative to "
                        "the kitchen?")
    assert got["the pantry area"]["type"] == "region_functional" and got["the pantry area"]["relational"]
    assert got["the kitchen"]["type"] == "region_room"


def test_hanging_on_is_relational(nlp):
    got = mentions(nlp, "Where is the painting hanging on the wall relative to the bed?")
    assert got["the painting"]["relational"]


def test_parse_artifacts_and_compass_regions(nlp):
    got = mentions(nlp, "These two photos were taken in succession. In which direction is the camera rotating?")
    assert got == {"the camera rotating": got["the camera rotating"]}
    assert got["the camera rotating"]["type"] == "camera"
    got = mentions(nlp, "If the iMac is in the northern part of the room, where is the northwest corner?")
    assert got["the northern part"]["type"] == "region_generic"
    assert got["the northwest corner"]["type"] == "region_generic"


def test_structural_landmarks_are_ade20k_not_lvis(nlp):
    got = mentions(nlp, "Where is the door relative to the window and the laptop computer?")
    assert got["the door"]["in_ade20k"] and not got["the door"]["in_lvis"]
    assert got["the window"]["in_ade20k"] and not got["the window"]["in_lvis"]
    assert got["the laptop computer"]["in_lvis"]


@pytest.mark.skipif(not LVIS, reason="LVIS list not downloaded")
def test_lvis_names():
    assert len(LVIS) > 1203
    assert {"chair", "toothbrush", "washbasin", "sofa"} <= LVIS
