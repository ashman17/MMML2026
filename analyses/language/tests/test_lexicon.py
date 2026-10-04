import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lexicon import LEFT_RIGHT, tag_question  # noqa: E402

EGO4 = {"A": "Behind", "B": "Right", "C": "Left", "D": "In front"}
DIAG4 = {"A": "Back left", "B": "Directly behind", "C": "Directly to the left", "D": "Front left"}
COMPASS4 = {"A": "Southwest", "B": "Northwest", "C": "Northeast", "D": "Southeast"}
ROT4 = {"A": "Up", "B": "Down", "C": "Left", "D": "Right"}


def tag(stem, opts, n_images=2):
    return tag_question(stem, opts, n_images)


def test_right_handed_is_not_a_direction():
    assert LEFT_RIGHT.search("which is a right-handed system") is None
    assert LEFT_RIGHT.search("Obtuse angle | Right angle") is None
    assert LEFT_RIGHT.search("Turn right") is not None


def test_self_anchored_at_second_camera():
    t = tag("Assuming I am taking the second photo, where was the camera positioned relative to me "
            "when the first photo was taken?", EGO4)
    assert (t["answer_space"], t["frame"], t["anchor_image"], t["anchor_is_first"]) == \
        ("ego_direction", "camera", 2, False)


def test_last_image_resolves_to_image_count_and_blocks_reordering():
    t = tag("The images are taken continuously from a first-person perspective. At the moment of the "
            "last image, in which direction is the white table relative to you?",
            {"A": "Left", "B": "Right", "C": "Front", "D": "Back"}, n_images=3)
    assert t["frame"] == "camera" and t["anchor_image"] == 3
    assert t["img_sequence"] and t["cap_first_person"]
    assert t["reorder_level"] == "unsafe"


@pytest.mark.parametrize("stem, expected", [
    ("Continuous shooting from a first-person perspective, with the direction in photo 2 as the front, "
     "what is the position of the first photo relative to the position where the second photo was taken?", 2),
    ("What is the position of the second photo relative to the position where the first photo was taken?", 1),
    ("In which direction from the location where the second photo was taken is the location where the "
     "first photo was taken?", 2),
    ("In the series of photos taken from a first-person perspective, where is the light switch relative "
     "to you in the last photo?", "last"),
])
def test_camera_camera_reference_image(stem, expected):
    t = tag(stem, {"A": "Rear left", "B": "Front left", "C": "Rear right", "D": "Front right"}, n_images=4)
    assert t["frame"] == "camera"
    assert t["anchor_image"] == (4 if expected == "last" else expected)


def test_axis_convention_is_absolute_and_mirror_unsafe():
    t = tag("The camera coordinate system is defined as +Y up, -Z forward, which is a right-handed system. "
            "How can the viewpoint in the second image be obtained from the viewpoint in the first image in "
            "its corresponding camera coordinate system?",
            {"A": "Rotate by a negative angle around the Z axis", "B": "Rotate by a negative angle around the Y axis",
             "C": "Rotate by a positive angle around the Z axis", "D": "Rotate by a positive angle around the Y axis"})
    assert (t["answer_space"], t["frame"], t["mirror_level"]) == ("axis_sign", "axes", "unsafe")
    assert t["op_absolute_frame"]


def test_hypothetical_agent_with_compass_premise():
    t = tag("A person enters the room with the fire extinguisher from the room with the piano, facing north. "
            "In which direction is the wooden jar in Figure 2 relative to the fire extinguisher?", COMPASS4)
    assert (t["answer_space"], t["frame"]) == ("compass", "compass")
    assert t["f_agent"] and t["op_perspective_taking"] and t["op_absolute_frame"] and t["has_premise"]
    assert t["mirror_level"] == "rewrite_stem"


def test_activity_defines_the_viewpoint():
    t = tag("When you are cooking in the kitchen area, in which direction is the dining area shown in "
            "Figure 1 relative to you?", DIAG4)
    assert t["frame"] == "agent"
    assert t["mirror_level"] == "swap_options"
    assert t["reorder_level"] == "safe"


def test_object_defined_front():
    t = tag("Taking the direction of the faucet outlet as the front, where is the sofa located?",
            {"A": "Front right", "B": "Front left", "C": "Rear right", "D": "Rear left"})
    assert t["frame"] == "object_front" and t["op_perspective_taking"]


def test_relative_direction_without_a_stated_frame():
    t = tag("In which direction is the bedroom relative to the sink?",
            {"A": "Left", "B": "Right", "C": "Behind", "D": "In front"})
    assert t["frame"] == "object_unspecified"


def test_metric_with_sometimes_distractor():
    t = tag("Which is higher: the top horizontal level of the white bath towel or the top horizontal level "
            "of the gold switch?",
            {"A": "Sometimes the gold switch is higher, sometimes the towel", "B": "The gold switch is higher",
             "C": "They are the same height", "D": "The white bath towel is higher"})
    assert t["answer_space"] == "metric" and t["frame"] == "none"


def test_region_names_are_not_measurements():
    t = tag("When I took the second photo, in which direction was the sleeping area located relative to me?",
            {"A": "Front", "B": "Back", "C": "Right", "D": "Left"})
    assert t["answer_space"] == "ego_direction" and t["frame"] == "camera" and t["anchor_image"] == 2
    t = tag("Assuming the long side of the sofa in Figure 1 is 2 meters, please estimate the floor area of "
            "the discussion area.", {"A": "7", "B": "10", "C": "13", "D": "4"})
    assert t["answer_space"] == "metric"


def test_count_across_views():
    t = tag("How many windows are there in total on the first floor of the house shown in these two pictures?",
            {"A": "6", "B": "5", "C": "4", "D": "3"})
    assert t["answer_space"] == "count" and t["op_cross_view_explicit"]


def test_temporal_order_question():
    t = tag("The car is moving forward. What is the order in which the pictures were taken?",
            {"A": "First, third, second", "B": "Second, first, third", "C": "Third, second, first",
             "D": "Second, third, first"}, n_images=3)
    assert t["answer_space"] == "order" and t["reorder_level"] == "unsafe"


def test_camera_rotation():
    t = tag("The images are taken continuously from a first-person perspective. In which direction is the "
            "camera rotating?", ROT4)
    assert t["op_camera_motion"] and not t["op_object_motion"]
    assert t["frame"] == "camera" and t["reorder_level"] == "unsafe"


def test_object_motion():
    t = tag("The pictures were taken in succession. Did the triangular wooden board move?",
            {"A": "Did not move", "B": "Moved to the left", "C": "Moved upward", "D": "Moved to the right"})
    assert t["op_object_motion"] and not t["op_camera_motion"]


def test_declared_shuffle_makes_reordering_safe():
    t = tag("These six photos were taken at the same location, each shot after rotating 60 degrees, and the "
            "order of the six photos has been shuffled. Assuming I am sitting on the chair shown in photo 5, "
            "which photo is most likely to show what I would see?",
            {"A": "Photo 2", "B": "Photo 3", "C": "Photo 1", "D": "Photo 4"}, n_images=6)
    assert t["reorder_level"] == "safe" and t["cap_same_place"]


def test_route_from_a_camera_position():
    t = tag("When I am at the location shown in the second picture, I want to go to the restroom. "
            "Which way should I go?",
            {"A": "Turn right and go straight", "B": "Go forward", "C": "Turn around and go straight",
             "D": "Turn left and go straight"})
    assert t["answer_space"] == "route" and t["frame"] == "camera" and t["anchor_image"] == 2


def test_statement_verification():
    t = tag("The picture is taken in first-person perspective in a continuous sequence, with the head of the "
            "bed on the north side of the bedroom. Which of the following statements is correct?",
            {"A": "There is no door with a green exit sign on the north side", "B": "There are two tables",
             "C": "There are a total of 5 porcelain vases", "D": "There is a door on the east side"})
    assert t["answer_space"] == "statement" and t["frame"] == "compass"
