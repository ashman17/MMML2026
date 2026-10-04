"""Transparent spatial-language lexicon for MMSI-Bench questions.

Every tag is a regular expression, or a small rule over regex hits, applied to the
question stem and the four options separately. Patterns were written from the
explore half only; 07_audit.py measures agreement with manual labels on the
confirm half.
"""

from __future__ import annotations

import re
from typing import Mapping


def rx(pattern: str) -> re.Pattern:
    return re.compile(pattern, re.I)


# ---------------------------------------------------------------- image references
IMG_NOUN = r"(?:photo|picture|image|figure|frame|shot|photograph|diagram|chapter)s?"
ORD = r"(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|last|final)"
NUM = r"(?:\d+|one|two|three|four|five|six)"
ORD_VALUE = {"first": 1, "one": 1, "second": 2, "two": 2, "third": 3, "three": 3,
             "fourth": 4, "four": 4, "fifth": 5, "five": 5, "sixth": 6, "six": 6,
             "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10, "last": "last", "final": "last"}

IMG_INDEX = rx(rf"\b{IMG_NOUN}\s*(?P<n>{NUM})\b")
IMG_ORDINAL = rx(rf"\b(?P<n>{ORD})\s+(?:{IMG_NOUN}|one)\b")
IMG_SEQUENCE = rx(r"\b(?:continuous\w*|consecutive\w*|in (?:succession|sequence)|succession|"
                  r"chronological|one after another|series of)\b")

# ---------------------------------------------------------------- direction vocabularies
LEFT_RIGHT = rx(r"\b(?:left|right(?!-hand|\s+angle))\b")
FRONT_BACK = rx(r"\b(?:front|back|behind|rear\w*|forward|backward|ahead)\b")
UP_DOWN = rx(r"\b(?:up|down|upward|downward|upper|lower|above|below)\b")
EGO = rx(r"\b(?:left|right(?!-hand|\s+angle)|front|back|behind|rear\w*|forward|backward|ahead|"
         r"up|down|upward|downward|upper|lower|above|below|not moving|stationary|no movement|"
         r"same position)\b")
COMPASS = rx(r"\b(?:north|south)(?:east|west)?(?:ern)?\b|\b(?:east|west)(?:ern)?\b")
EAST_WEST = rx(r"\b(?:east|west|northeast|northwest|southeast|southwest)\b")
AXES = rx(r"coordinate system|right-hand|[+-][XYZ]\b|\b[XYZ][- ]?axis\b|\baxis\b|"
          r"positive (?:direction|angle)|negative (?:direction|angle)")
ROTATION_SENSE = rx(r"\b(?:clockwise|counterclockwise|counter-clockwise)\b")
ANGLE = rx(r"\b\d+\s*degrees?\b|\b(?:clockwise|counterclockwise|counter-clockwise|obtuse|acute|"
           r"right angle)\b")

# ---------------------------------------------------------------- answer-space cues
NUMERIC_OPTION = rx(r"^(?:\d+(?:\.\d+)?|zero|one|two|three|four|five|six|seven|eight|nine|ten)$")
ORDER_OPTION = rx(r"^(?:(?:first|second|third|fourth)(?:\s+(?:image|picture|photo|one))?"
                  r"(?:,\s*(?:then\s+)?|\s+then\s+)){1,3}(?:first|second|third|fourth)"
                  r"(?:\s+(?:image|picture|photo|one))?$")
ORDER_STEM = rx(r"\border in which\b|\bwhich (?:picture|photo|image) was taken first\b|"
                r"\bsequence of the (?:photos|pictures|images)\b|\bclockwise order\b")
STATEMENT_STEM = rx(r"which of the following (?:\w+ ){0,2}(?:statements?|descriptions?)\b|"
                    r"choose the correct description")
METRIC_STEM = rx(r"\bwhich (?:\w+ ){0,3}(?:is|has|was)\s+(?:a\s+)?(?:greater|larger|smaller|higher|"
                 r"lower|taller|shorter|longer|wider|narrower|thicker|bigger|closer|faster)\b|"
                 r"\b(?:greater|larger|smaller|higher|lower|taller|shorter|longer|wider|narrower|"
                 r"thicker|bigger)\b(?:\s+\w+){0,3}\s*[:,]|\bwhich is (?:higher|taller|longer|wider|"
                 r"narrower|thicker|smaller|greater|larger|lower|shorter)\b|\bestimate\b|"
                 r"\b(?:floor|total|surface) area\b|\barea covered\b|"
                 r"\bmeters?\b|\bcm\b|\bclosest in height\b|\bspeed\b|\bcloser\b|\bfarther\b|"
                 r"\bradius\b|\bdiameter\b|\bplaced directly into\b|\bget stuck\b")
ROUTE = rx(r"\b(?:go|going|goes|walk\w*|turn(?:ed|s)?|move forward|straight|keep going|"
           r"look(?:ing)?|cannot reach|cannot see)\b")
ABSTAIN = rx(r"cannot be determined|unable to determine|cannot (?:see|reach)|either is possible|"
             r"both are possible|all are possible|none of the above|both at the same time|"
             r"taken at the same time")
SOMETIMES = rx(r"^(?:sometimes\b|in most cases\b)")

ANSWER_SPACES = ("axis_sign", "order", "statement", "compass", "angle", "metric", "count",
                 "route", "ego_direction", "entity")

# ---------------------------------------------------------------- frame anchors (stem)
CAMERA_SELF = rx(
    r"\bwhen (?:you|i) (?:took|take|takes|was taking|were taking|am taking|are taking|saw|see)\b|"
    r"\bwhen (?:taking|shooting) (?:the |photo|picture|image)|"
    r"\b(?:at|in) the moment (?:of|when|the)\b|"
    r"\b(?:assum\w*|suppose|if)(?: that)? (?:i|you) (?:am|are|was|were) (?:taking|shooting)\b|"
    r"\bwhen (?:i|you) (?:am|are|was|were) at the (?:location|position)\b|"
    r"\bfrom the position in the (?:first|second|last) (?:image|photo|picture)\b|"
    r"\bmy current position\b|\bwhen i see the scene\b|"
    rf"\bwhen (?:the )?{ORD} {IMG_NOUN} (?:is|was) taken\b|"
    rf"\bin the {ORD} {IMG_NOUN}\b(?=[^?]*\b(?:you|me)\b)|"
    r"\bat the position and facing the direction shown\b")
SELF_REF = rx(r"\b(?:relative to|in relation to|compared to) (?:you|me|yourself|myself|him|her|them|"
              r"the person|the photographer)\b|\bon (?:your|my) (?:left|right)\b|"
              r"\b(?:behind|in front of) (?:you|me)\b|\bof you\b|\bto your\b")
AGENT = rx(
    r"\b(?:when|if|suppose|assum\w*|after|as|once)(?: that)?\s+(?:you|i|a person|the person|someone|"
    r"he|she|they|my friend|the homeowner)\s+(?:(?:are|am|is|were|was)\s+)?(?:\w+ly\s+)?"
    r"(?:sit\w*|lie|lies|lying|stand\w*|enter\w*|go|goes|going|walk\w*|come|comes|coming|fac\w*|"
    r"cook\w*|wash\w*|us(?:e|es|ing)|brush\w*|open\w*|look\w*|watch\w*|work\w*|play\w*|leav\w*|"
    r"step\w*|prepar\w*|dr(?:y|ies|ying)|finish\w*|want\w*|clos\w*|close to|next to|at the stair\w*)\b|"
    r"\b(?:a person|someone|the person|he|she)\s+(?:is\s+)?(?:sit\w*|lie|lying|stand\w*|enter\w*|"
    r"walk\w*|com\w*|fac\w*|watch\w*|play\w*)\b|"
    r"\b(?:entering|after entering|upon entering)\b|\benter the room\b|"
    r"\b(?:going|go|walk) (?:up|down)(?:stairs| the stairs)\b|\bdirection of going (?:up|down)stairs\b|"
    r"\bwith (?:my|your|his|her) back to\b|\b(?:i|you) (?:am|are) (?:lying|sitting|standing)\b|"
    r"\bfacing (?:north|south|east|west)\b(?![^,.?]*\b(?:photo|picture|image)\b)|"
    r"\bif (?:you|someone|a person) walks?\b|\bcomes? out of\b|\bcoming down\b")
FRONT_DEF = rx(r"\bas the (?:front|back|forward direction)\b|\b(?:is|be|being) considered "
               r"(?:as )?(?:the )?(?:front|forward|north|back)\b|\bconsidered as the front\b|"
               r"\bconsidered the front\b|\bis the front\b|\bfacing forward\b|\bfaces forward\b")
FRONT_BY_CAMERA = rx(r"\b(?:photo|picture|image|figure|shooting|taken|camera)\b")
OBJECT_REF = rx(r"\b(?:relative to|in relation to|compared to|from)\s+(?:the|this|that|a)\s+"
                r"(?!(?:first|second|third|fourth|fifth|sixth|last|final|current|same|opposite)\b)"
                r"(?!(?:photo|picture|image|figure|position|location|shooting|camera|photographer|"
                r"direction|perspective|platform)s?\b)[a-z]")
CAMERA_WORD = rx(r"\b(?:camera|cameras|you|your|i|me|my|photographer|viewpoint|perspective|"
                 r"first-person|dashcam)\b")
IMAGE_AS_PLACE = rx(rf"\b(?:position|location|place|spot)s? (?:of|where|in which) (?:the )?"
                    rf"(?:shooting )?(?:{ORD} )?{IMG_NOUN}|\b{IMG_NOUN}(?: \d)? (?:was|were|is) taken\b|"
                    rf"\bshooting (?:position|location)\b|\brelative to (?:the )?{ORD} {IMG_NOUN}")

# ---------------------------------------------------------------- conditions stated in language
CAP_OVERLAP = rx(r"\boverlap\w*\b")
CAP_SAME_PLACE = rx(r"\bsame (?:location|position|spot|place|scene|room|house|building|site|pile)\b|"
                    r"\bsimilar locations\b|\b\d+ degrees apart\b|\bafter rotating\b")
CAP_SHUFFLED = rx(r"\bshuffled\b|\bscrambled\b")
CAP_FIRST_PERSON = rx(r"\bfirst-person\b")
CAP_RIG = rx(r"\b(?:dashcam|in-vehicle|in-car|onboard|vehicle-mounted|head camera|wrist camera|"
             r"camera on the robot|camera mounted|mounted on a moving car)\b|\bcamera facing\b")
HEDGE = rx(r"\b(?:likely|approximately|approximate|roughly|probably|estimate|closest|"
           r"most accurately|obviously|mainly)\b")
CONDITIONAL = rx(r"\b(?:given that|assum\w*|suppose|if)\b")

# ---------------------------------------------------------------- operations (stem)
CAM_MOTION = rx(r"\b(?:camera|you|your viewpoint|viewpoint|i)\b[^.?]{0,25}?\b(?:mov\w*|rotat\w*|"
                r"turn\w*)\b|\bhow did (?:i|you) (?:most likely )?move\b|\bhow does the camera move\b|"
                r"\bhow (?:was|is) this car moving\b|\bvehicle with the camera\b")
OBJ_MOTION = rx(r"\b(?:is|are|did|does|do)\s+(?:the\s+)?(?!camera\b|you\b)[a-z][a-z ]{1,40}?\s"
                r"(?:mov\w*|rotat\w*|turn\w*|walk\w*)\b|\bhow (?:do|does|did|should) (?:the )?"
                r"(?!camera\b)[a-z][a-z ]{1,30}?(?:move|rotate|turn)\b|\bmotion state\b|"
                r"\bmoving vehicles\b|\bin motion\b|\bprevious action\b|\bdescribes (?:his|her|its) "
                r"movement\b")
DIRECTION_QUERY = rx(r"\bwhich direction\b|\bwhich side\b|\bwhere (?:is|was|are|were|would|can)\b|"
                     r"\bposition\b|\b(?:to|on) the (?:left|right|north|south|east|west|"
                     r"northeast|northwest|southeast|southwest)\b|\bin front of\b|\bbehind\b|"
                     r"\bdirectly (?:in front|behind)\b|\bwhich wall\b|\bpositional relationship\b")
COUNT_STEM = rx(r"\bhow many\b|\bnumber of\b")
CROSS_VIEW = rx(r"\boverlap\w*\b|\bsame (?:location|position|spot|place|scene|room|house|building|"
                r"site|pile|object|area)\b|\bin total\b|\bdifferent (?:\w+ ){0,3}(?:appear|can|in|on|"
                r"captured|displaying)|\btotal\b|\bthese (?:two|three|six) (?:photos|pictures|images)\b")
TEXT_OR_CHIRAL = rx(r"\bletter\b|\blabeled\b|\bwritten\b|\bwriting\b|\bsigns?\b|\buppercase\b|"
                    r"\bcapital\b|\b[A-Z]-shape\b|\bclock\b|\"[A-Z]+\"")
HANDEDNESS = rx(r"\b(?:left|right) (?:gripper|hand|wrist|arm|foot|leg)\b|\bgrippers\b|\bright-hand\b|"
                r"\bin the left hand\b")
TRAFFIC = rx(r"\b(?:lane|driving|dashcam|car|cars|vehicle|vehicles|road|crosswalk|traffic|"
             r"toyota)\b")
MIRROR_SCENE = rx(r"\breflect\w*\b|\bin the mirror\b|\bis a mirror\b|\bmirror with\b")
TEMPORAL_REF = rx(rf"\b(?:last|final|next|previous) {IMG_NOUN}\b|\bmoment\b|\bfrom (?:taking )?"
                  rf"(?:the )?(?:{IMG_NOUN} ?1|first {IMG_NOUN}) to\b|\bwhen (?:i|you) went from\b|"
                  r"\bbefore\b|\bafter\b(?! (?:you|a person|entering|leaving))")


# ---------------------------------------------------------------- helpers
def _frac(pattern: re.Pattern, texts: list[str]) -> float:
    return sum(bool(pattern.search(t)) for t in texts) / len(texts)


def _ord_value(token: str):
    token = token.lower()
    return int(token) if token.isdigit() else ORD_VALUE.get(token)


def image_refs(text: str) -> list[tuple[int, object]]:
    """All image references in order of appearance as (char_offset, index or 'last')."""
    refs = [(m.start(), _ord_value(m.group("n"))) for m in IMG_INDEX.finditer(text)]
    refs += [(m.start(), _ord_value(m.group("n"))) for m in IMG_ORDINAL.finditer(text)]
    return sorted(r for r in refs if r[1] is not None)


def _first_ref_after(text: str, start: int, window: int = 80):
    segment = text[start:start + window]
    stop = re.search(r"[?]", segment)
    segment = segment[:stop.start()] if stop else segment
    refs = image_refs(segment)
    return refs[0][1] if refs else None


def anchor_image(stem: str):
    """Image whose camera defines 'me/you' or the reference viewpoint, or None.

    Priority: an explicit self-at-camera trigger, then a front definition that
    names an image, then 'relative to / from <image position>'.
    """
    for m in CAMERA_SELF.finditer(stem):
        refs_inside = image_refs(m.group(0))
        if refs_inside:
            return refs_inside[0][1]
        ref = _first_ref_after(stem, m.end())
        if ref is not None:
            return ref
    for clause in re.split(r"[,;()]|(?<=[.?])\s", stem):
        if FRONT_DEF.search(clause) and FRONT_BY_CAMERA.search(clause):
            refs = image_refs(clause)
            if refs:
                return refs[0][1]
    for m in re.finditer(r"\b(?:relative to|from|in relation to|from the perspective of)\b", stem, re.I):
        ref = _first_ref_after(stem, m.end(), window=60)
        if ref is not None:
            return ref
    return None


def answer_space(stem: str, opts: Mapping[str, str]) -> str:
    texts = list(opts.values())
    if _frac(AXES, texts) >= 0.5:
        return "axis_sign"
    if ORDER_STEM.search(stem) or sum(bool(ORDER_OPTION.match(t)) for t in texts) >= 2:
        return "order"
    if STATEMENT_STEM.search(stem):
        return "statement"
    if _frac(COMPASS, texts) >= 0.5:
        return "compass"
    if _frac(ANGLE, texts) >= 0.5:
        return "angle"
    if METRIC_STEM.search(stem) or any(SOMETIMES.search(t) for t in texts):
        return "metric"
    if sum(bool(NUMERIC_OPTION.match(t.strip())) for t in texts) >= 3 or COUNT_STEM.search(stem):
        return "count"
    if _frac(ROUTE, texts) >= 0.5:
        return "route"
    if _frac(EGO, texts) >= 0.5:
        return "ego_direction"
    return "entity"


def mirror_level(stem: str, opts: Mapping[str, str]) -> tuple[str, list[str]]:
    """Validity of horizontal mirroring for Idea 3.

    unsafe:          handedness or reading cannot be fixed by swapping words
    rewrite_stem:    the stem itself must be rewritten (left/right, east/west, clockwise)
    swap_options:    only the option texts need a left/right swap
    no_change:       no handed language anywhere
    """
    text = stem + " " + " ".join(opts.values())
    unsafe = [name for name, pat in (("axes", AXES), ("text_or_chiral", TEXT_OR_CHIRAL),
                                     ("handedness", HANDEDNESS), ("traffic", TRAFFIC),
                                     ("mirror_in_scene", MIRROR_SCENE)) if pat.search(text)]
    if unsafe:
        return "unsafe", unsafe
    rewrite = [name for name, pat, where in (("left_right_in_stem", LEFT_RIGHT, stem),
                                             ("east_west", EAST_WEST, text),
                                             ("rotation_sense", ROTATION_SENSE, text))
               if pat.search(where)]
    if rewrite:
        return "rewrite_stem", rewrite
    if any(LEFT_RIGHT.search(t) for t in opts.values()):
        return "swap_options", ["left_right_in_options"]
    return "no_change", []


def reorder_level(stem: str, opts: Mapping[str, str], space: str, motion: bool) -> tuple[str, list[str]]:
    """Validity of presenting images in a different order while keeping their labels (Idea 3).

    unsafe:           meaning depends on temporal order (motion, 'last image', order questions)
    label_dependent:  ordinals like 'the first photo' must be read as labels, not positions
    safe:             only explicit numeric labels, or no image references
    """
    if CAP_SHUFFLED.search(stem):
        return "safe", ["order_declared_shuffled"]
    reasons = []
    if space == "order":
        reasons.append("order_question")
    if motion:
        reasons.append("motion")
    if TEMPORAL_REF.search(stem) or any(TEMPORAL_REF.search(t) for t in opts.values()):
        reasons.append("temporal_reference")
    if reasons:
        return "unsafe", reasons
    if IMG_ORDINAL.search(stem) or any(IMG_ORDINAL.search(t) for t in opts.values()):
        return "label_dependent", ["ordinal_image_reference"]
    if IMG_SEQUENCE.search(stem):
        return "label_dependent", ["sequence_stated"]
    return "safe", []


def tag_question(stem: str, opts: Mapping[str, str], n_images: int) -> dict:
    texts = list(opts.values())
    space = answer_space(stem, opts)

    axes_stem = bool(AXES.search(stem))
    compass_stem = bool(COMPASS.search(stem))
    compass_options = _frac(COMPASS, texts) >= 0.5
    agent = bool(AGENT.search(stem))
    camera_self = bool(CAMERA_SELF.search(stem)) or (bool(SELF_REF.search(stem)) and bool(image_refs(stem)))
    front_clauses = [c for c in re.split(r"[,;()]|(?<=[.?])\s", stem) if FRONT_DEF.search(c)]
    front_by_camera = any(FRONT_BY_CAMERA.search(c) for c in front_clauses)
    front_by_object = bool(front_clauses) and not front_by_camera
    object_ref = bool(OBJECT_REF.search(stem))
    directional = space in ("ego_direction", "angle", "route")

    if axes_stem or space == "axis_sign":
        frame = "axes"
    elif compass_stem or compass_options:
        frame = "compass"
    elif agent:
        frame = "agent"
    elif front_by_object:
        frame = "object_front"
    elif camera_self or front_by_camera or (directional and (CAMERA_WORD.search(stem)
                                                              or IMAGE_AS_PLACE.search(stem))):
        frame = "camera"
    elif directional and object_ref:
        frame = "object_unspecified"
    elif directional or space in ("entity", "statement") and DIRECTION_QUERY.search(stem):
        frame = "viewer_implicit"
    else:
        frame = "none"

    anchor = anchor_image(stem) if frame in ("camera", "axes", "compass", "agent") else None
    anchor_index = n_images if anchor == "last" else anchor

    sentences = [s for s in re.split(r"(?<=[.?!])\s+", stem) if s.strip()]
    n_declarative = sum(not s.rstrip().endswith("?") for s in sentences)
    n_parenthetical = len(re.findall(r"\([^)]*\)", stem))

    cam_motion = bool(CAM_MOTION.search(stem)) and space in ("ego_direction", "route", "statement", "entity", "angle")
    obj_motion = bool(OBJ_MOTION.search(stem)) and not cam_motion
    distinct_refs = {r[1] for r in image_refs(stem)}

    mirror, mirror_why = mirror_level(stem, opts)
    reorder, reorder_why = reorder_level(stem, opts, space, cam_motion or obj_motion)

    return {
        "answer_space": space,
        "frame": frame,
        "anchor_image": anchor_index,
        "anchor_is_first": (anchor_index == 1) if anchor_index is not None else None,
        "f_axes": axes_stem or space == "axis_sign",
        "f_compass_stem": compass_stem,
        "f_compass_options": compass_options,
        "f_agent": agent,
        "f_camera_self": camera_self,
        "f_front_by_camera": front_by_camera,
        "f_front_by_object": front_by_object,
        "f_object_ref": object_ref,
        "n_declarative": n_declarative,
        "n_parenthetical": n_parenthetical,
        "has_premise": n_declarative > 0 or n_parenthetical > 0 or bool(CONDITIONAL.search(stem)),
        "img_index": bool(IMG_INDEX.search(stem)),
        "img_ordinal": bool(IMG_ORDINAL.search(stem)),
        "img_sequence": bool(IMG_SEQUENCE.search(stem)),
        "img_none": not (IMG_INDEX.search(stem) or IMG_ORDINAL.search(stem) or IMG_SEQUENCE.search(stem)),
        "cap_overlap": bool(CAP_OVERLAP.search(stem)),
        "cap_same_place": bool(CAP_SAME_PLACE.search(stem)),
        "cap_first_person": bool(CAP_FIRST_PERSON.search(stem)),
        "cap_rig": bool(CAP_RIG.search(stem)),
        "hedge": bool(HEDGE.search(stem)),
        "abstain_option": any(ABSTAIN.search(t) for t in texts),
        "lr_in_stem": bool(LEFT_RIGHT.search(stem)),
        "lr_in_options": any(LEFT_RIGHT.search(t) for t in texts),
        "fb_in_options": any(FRONT_BACK.search(t) for t in texts),
        "ud_in_options": any(UP_DOWN.search(t) for t in texts),
        "op_direction": space in ("ego_direction", "compass", "angle", "route")
                        or (space in ("entity", "statement") and bool(DIRECTION_QUERY.search(stem))),
        "op_metric": space == "metric",
        "op_count": space == "count" or bool(COUNT_STEM.search(stem)),
        "op_perspective_taking": agent or front_by_object,
        "op_absolute_frame": axes_stem or compass_stem or compass_options or space == "axis_sign",
        "op_cross_view_explicit": bool(CROSS_VIEW.search(stem)) or len(distinct_refs) >= 2,
        "op_premise_chain": (n_declarative > 0 or n_parenthetical > 0 or bool(CONDITIONAL.search(stem)))
                            and (space in ("ego_direction", "compass", "angle", "route", "statement")),
        "op_camera_motion": cam_motion,
        "op_object_motion": obj_motion,
        "op_temporal_order": space == "order",
        "op_route": space == "route",
        "op_statement_check": space == "statement",
        "mirror_level": mirror,
        "mirror_reasons": ";".join(mirror_why),
        "reorder_level": reorder,
        "reorder_reasons": ";".join(reorder_why),
    }
