"""Entity-mention typing on spaCy noun chunks (used by 02_entities.py and tests).

A mention is a noun chunk whose head is not a pronoun, an image reference, a
direction word or a query word ("direction", "relation", ...). Each mention gets
one type and a set of descriptor modes:

  type   object | region_room ("kitchen") | region_functional ("toothbrushing
         area") | region_generic ("the area") | part ("the top surface") |
         camera | quantity ("the height", excluded from entity counts)
  modes  attribute ("green", "wooden"), relational ("next to the sofa",
         "hanging on the wall"), image_anchored ("from Image 1"),
         in_image_position ("on the right in Figure 2"), bare (none of these)
"""

from __future__ import annotations

import ast
import re

from common import CACHE

LVIS_PATH = CACHE / "lvis_v1_categories.py"
LVIS_URL = ("https://raw.githubusercontent.com/facebookresearch/detectron2/main/"
            "detectron2/data/datasets/lvis_v1_categories.py")

IMAGE_NOUNS = {"photo", "picture", "image", "figure", "frame", "shot", "photograph", "diagram"}
SKIP_HEADS = {"direction", "relation", "relative", "position", "location", "following", "way", "order",
              "perspective", "viewpoint", "moment", "time", "sequence", "series", "degree", "angle",
              "option", "description", "statement", "route", "scene", "view", "shooting", "lot",
              "total", "number", "one", "same", "case", "left", "right", "front", "back", "rear",
              "north", "south", "east", "west", "northeast", "northwest", "southeast", "southwest",
              "forward", "rest", "axis", "system", "clockwise", "counterclockwise", "x", "y", "z",
              "thing", "object", "item", "side", "process", "movement", "rotation", "change",
              "answer", "question", "situation", "trajectory", "meter", "centimeter",
              "cm", "m", "step", "turn", "orientation", "both", "shape", "overlap", "color", "colour",
              "pattern", "style", "type", "kind", "appearance", "feature", "material", "rotating", "move",
              "succession", "show", "spot", "interval", "relationship", "layout", "observation"}
ATTRIBUTE_NOUNS = {"volume", "size", "color", "colour", "pattern", "shape", "height", "width", "length",
                   "number", "logo", "text", "design", "print", "stripe", "lid", "cover"}
COMPASS_WORDS = {"north", "south", "east", "west", "northeast", "northwest", "southeast", "southwest",
                 "northern", "southern", "eastern", "western"}
QUANTITY_HEADS = {"height", "width", "length", "radius", "diameter", "size", "gap", "thickness", "volume",
                  "speed", "depth", "distance"}
NAMED_ROOMS = {"kitchen", "bedroom", "bathroom", "restroom", "corridor", "hallway", "hall", "passage",
               "stairwell", "lounge", "balcony", "office", "study", "entrance", "exit", "pantry", "closet",
               "foyer", "lobby", "gym", "library", "porch", "terrace", "yard", "garden", "street", "room",
               "classroom", "attic", "basement", "garage", "doorway", "aisle", "courtyard"}
AREA_HEADS = {"area", "zone", "space", "region", "section", "corner"}
PART_HEADS = {"surface", "top", "bottom", "edge", "end", "center", "middle", "backrest", "cushion", "leg",
              "handle", "opening", "floor", "face", "tip", "level", "ceiling", "part"}
CAMERA_HEADS = {"camera", "photographer", "dashcam"}
MATERIALS = {"wooden", "wood", "glass", "metal", "stone", "ceramic", "plastic", "leather", "fabric",
             "porcelain", "marble", "brick", "steel", "iron", "bronze", "paper", "cardboard", "golden",
             "silver", "gold", "transparent"}
NOT_ATTRIBUTES = {"first", "second", "third", "fourth", "fifth", "last", "other", "same", "entire", "whole",
                  "following", "current", "different", "separate", "single", "main", "actual", "total",
                  "initial", "final", "original", "previous", "next", "specific", "certain"}
POSITION_WORDS = {"left", "right", "top", "bottom", "middle", "center", "front", "back", "far", "rear",
                  "upper", "lower", "corner", "side", "edge", "leftmost", "rightmost", "foreground",
                  "background"}
SPATIAL_PREPS = {"on", "above", "under", "below", "behind", "beside", "near", "by", "at", "between", "with",
                 "against", "inside", "outside", "over", "along", "opposite", "across", "around", "in",
                 "into", "underneath", "beneath", "atop"}
RELATIONAL_VERBS = {"hang", "sit", "stand", "lie", "lean", "mount", "put", "hold", "cover", "surround",
                    "attach", "fix", "insert", "stack", "carry", "face", "point", "display", "label", "park",
                    "store", "fill", "decorate", "contain"}
COCO80 = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"}
COCO_ALIASES = {"sofa": "couch", "television": "tv", "fridge": "refrigerator", "plant": "potted plant",
                "table": "dining table", "phone": "cell phone", "monitor": "tv", "people": "person",
                "man": "person", "woman": "person", "pedestrian": "person", "bike": "bicycle"}
# ADE20K-150 (SceneParse150) class names, the standard semantic-segmentation
# vocabulary that includes structural "stuff" (wall, door, stairs) absent from LVIS.
ADE20K150 = {
    "wall", "building", "sky", "floor", "tree", "ceiling", "road", "bed", "windowpane", "grass", "cabinet",
    "sidewalk", "person", "earth", "door", "table", "mountain", "plant", "curtain", "chair", "car", "water",
    "painting", "sofa", "shelf", "house", "sea", "mirror", "rug", "field", "armchair", "seat", "fence", "desk",
    "rock", "wardrobe", "lamp", "bathtub", "railing", "cushion", "base", "box", "column", "signboard",
    "chest of drawers", "counter", "sand", "sink", "skyscraper", "fireplace", "refrigerator", "grandstand",
    "path", "stairs", "runway", "case", "pool table", "pillow", "screen door", "stairway", "river", "bridge",
    "bookcase", "blind", "coffee table", "toilet", "flower", "book", "hill", "bench", "countertop", "stove",
    "palm", "kitchen island", "computer", "swivel chair", "boat", "bar", "arcade machine", "hovel", "bus",
    "towel", "light", "truck", "tower", "chandelier", "awning", "streetlight", "booth", "television receiver",
    "airplane", "dirt track", "apparel", "pole", "land", "bannister", "escalator", "ottoman", "bottle",
    "buffet", "poster", "stage", "van", "ship", "fountain", "conveyer belt", "canopy", "washer", "plaything",
    "swimming pool", "stool", "barrel", "basket", "waterfall", "tent", "bag", "minibike", "cradle", "oven",
    "ball", "food", "step", "tank", "trade name", "microwave", "pot", "animal", "bicycle", "lake",
    "dishwasher", "screen", "blanket", "sculpture", "hood", "sconce", "vase", "traffic light", "tray",
    "ashcan", "fan", "pier", "crt screen", "plate", "monitor", "bulletin board", "shower", "radiator",
    "glass", "clock", "flag"}
ADE_ALIASES = {"window": "windowpane", "stair": "stairs", "staircase": "stairway", "television": "television receiver",
               "tv": "television receiver", "trash": "ashcan", "clothe": "apparel", "clothes": "apparel",
               "carpet": "rug", "pillar": "column", "drawer": "chest of drawers", "people": "person", "man": "person", "woman": "person", "pool": "swimming pool"}
assert len(ADE20K150) == 150, len(ADE20K150)
ENTITY_TYPES = ["object", "region_room", "region_functional", "region_generic", "part", "camera"]
MODES = ["attribute", "relational", "image_anchored", "in_image_position"]


def head_words(names: set[str]) -> set[str]:
    """Last word of each multiword name ("laptop computer" -> "computer")."""
    return {n.split()[-1] for n in names}


def in_vocab(phrase: str, head: str, names: set[str], heads: set[str], aliases: dict[str, str]) -> bool:
    """A mention matches when its compound phrase or head is a class name, or
    its head is the head of a multiword class name."""
    head = aliases.get(head, head)
    return phrase in names or head in names or head in heads


def load_lvis_names() -> set[str]:
    """LVIS v1 names and synonyms, lowercased, with sense tags like "(weapon)" removed."""
    if not LVIS_PATH.exists():
        raise FileNotFoundError(f"Download {LVIS_URL} to {LVIS_PATH}")
    text = LVIS_PATH.read_text()
    start = text.index("LVIS_CATEGORIES = [") + len("LVIS_CATEGORIES = ")
    categories = ast.literal_eval(text[start: text.rindex("]") + 1])
    assert len(categories) == 1203, len(categories)
    names = set()
    for cat in categories:
        for name in [cat["name"], *cat["synonyms"]]:
            names.add(re.sub(r"\s*\(.*?\)", "", name).replace("_", " ").strip().lower())
    return names


def mention_type(head: str, words, root) -> str:
    if head in CAMERA_HEADS or any(t.lemma_.lower() in CAMERA_HEADS for t in words):
        return "camera"
    if head in QUANTITY_HEADS:
        return "quantity"
    if head in AREA_HEADS:
        modifiers = [t for t in words if t.i != root.i and t.dep_ in ("compound", "amod")]
        functional = [t for t in modifiers if t.lower_ not in NOT_ATTRIBUTES | POSITION_WORDS | COMPASS_WORDS]
        if head == "corner" and not functional:
            return "region_generic" if any(t.lower_ in COMPASS_WORDS for t in modifiers) else "part"
        return "region_functional" if functional else "region_generic"
    if head in NAMED_ROOMS:
        return "region_room"
    if head in PART_HEADS:
        return "region_generic" if any(t.lower_ in COMPASS_WORDS for t in words) else "part"
    return "object"


def descriptor_modes(chunk, root) -> dict[str, bool]:
    inside = [t for t in chunk if t.i != root.i]
    modes = dict.fromkeys(MODES, False)
    modes["attribute"] = any(
        (t.dep_ == "amod" and t.lower_ not in NOT_ATTRIBUTES and t.lower_ not in POSITION_WORDS)
        or t.lower_ in MATERIALS for t in inside)
    modes["in_image_position"] = any(t.lower_ in {"leftmost", "rightmost"} for t in inside)
    for child in root.children:
        if child.i < chunk.end:
            continue
        objs = [g for g in child.children if g.dep_ == "pobj"]
        obj_lemmas = {o.lemma_.lower() for o in objs}
        if child.dep_ == "prep" and obj_lemmas & IMAGE_NOUNS:
            modes["image_anchored"] = True
        elif child.dep_ == "prep" and obj_lemmas & POSITION_WORDS:
            modes["in_image_position"] = True
            nested = {p.lemma_.lower() for o in objs for g in o.children if g.dep_ == "prep" for p in g.children}
            if nested & IMAGE_NOUNS:
                modes["image_anchored"] = True
        elif child.dep_ == "prep" and child.lower_ == "with" and obj_lemmas & ATTRIBUTE_NOUNS:
            modes["attribute"] = True
        elif child.dep_ == "prep" and child.lower_ in SPATIAL_PREPS and "relation" not in obj_lemmas:
            modes["relational"] = True
        elif child.dep_ == "advmod" and child.lower_ in {"next", "directly", "right"}:
            modes["relational"] = True
        elif child.dep_ in ("acl", "relcl") and child.lemma_.lower() in RELATIONAL_VERBS:
            modes["relational"] = True
    return modes


def mentions_from_doc(doc, lvis: set[str], lvis_heads: set[str] | None = None) -> list[dict]:
    """Typed mentions for one parsed text. Vocabulary flags are set for objects only.

    LVIS matching is lenient (an LVIS name, or the head of a multiword name),
    so its coverage is an upper bound; COCO and ADE20K require the name itself.
    """
    lvis_heads = head_words(lvis) if lvis_heads is None else lvis_heads
    out = []
    for chunk in doc.noun_chunks:
        root = chunk.root
        head = root.lemma_.lower()
        words = [t for t in chunk if t.dep_ not in ("det", "poss", "nummod") and t.pos_ != "DET"]
        has_camera = any(t.lemma_.lower() in CAMERA_HEADS for t in words)
        if not has_camera and (root.pos_ not in ("NOUN", "PROPN") or head in IMAGE_NOUNS
                               or head in SKIP_HEADS or not head.isalpha()):
            continue
        kind = mention_type(head, words, root)
        modes = descriptor_modes(chunk, root) if kind != "quantity" else dict.fromkeys(MODES, False)
        vocab = {"in_coco": None, "in_lvis": None, "in_ade20k": None}
        if kind == "object":
            vocab = dict.fromkeys(vocab, False)
            for form in {head, root.lower_}:
                phrase = " ".join([t.lemma_.lower() for t in words if t.dep_ == "compound"] + [form])
                vocab["in_coco"] |= in_vocab(phrase, form, COCO80, set(), COCO_ALIASES)
                vocab["in_lvis"] |= in_vocab(phrase, form, lvis, lvis_heads, {})
                vocab["in_ade20k"] |= in_vocab(phrase, form, ADE20K150, set(), ADE_ALIASES)
        out.append({"text": chunk.text, "head": head, "type": kind, **modes,
                    "bare": not any(modes.values()), **vocab})
    return out
