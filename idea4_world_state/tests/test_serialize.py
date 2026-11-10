"""Flat and graph serializers: Section 8.4 text and atom round-trips."""

import json
import random

from idea4_world_state import contracts
from idea4_world_state.inventory import InventoryError, atom_id, atom_text, atoms, parse_inventory
from idea4_world_state.serialize import parse_flat, parse_graph, render_flat, render_graph

# Section 8.4, including the newline that terminates the last line.
FLAT_VALID_FULL = """\
Evidence extracted by a model from the images (may be incomplete or wrong):
Query frame: camera:cam2 (basis: stem asks from the second viewpoint).
Temporal order: unknown.
Camera cam1 took image_1. Camera cam2 took image_2.
[e1] e1 is an object labeled "red chair"; it is the same entity across views (uncertainty medium).
[e2] e2 is a region labeled "kitchen doorway".
[o1] In image_1 (left), e1 is observed: "red chair by window".
[o2] In image_2 (center), e1 is observed: "same chair, seen from behind".
[o3] In image_1 (right), e2 is observed: "open doorway".
[c1] In frame image_plane:image_1, for image_1, e1 is left_of e2. Support: o1, o3. Uncertainty: low.
[c2] Across views, attribute color of e1 is "red". Support: o1. Uncertainty: low.
[u1] Unknown: whether the doorway is visible in image 2.
"""

GRAPH_VALID_FULL = """\
World state graph extracted by a model from the images (may be incomplete or wrong):
Query frame: camera:cam2 | basis: stem asks from the second viewpoint
Temporal order: unknown
Nodes:
  cam1 [camera] image_1
  cam2 [camera] image_2
  e1 [object] "red chair" | cross-view: same (uncertainty medium)
    o1 @image_1 left: "red chair by window"
    o2 @image_2 center: "same chair, seen from behind"
    c2 color = "red" | across views | support o1 | uncertainty low
  e2 [region] "kitchen doorway"
    o3 @image_1 right: "open doorway"
Identity links: none
Edges by frame:
  image_plane:image_1
    c1 e1 -left_of-> e2 | image_1 | support o1,o3 | uncertainty low
Unknowns:
  u1 whether the doorway is visible in image 2
"""

_SPATIAL = contracts.RELATION_PREDICATES["spatial"]
_CAMERA_MOTION = contracts.RELATION_PREDICATES["camera_motion"]
_OBJECT_MOTION = contracts.RELATION_PREDICATES["object_motion"]
_WHERE = contracts.WHERE_VALUES
_UNCERTAINTY = contracts.UNCERTAINTY


def _expectations():
    return json.loads((contracts.FIXTURES_DIR / "expectations.json").read_text(encoding="utf-8"))


def _load(name):
    return json.loads((contracts.FIXTURES_DIR / name).read_text(encoding="utf-8"))


def _valid_items():
    return [(name, item) for name, item in _expectations().items() if item["semantically_valid"]]


def _claim_facts(inv):
    return {
        (
            claim["id"],
            claim["kind"],
            claim["subject"],
            claim["predicate"],
            claim["object"],
            claim["value"],
        )
        for claim in inv["claims"]
    }


def _atom_claim_facts(found):
    return {
        (atom[1], atom[2], atom[3], atom[4], atom[5], atom[6])
        for atom in found
        if atom[0] == "claim"
    }


def _assert_round_trip(inv):
    found = atoms(inv)
    flat = render_flat(inv)
    graph = render_graph(inv)
    assert flat == render_flat(inv)
    assert graph == render_graph(inv)
    assert isinstance(parse_flat(flat), frozenset)
    assert isinstance(parse_graph(graph), frozenset)
    assert parse_flat(flat) == found == parse_graph(graph)
    assert _atom_claim_facts(found) == _claim_facts(inv)
    assert sum(atom[0] == "claim" for atom in found) == len(inv["claims"])
    for atom in found:
        ident = atom_id(atom)
        text = atom_text(atom)
        if atom[0] in ("query_frame", "temporal_order", "camera"):
            assert text in flat
        else:
            assert f"[{ident}] {text}" in flat


def _strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)


def _claim(cid, **fields):
    base = {
        "id": cid,
        "kind": "relation",
        "subject": "e1",
        "predicate": "left_of",
        "object": "e2",
        "value": None,
        "frame": "unknown",
        "image_id": None,
        "support": ["o1"],
        "uncertainty": "low",
        "note": None,
    }
    base.update(fields)
    return base


def _random_inventory(rng: random.Random, index: int) -> tuple[dict, int]:
    """A valid inventory. Index forces every required feature; rng varies the rest."""
    n_images = rng.randint(2, 4)

    def uncertainty():
        return rng.choice(_UNCERTAINTY)
    cameras = [
        {"id": f"cam{i}", "image_id": f"image_{i}"}
        for i in range(1, n_images + 1)
    ]
    entities = [
        {
            "id": "e1",
            "type": "object",
            "label": 'red "chair"',
            "cross_view": {"status": "same", "uncertainty": uncertainty()},
        },
        {"id": "e2", "type": "region", "label": "floor", "cross_view": None},
    ]
    observations = [
        {
            "id": "o1",
            "entity": "e1",
            "image_id": "image_1",
            "where": rng.choice(_WHERE),
            "description": "seen | from the side",
        },
        {
            "id": "o2",
            "entity": "e1",
            "image_id": "image_2",
            "where": "center",
            "description": "same chair",
        },
        {
            "id": "o3",
            "entity": "e2",
            "image_id": "image_1",
            "where": "bottom",
            "description": "floor area",
        },
    ]
    links = [{
        "id": "i1",
        "entities": ["e1", "e2"],
        "status": "tentative" if index % 2 == 0 else "different",
        "uncertainty": uncertainty(),
    }]
    claims = [
        _claim(
            "c1",
            predicate=rng.choice(_SPATIAL),
            frame="image_plane:image_1",
            image_id="image_1",
            support=["o1", "o3"],
            uncertainty=uncertainty(),
        ),
        _claim(
            "c2",
            kind="attribute",
            predicate="color",
            object=None,
            value="red",
            frame="unknown",
            image_id=None,
            support=["o1"],
            uncertainty=uncertainty(),
        ),
        _claim(
            "c3",
            kind="attribute",
            predicate="state",
            object=None,
            value="open",
            frame="unknown",
            image_id="image_2",
            support=["o2"],
            uncertainty=uncertainty(),
        ),
        _claim(
            "c4",
            kind="attribute",
            subject="e2",
            predicate="other",
            object=None,
            value="worn",
            frame="unknown",
            image_id=None,
            support=["o3"],
            uncertainty="high",
            note='mark "x" | scratch',
        ),
    ]
    if index % 2 == 0:
        claims.append(_claim(
            "c5",
            subject="cam2",
            predicate=rng.choice(_CAMERA_MOTION),
            object="cam1",
            frame="unknown",
            image_id="image_2",
            support=["o2"],
            uncertainty=uncertainty(),
        ))
    else:
        claims.append(_claim(
            "c5",
            subject="e1",
            predicate=rng.choice(_OBJECT_MOTION),
            object="e2" if index % 4 == 3 else "cam1",
            frame="camera:cam1",
            image_id="image_2",
            support=["o1", "o2"],
            uncertainty=uncertainty(),
        ))
    mode = index % 4
    if mode == 0:
        claims.append(_claim(
            "c6",
            predicate="near",
            frame="camera:cam1",
            image_id=None,
            support=["o1"],
        ))
    elif mode == 1:
        claims.append(_claim(
            "c6",
            kind="attribute",
            predicate="size",
            object=None,
            value="large",
            frame="camera:cam1",
            image_id=None,
            support=["o1"],
        ))
    elif mode == 2:
        claims.append(_claim(
            "c6",
            predicate="on",
            frame="unknown",
            image_id=None,
            support=["o1", "o3"],
        ))
    else:
        claims.append(_claim(
            "c6",
            kind="attribute",
            predicate="text",
            object=None,
            value='say "hi"',
            frame="image_plane:image_1",
            image_id="image_1",
            support=["o1"],
        ))
    tricky_basis = 'stem "asks" | second view'
    if index % 2 == 0:
        order = [f"image_{i}" for i in range(1, n_images + 1)]
        rng.shuffle(order)
        temporal = {"status": "supported", "order": order, "basis": tricky_basis}
        query_basis = tricky_basis
    else:
        temporal = {"status": "unknown", "order": None, "basis": None}
        query_basis = None
    frames = ("unknown", "camera:cam1", "camera:cam2", "image_plane:image_1", "entity:e1", "entity:e2")
    document = {
        "schema_version": "ws-1",
        "query_frame": {"frame": frames[index % len(frames)], "basis": query_basis},
        "temporal_order": temporal,
        "cameras": cameras,
        "entities": entities,
        "observations": observations,
        "identity_links": links,
        "claims": claims,
        "unknowns": ["maybe\nnot sure"],
    }
    return document, n_images


def _tricky_inventory():
    """Quotes, pipes, newlines, and the flat/graph delimiter strings."""
    return {
        "schema_version": "ws-1",
        "query_frame": {"frame": "entity:e1", "basis": "see (a). next"},
        "temporal_order": {
            "status": "supported",
            "order": ["image_2", "image_1"],
            "basis": 'door "opened" | later',
        },
        "cameras": [
            {"id": "cam1", "image_id": "image_1"},
            {"id": "cam2", "image_id": "image_2"},
        ],
        "entities": [
            {
                "id": "e1",
                "type": "object",
                "label": 'red "chair"',
                "cross_view": {"status": "same", "uncertainty": "low"},
            },
            {"id": "e2", "type": "region", "label": "door|frame", "cross_view": None},
        ],
        "observations": [
            {
                "id": "o1",
                "entity": "e1",
                "image_id": "image_1",
                "where": "left",
                "description": 'seen | from "behind"',
            },
            {
                "id": "o2",
                "entity": "e1",
                "image_id": "image_2",
                "where": "center",
                "description": "same",
            },
            {
                "id": "o3",
                "entity": "e2",
                "image_id": "image_1",
                "where": "bottom",
                "description": "floor",
            },
        ],
        "identity_links": [
            {"id": "i1", "entities": ["e1", "e2"], "status": "tentative", "uncertainty": "high"}
        ],
        "claims": [
            _claim(
                "c1",
                frame="image_plane:image_1",
                image_id="image_1",
                support=["o1", "o3"],
            ),
            _claim(
                "c2",
                kind="attribute",
                predicate="other",
                object=None,
                value="x. Note: y",
                frame="unknown",
                image_id=None,
                support=["o1"],
                uncertainty="medium",
                note='a | note "b"',
            ),
            _claim(
                "c3",
                kind="attribute",
                predicate="state",
                object=None,
                value="closed",
                frame="unknown",
                image_id="image_2",
                support=["o2"],
            ),
            _claim(
                "c4",
                subject="cam2",
                predicate="camera_moved_forward",
                object="cam1",
                frame="unknown",
                image_id="image_2",
                support=["o2"],
            ),
            _claim(
                "c5",
                predicate="near",
                frame="camera:cam1",
                image_id=None,
                support=["o1"],
            ),
        ],
        "unknowns": ["maybe\nnot", "has | pipe"],
    }


def test_valid_full_matches_section_8_4():
    document = _load("valid_full.json")
    assert render_flat(document) == FLAT_VALID_FULL
    assert render_graph(document) == GRAPH_VALID_FULL


def test_round_trip_valid_fixtures():
    for name, expected in _valid_items():
        document = _load(name)
        parsed = parse_inventory(json.dumps(document), expected["n_images"])
        if name == "invalid_cross_view_missing.json":
            assert parsed["entities"][0]["cross_view"] == {"status": "same", "uncertainty": "medium"}
        else:
            assert parsed == document, name
        _assert_round_trip(parsed)


def test_serializers_add_no_claims_and_are_deterministic():
    for name, _expected in _valid_items():
        document = _load(name)
        flat = render_flat(document)
        graph = render_graph(document)
        assert flat == render_flat(document)
        assert graph == render_graph(document)
        for found in (atoms(document), parse_flat(flat), parse_graph(graph)):
            assert _atom_claim_facts(found) == _claim_facts(document)
    full = _load("valid_full.json")
    assert "left_of" in render_flat(full)
    assert "right_of" not in render_flat(full)
    assert "right_of" not in render_graph(full)


def test_special_characters_round_trip():
    document = _tricky_inventory()
    assert parse_inventory(json.dumps(document), 2) == document
    _assert_round_trip(document)
    found = atoms(document)
    assert ("unknown", "u1", "maybe\nnot") in found
    assert any(atom[0] == "entity" and 'red "chair"' == atom[3] for atom in found)
    note = next(atom for atom in found if atom[0] == "claim" and atom[1] == "c2")
    assert note[11] == 'a | note "b"'
    assert note[6] == "x. Note: y"


def test_random_valid_inventories_round_trip():
    rng = random.Random(0)
    seen = {
        "identity_link": False,
        "camera_motion": False,
        "object_motion": False,
        "other_note": False,
        "temporal": False,
        "view_specific": False,
        "across_view": False,
        "quote": False,
        "pipe": False,
        "newline": False,
    }
    for index in range(200):
        document, n_images = _random_inventory(rng, index)
        try:
            parsed = parse_inventory(json.dumps(document), n_images)
        except InventoryError as exc:
            raise AssertionError(f"index {index} rejected as {exc.rule}: {exc.message}") from exc
        assert parsed == document
        _assert_round_trip(document)
        predicates = {claim["predicate"] for claim in document["claims"]}
        seen["identity_link"] |= bool(document["identity_links"])
        seen["camera_motion"] |= bool(predicates & set(_CAMERA_MOTION))
        seen["object_motion"] |= bool(predicates & set(_OBJECT_MOTION))
        seen["other_note"] |= any(
            claim["predicate"] == "other" and claim["note"] for claim in document["claims"]
        )
        seen["temporal"] |= document["temporal_order"]["status"] == "supported"
        seen["view_specific"] |= any(claim["image_id"] is not None for claim in document["claims"])
        seen["across_view"] |= any(claim["image_id"] is None for claim in document["claims"])
        strings = list(_strings(document))
        seen["quote"] |= any('"' in item for item in strings)
        seen["pipe"] |= any("|" in item for item in strings)
        seen["newline"] |= any("\n" in item for item in strings)
    missing = [name for name, hit in seen.items() if not hit]
    assert not missing
