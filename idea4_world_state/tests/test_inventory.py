"""Inventory parsing, semantic rules, and fact atoms."""

import copy
import json
from pathlib import Path

import pytest

from idea4_world_state import contracts
from idea4_world_state.inventory import (
    ERROR_RULES,
    InventoryError,
    atom_id,
    atom_sort_key,
    atom_text,
    atoms,
    parse_inventory,
    parse_inventory_with_repairs,
)

# rule token, and a fragment of the message that names the fixture's intent
_INVALID = {
    "invalid_unresolved_ref.json": ("unresolved_ref", "e9"),
    "invalid_over_limit.json": ("schema", "entities"),
    "invalid_truncated.txt": ("json", "JSON"),
    "invalid_image_plane_mismatch.json": ("image_id", "image_2"),
    "invalid_motion_endpoints.json": ("motion_endpoints", "camera"),
    "invalid_state_without_image.json": ("image_id", "state"),
    "invalid_other_without_note.json": ("note", "other"),
    "invalid_temporal_not_permutation.json": ("temporal_order", "permutation"),
    "invalid_duplicate_id.json": ("duplicate_id", "e1"),
    "invalid_camera_count.json": ("camera_count", "2"),
}


def _expectations():
    return json.loads((contracts.FIXTURES_DIR / "expectations.json").read_text(encoding="utf-8"))


def _text(name):
    return (contracts.FIXTURES_DIR / name).read_text(encoding="utf-8")


def _inv(name):
    return json.loads(_text(name))


def _rejects(document, n_images=2):
    with pytest.raises(InventoryError) as caught:
        parse_inventory(json.dumps(document), n_images)
    return caught.value


def test_error_rules_are_the_stable_tokens():
    assert ERROR_RULES == (
        "format",
        "json",
        "not_object",
        "schema",
        "camera_count",
        "camera_mapping",
        "duplicate_id",
        "image_range",
        "unresolved_ref",
        "cross_view",
        "image_id",
        "motion_endpoints",
        "note",
        "temporal_order",
    )


def test_fixtures_match_expectations():
    expectations = _expectations()
    assert set(_INVALID) == {name for name, item in expectations.items() if not item["semantically_valid"]}
    for name, expected in expectations.items():
        text = _text(name)
        if expected["semantically_valid"]:
            parsed = parse_inventory(text, expected["n_images"])
            original = json.loads(text)
            if name == "invalid_cross_view_missing.json":
                assert parsed["entities"][0]["cross_view"] == {"status": "same", "uncertainty": "medium"}
                original["entities"][0]["cross_view"] = parsed["entities"][0]["cross_view"]
            assert parsed == original, name
            continue
        with pytest.raises(InventoryError) as caught:
            parse_inventory(text, expected["n_images"])
        rule, fragment = _INVALID[name]
        assert caught.value.rule == rule, name
        assert fragment in caught.value.message, (name, caught.value.message)


def test_fenced_block_and_bare_json_agree():
    raw = _text("valid_full.json")
    bare = parse_inventory(raw, 2)
    for fenced in (
        f"```json\n{raw.strip()}\n```",
        f"```\n{raw.strip()}\n```",
        f"\n```JSON\n{raw.strip()}\n```\n",
    ):
        assert parse_inventory(fenced, 2) == bare


def test_truncated_and_fenced_prose_raise():
    raw = _text("valid_full.json").strip()
    cases = {
        _text("invalid_truncated.txt"): "json",
        "{": "json",
        f"```json\n{raw[:40]}\n```": "json",
        f"Here is the inventory:\n```json\n{raw}\n```\nDone.": "format",
        f"```json\n{raw}\n```\n```\n[]\n```": "format",
        "not json": "format",
        "": "format",
        "[]": "not_object",
        "null": "not_object",
    }
    for text, rule in cases.items():
        with pytest.raises(InventoryError) as caught:
            parse_inventory(text, 2)
        assert caught.value.rule == rule, (rule, text[:40], caught.value)


def test_schema_lengths_for_note_and_basis():
    assert contracts.MAX_LENGTHS["note"] == 120
    assert contracts.MAX_LENGTHS["query_frame.basis"] == 160
    assert contracts.MAX_LENGTHS["temporal_order.basis"] == 160
    schema = json.loads(contracts.SCHEMA_PATH.read_text(encoding="utf-8"))
    note = schema["properties"]["claims"]["items"]["properties"]["note"]
    assert note["maxLength"] == 120
    assert schema["properties"]["query_frame"]["properties"]["basis"]["maxLength"] == 160
    assert schema["properties"]["temporal_order"]["properties"]["basis"]["maxLength"] == 160
    supported = schema["properties"]["temporal_order"]["allOf"][1]["then"]["properties"]["basis"]
    assert supported["maxLength"] == 160

    document = _inv("valid_full.json")
    document["claims"][1]["note"] = "n" * 120
    document["query_frame"]["basis"] = "b" * 160
    assert parse_inventory(json.dumps(document), 2)["claims"][1]["note"] == "n" * 120

    too_long_note = copy.deepcopy(document)
    too_long_note["claims"][1]["note"] = "n" * 121
    assert _rejects(too_long_note).rule == "schema"
    too_long_basis = copy.deepcopy(document)
    too_long_basis["query_frame"]["basis"] = "b" * 161
    assert _rejects(too_long_basis).rule == "schema"
    too_long_order = copy.deepcopy(document)
    too_long_order["temporal_order"] = {
        "status": "supported",
        "order": ["image_2", "image_1"],
        "basis": "t" * 161,
    }
    assert _rejects(too_long_order).rule == "schema"
    ok_order = copy.deepcopy(too_long_order)
    ok_order["temporal_order"]["basis"] = "t" * 160
    assert parse_inventory(json.dumps(ok_order), 2)["temporal_order"]["basis"] == "t" * 160


def test_remaining_semantic_rules():
    swapped = _inv("valid_full.json")
    swapped["cameras"].reverse()
    assert _rejects(swapped).rule == "camera_mapping"

    out_of_range = _inv("valid_full.json")
    out_of_range["observations"][0]["image_id"] = "image_9"
    assert _rejects(out_of_range).rule == "image_range"

    extra_cross = _inv("valid_tentative_identity.json")
    extra_cross["entities"][0]["cross_view"] = {"status": "same", "uncertainty": "low"}
    assert _rejects(extra_cross).rule == "cross_view"

    same_camera = _inv("valid_changing_observation.json")
    same_camera["claims"][2]["predicate"] = "camera_moved_forward"
    same_camera["claims"][2]["subject"] = "cam1"
    same_camera["claims"][2]["object"] = "cam1"
    assert _rejects(same_camera).rule == "motion_endpoints"

    same_entity = _inv("valid_changing_observation.json")
    same_entity["claims"][2]["object"] = "e1"
    assert _rejects(same_entity).rule == "motion_endpoints"

    missing_frame = _inv("valid_full.json")
    missing_frame["query_frame"]["frame"] = "entity:e9"
    assert _rejects(missing_frame).rule == "unresolved_ref"


def test_empty_note_satisfies_other():
    document = _inv("valid_unknown_frame.json")
    other = next(claim for claim in document["claims"] if claim["predicate"] == "other")
    other["note"] = ""
    parsed = parse_inventory(json.dumps(document), 2)
    assert next(claim for claim in parsed["claims"] if claim["predicate"] == "other")["note"] == ""


def test_changing_observation_state_frames_stay_unknown():
    document = _inv("valid_changing_observation.json")
    states = [claim for claim in document["claims"] if claim["predicate"] == "state"]
    assert [claim["frame"] for claim in states] == ["unknown", "unknown"]
    assert [claim["image_id"] for claim in states] == ["image_1", "image_2"]
    assert parse_inventory(json.dumps(document), 2) == document


def test_valid_full_atoms_and_ids():
    document = _inv("valid_full.json")
    found = atoms(document)
    assert ("query_frame", "camera:cam2", "stem asks from the second viewpoint") in found
    assert ("temporal_order", "unknown", None, None) in found
    assert ("camera", "cam1", "image_1") in found
    assert ("camera", "cam2", "image_2") in found
    assert ("entity", "e1", "object", "red chair", "same", "medium") in found
    assert ("entity", "e2", "region", "kitchen doorway", None, None) in found
    assert ("observation", "o1", "e1", "image_1", "left", "red chair by window") in found
    assert (
        "claim",
        "c1",
        "relation",
        "e1",
        "left_of",
        "e2",
        None,
        "image_plane:image_1",
        "image_1",
        ("o1", "o3"),
        "low",
        None,
    ) in found
    assert (
        "claim",
        "c2",
        "attribute",
        "e1",
        "color",
        None,
        "red",
        "unknown",
        None,
        ("o1",),
        "low",
        None,
    ) in found
    assert ("unknown", "u1", "whether the doorway is visible in image 2") in found
    assert len(found) == 12
    assert [atom_id(atom) for atom in sorted(found, key=atom_sort_key)] == [
        "query_frame",
        "temporal_order",
        "cam1",
        "cam2",
        "e1",
        "e2",
        "o1",
        "o2",
        "o3",
        "c1",
        "c2",
        "u1",
    ]
    query = next(atom for atom in found if atom[0] == "query_frame")
    assert atom_text(query) == "Query frame: camera:cam2 (basis: stem asks from the second viewpoint)."
    color = next(atom for atom in found if atom[0] == "claim" and atom[1] == "c2")
    assert atom_text(color) == (
        'Across views, attribute color of e1 is "red". Support: o1. Uncertainty: low.'
    )


def test_shared_entity_id_fills_missing_cross_view():
    parsed = parse_inventory(_text("invalid_cross_view_missing.json"), 2)
    assert parsed["entities"][0]["cross_view"] == {"status": "same", "uncertainty": "medium"}
    saved = Path("/Users/zhengfeili/2026_Fall/11777/11777Project/MMML2026/runs/idea4_world_state/debug_extract_v3/inventories")
    if not saved.is_dir():
        return
    for name in ("14", "236", "319", "33", "531", "912"):
        document = json.loads((saved / f"{name}.error.json").read_text(encoding="utf-8"))
        inventory = parse_inventory(document["text"], 2)
        shared = [
            entity for entity in inventory["entities"]
            if len({obs["image_id"] for obs in inventory["observations"] if obs["entity"] == entity["id"]}) >= 2
        ]
        assert shared, name
        assert all(entity["cross_view"] == {"status": "same", "uncertainty": "medium"} for entity in shared), name
        _inventory, repairs = parse_inventory_with_repairs(document["text"], 2)
        assert {repair["entity_id"] for repair in repairs} == {entity["id"] for entity in shared}
        assert all(repair["uncertainty_source"] == "parser_default" for repair in repairs)


def test_unknowns_are_numbered_by_position():
    document = _inv("valid_empty.json")
    document["unknowns"] = ["b second", "a first"]
    numbered = {atom_id(atom): atom[2] for atom in atoms(document) if atom[0] == "unknown"}
    assert numbered == {"u1": "b second", "u2": "a first"}


def test_tentative_identity_atom():
    document = _inv("valid_tentative_identity.json")
    link = next(atom for atom in atoms(document) if atom[0] == "identity_link")
    assert link == ("identity_link", "i1", "e1", "e2", "tentative", "high")
    assert atom_id(link) == "i1"
    assert atom_text(link) == "Identity link between e1 and e2 is tentative (uncertainty high)."
