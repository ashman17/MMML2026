"""Skeleton tests: CLI stubs, contract enums, and ws-1 fixtures."""

import copy
import json
import re
import subprocess
import sys

import pytest
from jsonschema import Draft202012Validator

from idea4_world_state import contracts

ENUM_DEFS = {
    "ENTITY_TYPES": "entity_type",
    "WHERE_VALUES": "where",
    "ATTRIBUTE_PREDICATES": "attribute_predicate",
    "UNCERTAINTY": "uncertainty",
    "IDENTITY_LINK_STATUSES": "identity_link_status",
    "CROSS_VIEW_STATUSES": "cross_view_status",
    "TEMPORAL_STATUSES": "temporal_status",
    "CLAIM_KINDS": "claim_kind",
}
PATTERN_DEFS = {
    "FRAME_PATTERN": "frame",
    "CAMERA_ID_PATTERN": "camera_id",
    "IMAGE_ID_PATTERN": "image_id",
    "ENTITY_ID_PATTERN": "entity_id",
    "OBSERVATION_ID_PATTERN": "observation_id",
    "IDENTITY_LINK_ID_PATTERN": "identity_link_id",
    "CLAIM_ID_PATTERN": "claim_id",
    "ENTITY_OR_CAMERA_ID_PATTERN": "entity_or_camera_id",
}
RELATION_GROUP_ORDER = ("spatial", "camera_motion", "object_motion", "other")
GITIGNORE_ENTRIES = (
    "runs/idea4_world_state/_cache/",
    "runs/idea4_world_state/*/inputs/",
    "runs/idea4_world_state/*/requests/",
)


def _run(argv):
    return subprocess.run(
        [sys.executable, "-m", "idea4_world_state", *argv],
        cwd=contracts.REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def _schema():
    schema = json.loads(contracts.SCHEMA_PATH.read_text())
    Draft202012Validator.check_schema(schema)
    return schema


def _validator():
    return Draft202012Validator(_schema())


def _expectations():
    return json.loads((contracts.FIXTURES_DIR / "expectations.json").read_text())


def _load_fixture(name):
    return json.loads((contracts.FIXTURES_DIR / name).read_text())


def _relation_predicates():
    predicates = []
    for group in RELATION_GROUP_ORDER:
        predicates.extend(contracts.RELATION_PREDICATES[group])
    return predicates


def _assert_frame_resolves(frame, image_ids, camera_ids, entity_ids):
    if frame == "unknown":
        return
    kind, ident = frame.split(":", 1)
    if kind == "image_plane":
        assert ident in image_ids, f"unresolved frame {frame}"
    elif kind == "camera":
        assert ident in camera_ids, f"unresolved frame {frame}"
    elif kind == "entity":
        assert ident in entity_ids, f"unresolved frame {frame}"
    else:
        raise AssertionError(frame)


def assert_field_rules(inv, n_images):
    """Section 8.3 rules a schema cannot settle on its own."""
    assert inv["schema_version"] == contracts.SCHEMA_VERSION
    cameras = inv["cameras"]
    assert len(cameras) == n_images
    image_ids = []
    camera_ids = []
    for index, camera in enumerate(cameras, start=1):
        assert camera["id"] == f"cam{index}"
        assert camera["image_id"] == f"image_{index}"
        camera_ids.append(camera["id"])
        image_ids.append(camera["image_id"])
    assert len(set(camera_ids)) == len(camera_ids)
    assert len(set(image_ids)) == len(image_ids)

    entities = inv["entities"]
    observations = inv["observations"]
    links = inv["identity_links"]
    claims = inv["claims"]
    unknowns = inv["unknowns"]
    assert len(entities) <= contracts.MAX_COUNTS["entities"], "entity count"
    assert len(observations) <= contracts.MAX_COUNTS["observations"], "observation count"
    assert len(links) <= contracts.MAX_COUNTS["identity_links"], "identity link count"
    assert len(claims) <= contracts.MAX_COUNTS["claims"], "claim count"
    assert len(unknowns) <= contracts.MAX_COUNTS["unknowns"], "unknown count"

    entity_ids = [entity["id"] for entity in entities]
    observation_ids = [obs["id"] for obs in observations]
    for ids in (entity_ids, observation_ids, [link["id"] for link in links], [claim["id"] for claim in claims]):
        assert len(ids) == len(set(ids)), "duplicate id"

    images_by_entity = {}
    for obs in observations:
        images_by_entity.setdefault(obs["entity"], set()).add(obs["image_id"])

    nodes = set(entity_ids) | set(camera_ids)
    image_id_set = set(image_ids)
    relation_predicates = _relation_predicates()
    camera_motion = set(contracts.RELATION_PREDICATES["camera_motion"])

    for entity in entities:
        assert re.fullmatch(contracts.ENTITY_ID_PATTERN, entity["id"])
        assert entity["type"] in contracts.ENTITY_TYPES
        assert len(entity["label"]) <= contracts.MAX_LENGTHS["label"]
        seen = images_by_entity.get(entity["id"], set())
        if len(seen) >= 2:
            assert entity["cross_view"] is not None, entity["id"]
            assert entity["cross_view"]["status"] in contracts.CROSS_VIEW_STATUSES
            assert entity["cross_view"]["uncertainty"] in contracts.UNCERTAINTY
        else:
            assert entity["cross_view"] is None, entity["id"]

    for obs in observations:
        assert re.fullmatch(contracts.OBSERVATION_ID_PATTERN, obs["id"])
        assert obs["entity"] in nodes, f"unresolved entity {obs['entity']}"
        assert obs["image_id"] in image_id_set, obs["image_id"]
        assert obs["where"] in contracts.WHERE_VALUES
        assert len(obs["description"]) <= contracts.MAX_LENGTHS["description"]

    for link in links:
        assert re.fullmatch(contracts.IDENTITY_LINK_ID_PATTERN, link["id"])
        assert link["status"] in contracts.IDENTITY_LINK_STATUSES
        assert link["uncertainty"] in contracts.UNCERTAINTY
        assert len(link["entities"]) == 2
        assert link["entities"][0] != link["entities"][1]
        for entity_id in link["entities"]:
            assert entity_id in set(entity_ids), f"unresolved entity {entity_id}"

    for claim in claims:
        assert re.fullmatch(contracts.CLAIM_ID_PATTERN, claim["id"])
        assert claim["kind"] in contracts.CLAIM_KINDS
        assert claim["subject"] in nodes, f"unresolved subject {claim['subject']}"
        assert claim["uncertainty"] in contracts.UNCERTAINTY
        assert re.fullmatch(contracts.FRAME_PATTERN, claim["frame"])
        _assert_frame_resolves(claim["frame"], image_id_set, set(camera_ids), set(entity_ids))
        assert claim["support"], "empty support"
        for support_id in claim["support"]:
            assert support_id in set(observation_ids), f"unresolved observation {support_id}"
        if claim["kind"] == "relation":
            assert claim["predicate"] in relation_predicates
            assert claim["object"] in nodes, f"unresolved object {claim['object']}"
            assert claim["value"] is None
        else:
            assert claim["predicate"] in contracts.ATTRIBUTE_PREDICATES
            assert claim["object"] is None
            assert isinstance(claim["value"], str)
            assert len(claim["value"]) <= contracts.MAX_LENGTHS["value"]
        if claim["predicate"] == "other":
            assert isinstance(claim["note"], str), "note required for other"
        if claim["predicate"] in camera_motion:
            assert claim["subject"] in set(camera_ids)
            assert claim["object"] in set(camera_ids)
        if claim["frame"].startswith("image_plane:"):
            assert claim["image_id"] == claim["frame"].split(":", 1)[1]
        elif claim["image_id"] is not None:
            assert claim["image_id"] in image_id_set

    temporal = inv["temporal_order"]
    assert temporal["status"] in contracts.TEMPORAL_STATUSES
    if temporal["status"] == "unknown":
        assert temporal["order"] is None
        assert temporal["basis"] is None
    else:
        assert isinstance(temporal["basis"], str)
        assert sorted(temporal["order"]) == sorted(image_ids)
        assert len(temporal["order"]) == len(image_ids)

    frame = inv["query_frame"]["frame"]
    assert re.fullmatch(contracts.FRAME_PATTERN, frame)
    _assert_frame_resolves(frame, image_id_set, set(camera_ids), set(entity_ids))
    basis = inv["query_frame"]["basis"]
    assert basis is None or isinstance(basis, str)

    for unknown in unknowns:
        assert isinstance(unknown, str)
        assert len(unknown) <= contracts.MAX_LENGTHS["unknown"]

    if not entities and not observations and not links and not claims:
        assert inv["query_frame"]["frame"] == "unknown"
        assert temporal["status"] == "unknown"
        assert unknowns, "empty inventory needs an unknown"


def test_help_lists_commands():
    top = _run(["--help"])
    assert top.returncode == 0
    assert "{prepare,run,freeze,analyze,audit}" in top.stdout
    for name in ("prepare", "run", "freeze", "analyze", "audit", "make", "import", "outcomes"):
        assert name in top.stdout
    audit = _run(["audit", "--help"])
    assert audit.returncode == 0
    assert "{make,import,outcomes}" in audit.stdout


def test_help_lists_flag_contract():
    run = _run(["run", "--help"])
    assert run.returncode == 0
    for text in (
        "--profile {raw_all,eval200,eval100,dev,timing,synthetic10}",
        "--conditions {raw|flat,graph|raw,flat,graph}",
        "http://localhost:8000/v1",
        "qwen3-vl-8b",
        "IDEA4_API_KEY",
        "4096",
        "768",
        "300",
        "--resume",
        "--mock",
        "--freeze",
    ):
        assert text in run.stdout
    # Help wraps long paths, so compare with whitespace removed.
    assert "runs/idea4_world_state/_cache" in "".join(run.stdout.split())
    freeze = _run(["freeze", "--help"])
    assert "{l40s,h100}" in freeze.stdout
    assert "--concurrency-run" in freeze.stdout
    audit = _run(["audit", "make", "--help"])
    assert "--n" in audit.stdout and "20" in audit.stdout
    prepare = _run(["prepare", "--help"])
    prepare_help = "".join(prepare.stdout.split())
    assert "data/MMSI_Bench.parquet" in prepare_help
    assert "analyses/language/splits/language_explore_confirm.json" in prepare_help


@pytest.mark.parametrize(
    ("argv", "owner"),
    [
        (["prepare", "--dataset", "/tmp/missing.parquet", "--out", "/tmp/x"], "WP1"),
        (
            ["run", "--prepare-dir", "/tmp/p", "--profile", "dev", "--conditions", "raw",
             "--server-manifest", "/tmp/m", "--out", "/tmp/r"],
            "WP5",
        ),
        (
            ["run", "--prepare-dir", "/tmp/p", "--profile", "eval200", "--conditions", "flat,graph",
             "--server-manifest", "/tmp/m", "--out", "/tmp/r"],
            "WP5",
        ),
        (
            ["run", "--prepare-dir", "/tmp/p", "--profile", "synthetic10", "--conditions", "raw,flat,graph",
             "--server-manifest", "/tmp/m", "--out", "/tmp/r"],
            "WP5",
        ),
        (
            ["freeze", "--dev-run", "/tmp/d", "--timing-run", "/tmp/t", "--synthetic-run", "/tmp/s",
             "--allocation-su", "1", "--gpu", "l40s", "--walltime-hours", "4",
             "--deadline", "2026-12-31", "--out", "/tmp/f"],
            "WP5",
        ),
        (["analyze", "--run", "/tmp/r", "--out", "/tmp/a"], "WP7"),
        (["audit", "make", "--run", "/tmp/r", "--out", "/tmp/a"], "WP8"),
        (["audit", "import", "--packet", "/tmp/p", "--labels", "/tmp/l"], "WP8"),
        (["audit", "outcomes", "--run", "/tmp/r", "--analysis", "/tmp/a", "--out", "/tmp/o"], "WP8"),
    ],
)
def test_dispatch_reaches_owner(argv, owner, tmp_path):
    proc = _run([arg.replace("/tmp", str(tmp_path / "missing")) for arg in argv])
    assert "Traceback" not in proc.stderr
    if "not implemented" in proc.stderr:
        assert proc.returncode == contracts.EXIT_CODES["configuration_or_contract"]
        assert proc.stderr.strip() == f"not implemented (owned by {owner})"
        assert proc.stdout == ""
    else:
        # Implemented commands must reject missing inputs cleanly.
        assert proc.returncode != 0


def test_unknown_choice_does_not_reach_stub():
    proc = _run([
        "run", "--prepare-dir", "/tmp/p", "--profile", "nope", "--conditions", "flat",
        "--server-manifest", "/tmp/m", "--out", "/tmp/r",
    ])
    assert proc.returncode == 2
    assert "not implemented" not in proc.stderr
    assert "invalid choice" in proc.stderr


def test_status_partitions_and_pins():
    assert contracts.EXIT_CODES == {
        "complete": 0,
        "paused": 75,
        "configuration_or_contract": 2,
    }
    assert set(contracts.TERMINAL_RECORD_STATUSES) | set(contracts.NONTERMINAL_RECORD_STATUSES) == set(contracts.RECORD_STATUSES)
    assert contracts.TERMINAL_RECORD_STATUSES.isdisjoint(contracts.NONTERMINAL_RECORD_STATUSES)
    assert contracts.NONTERMINAL_RECORD_STATUSES == frozenset({"pending_transport"})
    assert set(contracts.TERMINAL_ATTEMPT_OUTCOMES) | set(contracts.NONTERMINAL_ATTEMPT_OUTCOMES) == set(contracts.ATTEMPT_OUTCOMES)
    assert contracts.TERMINAL_ATTEMPT_OUTCOMES == frozenset({"ok", "truncated"})
    assert contracts.SEED == 0
    assert contracts.SCHEMA_VERSION == "ws-1"
    assert contracts.DATASET_SHA256 == "72a9c94699ca88f3eba79f330ab28965ee537bccd7cbf563265982a6ff8b74b4"
    assert contracts.INFERENCE_COLUMNS == ("id", "question_type", "question", "images")
    assert contracts.ANSWER_COLUMN not in contracts.INFERENCE_COLUMNS
    assert "gold" not in contracts.PREDICTION_COLUMNS
    assert "answer" not in contracts.PREDICTION_COLUMNS
    assert contracts.DEFAULT_DATASET == contracts.REPO_ROOT / "data" / "MMSI_Bench.parquet"
    assert contracts.DEFAULT_SPLIT == contracts.REPO_ROOT / "analyses" / "language" / "splits" / "language_explore_confirm.json"
    assert contracts.PREPARE_DIR == contracts.PACKAGE_DIR / "artifacts" / "prepare_v1"
    assert contracts.CACHE_DIR == contracts.RUNS_ROOT / "_cache"
    assert "synthetic10" in contracts.PROFILES
    assert "synthetic10" not in contracts.SELECTION_PROFILES
    lines = (contracts.REPO_ROOT / ".gitignore").read_text().splitlines()
    for entry in GITIGNORE_ENTRIES:
        assert entry in lines


def test_enums_and_limits_match_schema():
    schema = _schema()
    defs = schema["$defs"]
    enum_defs = {name for name, definition in defs.items() if "enum" in definition}
    assert enum_defs == set(ENUM_DEFS.values()) | {"relation_predicate"}
    for const, def_name in ENUM_DEFS.items():
        assert list(getattr(contracts, const)) == defs[def_name]["enum"]
    assert list(contracts.RELATION_PREDICATES) == list(RELATION_GROUP_ORDER)
    assert _relation_predicates() == defs["relation_predicate"]["enum"]
    pattern_defs = {name for name, definition in defs.items() if "pattern" in definition}
    assert pattern_defs == set(PATTERN_DEFS.values())
    for const, def_name in PATTERN_DEFS.items():
        assert getattr(contracts, const) == defs[def_name]["pattern"]
    assert schema["properties"]["schema_version"]["const"] == contracts.SCHEMA_VERSION
    for key, limit in contracts.MAX_COUNTS.items():
        assert schema["properties"][key]["maxItems"] == limit
    assert schema["properties"]["entities"]["items"]["properties"]["label"]["maxLength"] == contracts.MAX_LENGTHS["label"]
    assert schema["properties"]["observations"]["items"]["properties"]["description"]["maxLength"] == contracts.MAX_LENGTHS["description"]
    value_branches = schema["properties"]["claims"]["items"]["properties"]["value"]["anyOf"]
    string_branch = next(branch for branch in value_branches if branch.get("type") == "string")
    assert string_branch["maxLength"] == contracts.MAX_LENGTHS["value"]
    attribute_then = schema["properties"]["claims"]["items"]["allOf"][1]["then"]["properties"]["value"]
    assert attribute_then["maxLength"] == contracts.MAX_LENGTHS["value"]
    assert schema["properties"]["unknowns"]["items"]["maxLength"] == contracts.MAX_LENGTHS["unknown"]
    description = schema["description"]
    assert "WP2" in description
    assert "cross-reference resolution" in description
    assert "camera-per-image count" in description
    assert "cross_view required if and only if" in description
    assert "note required when predicate is other" in description
    image_body = contracts.IMAGE_ID_PATTERN[1:-1]
    camera_body = contracts.CAMERA_ID_PATTERN[1:-1]
    entity_body = contracts.ENTITY_ID_PATTERN[1:-1]
    assert f"image_plane:{image_body}" in contracts.FRAME_PATTERN
    assert f"camera:{camera_body}" in contracts.FRAME_PATTERN
    assert f"entity:{entity_body}" in contracts.FRAME_PATTERN
    assert contracts.ENTITY_OR_CAMERA_ID_PATTERN == f"^(?:{entity_body}|{camera_body})$"
    for sample in ("unknown", "image_plane:image_1", "image_plane:image_10", "camera:cam2", "entity:e10"):
        assert re.fullmatch(contracts.FRAME_PATTERN, sample)
    for sample in ("image_plane:image_0", "camera:cam0", "entity:e0", "world", "image_1"):
        assert re.fullmatch(contracts.FRAME_PATTERN, sample) is None


def test_fixtures_match_expectations():
    expectations = _expectations()
    names = {path.name for path in contracts.FIXTURES_DIR.iterdir() if path.name != "expectations.json"}
    assert names == set(expectations)
    validator = _validator()
    for name, expected in expectations.items():
        assert isinstance(expected["n_images"], int) and expected["n_images"] >= 1
        assert isinstance(expected["reason"], str) and expected["reason"]
        raw = (contracts.FIXTURES_DIR / name).read_text()
        try:
            document = json.loads(raw)
            parsed = True
        except json.JSONDecodeError:
            document = None
            parsed = False
        assert parsed is expected["json_parses"]
        if name.startswith("valid_"):
            assert expected["json_parses"] and expected["schema_valid"] and expected["semantically_valid"]
        if parsed:
            errors = list(validator.iter_errors(document))
            assert (not errors) is expected["schema_valid"]
        else:
            assert expected["schema_valid"] is False
            assert expected["semantically_valid"] is False
    assert expectations["invalid_over_limit.json"]["schema_valid"] is False
    assert expectations["invalid_unresolved_ref.json"]["schema_valid"] is True
    assert expectations["invalid_unresolved_ref.json"]["semantically_valid"] is False
    assert expectations["invalid_truncated.txt"]["json_parses"] is False


def test_valid_fixtures_follow_field_rules():
    expectations = _expectations()
    for name, expected in expectations.items():
        if not name.startswith("valid_"):
            continue
        assert_field_rules(_load_fixture(name), expected["n_images"])


def test_named_invalid_fixtures_break_field_rules():
    with pytest.raises(AssertionError, match="unresolved object e9"):
        assert_field_rules(_load_fixture("invalid_unresolved_ref.json"), 2)
    with pytest.raises(AssertionError, match="entity count"):
        assert_field_rules(_load_fixture("invalid_over_limit.json"), 2)


def test_schema_structural_rules():
    validator = _validator()
    base = _load_fixture("valid_full.json")
    assert list(validator.iter_errors(base)) == []

    def fails(mutate):
        document = copy.deepcopy(base)
        mutate(document)
        assert list(validator.iter_errors(document))

    fails(lambda doc: doc["entities"][0].__setitem__("label", "x" * 61))
    fails(lambda doc: doc["claims"][0].__setitem__("value", "x"))
    fails(lambda doc: doc["claims"][0].__setitem__("predicate", "color"))
    fails(lambda doc: doc["claims"][0].__setitem__("image_id", None))
    fails(lambda doc: doc["claims"][1].__setitem__("object", "e2"))
    fails(lambda doc: doc["claims"][1].__setitem__("value", None))
    fails(lambda doc: doc["query_frame"].__setitem__("frame", "world"))
    fails(lambda doc: doc["temporal_order"].__setitem__("order", ["image_1"]))
    fails(lambda doc: doc.__setitem__("extra", 1))
    fails(lambda doc: doc["entities"].extend(
        {"id": f"e{index}", "type": "object", "label": f"object {index}", "cross_view": None}
        for index in range(3, 10)
    ))

    supported = copy.deepcopy(base)
    supported["temporal_order"] = {
        "status": "supported",
        "order": ["image_2", "image_1"],
        "basis": "door state changes",
    }
    assert list(validator.iter_errors(supported)) == []
    # A supported order that is not a permutation of the input images is still
    # schema-valid. WP2 checks the permutation.
    not_a_permutation = copy.deepcopy(base)
    not_a_permutation["temporal_order"] = {
        "status": "supported",
        "order": ["image_9"],
        "basis": "not checked here",
    }
    assert list(validator.iter_errors(not_a_permutation)) == []
    unresolved = _load_fixture("invalid_unresolved_ref.json")
    assert list(validator.iter_errors(unresolved)) == []
