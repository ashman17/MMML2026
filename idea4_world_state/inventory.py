"""ws-1 parsing, semantic checks, and fact atoms.

`parse_inventory` accepts a bare JSON object or exactly one ``` fenced block
(optional language tag, no other surrounding text). It then applies
`inventory.schema.json` and the Section 8.3 semantic rules. The first failure
wins. `InventoryError.rule` is one of `ERROR_RULES`.

Atom tuples (all hashable), in `atom_sort_key` order:

- ``("query_frame", frame, basis | None)``
- ``("temporal_order", status, order | None, basis | None)``
- ``("camera", camera_id, image_id)``
- ``("entity", entity_id, type, label, cross_status | None, cross_uncertainty | None)``
- ``("observation", observation_id, entity_id, image_id, where, description)``
- ``("identity_link", link_id, entity_a, entity_b, status, uncertainty)``
- ``("claim", claim_id, kind, subject, predicate, object | None, value | None,
  frame, image_id | None, support, uncertainty, note | None)``
- ``("unknown", "uN", text)`` with ``u1..`` assigned by list position

`support` is a tuple of observation ids. `order` is a tuple of image ids.
`atom_text` is the flat sentence without a leading ``[id]`` bracket.
"""

from __future__ import annotations

import json
import re
from copy import deepcopy

from jsonschema import Draft202012Validator

from idea4_world_state import contracts

PARSER_POLICY = {
    "cross_view_repair": {
        "when": "same entity id is observed in two or more images and cross_view is null",
        "field": "cross_view",
        "status": "same",
        "uncertainty": "medium",
        "uncertainty_source": "parser_default",
    }
}


def parser_policy() -> dict:
    """The parser behavior a freeze must record and later runs must match."""
    return deepcopy(PARSER_POLICY)


ERROR_RULES = (
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

_KIND_RANK = {
    "query_frame": 0,
    "temporal_order": 1,
    "camera": 2,
    "entity": 3,
    "observation": 4,
    "identity_link": 5,
    "claim": 6,
    "unknown": 7,
}
_CAMERA_MOTION = frozenset(contracts.RELATION_PREDICATES["camera_motion"])
_OBJECT_MOTION = frozenset(contracts.RELATION_PREDICATES["object_motion"])
_OPENER = re.compile(r"```[^\n`]*\Z")
_NUMBERED_ID = re.compile(r"[a-z]+(\d+)\Z")


class InventoryError(Exception):
    """A ws-1 response violated a parse, schema, or semantic rule.

    `rule` is a stable token from `ERROR_RULES`. `message` describes this
    occurrence. ``str(exc)`` is ``"{rule}: {message}"``.
    """

    def __init__(self, rule: str, message: str) -> None:
        self.rule = rule
        self.message = message
        super().__init__(f"{rule}: {message}")


def _quote(text: str) -> str:
    return json.dumps(text, ensure_ascii=False)


def _needs_quotes(text: str) -> bool:
    if text == "" or text != text.strip():
        return True
    if "|" in text or ")" in text:
        return True
    return _quote(text) != f'"{text}"'


def _quote_if_needed(text: str) -> str:
    if _needs_quotes(text):
        return _quote(text)
    return text


def _query_frame_atom(frame: str, basis: str | None) -> tuple:
    return ("query_frame", frame, basis)


def _temporal_atom(status: str, order, basis: str | None) -> tuple:
    if order is not None:
        order = tuple(order)
    return ("temporal_order", status, order, basis)


def _camera_atom(camera_id: str, image_id: str) -> tuple:
    return ("camera", camera_id, image_id)


def _entity_atom(
    entity_id: str,
    entity_type: str,
    label: str,
    cross_status: str | None,
    cross_uncertainty: str | None,
) -> tuple:
    return ("entity", entity_id, entity_type, label, cross_status, cross_uncertainty)


def _observation_atom(
    observation_id: str,
    entity_id: str,
    image_id: str,
    where: str,
    description: str,
) -> tuple:
    return ("observation", observation_id, entity_id, image_id, where, description)


def _link_atom(
    link_id: str,
    left: str,
    right: str,
    status: str,
    uncertainty: str,
) -> tuple:
    return ("identity_link", link_id, left, right, status, uncertainty)


def _claim_atom(
    claim_id: str,
    kind: str,
    subject: str,
    predicate: str,
    obj: str | None,
    value: str | None,
    frame: str,
    image_id: str | None,
    support,
    uncertainty: str,
    note: str | None,
) -> tuple:
    return (
        "claim",
        claim_id,
        kind,
        subject,
        predicate,
        obj,
        value,
        frame,
        image_id,
        tuple(support),
        uncertainty,
        note,
    )


def _unknown_atom(unknown_id: str, text: str) -> tuple:
    return ("unknown", unknown_id, text)


def atom_id(atom: tuple) -> str:
    """Stable id: `query_frame`, `temporal_order`, `cam1`, `e1`, `o1`, `i1`, `c1`, `u1`."""
    kind = atom[0]
    if kind in ("query_frame", "temporal_order"):
        return kind
    if kind in ("camera", "entity", "observation", "identity_link", "claim", "unknown"):
        return atom[1]
    raise ValueError(f"unknown atom kind {kind!r}")


def atom_sort_key(atom: tuple) -> tuple[int, int, str]:
    """Audit order A1.. and the order both serializers emit.

    Query frame, temporal order, cameras, entities, observations, identity
    links, claims, then unknowns. Numbered ids sort numerically (`cam2` before
    `cam10`).
    """
    ident = atom_id(atom)
    match = _NUMBERED_ID.fullmatch(ident)
    number = int(match.group(1)) if match else 0
    return (_KIND_RANK[atom[0]], number, ident)


def _claim_text(atom: tuple) -> str:
    _, _, kind, subject, predicate, obj, value, frame, image_id, support, uncertainty, note = atom
    if kind == "relation":
        action = f"{subject} is {predicate} {obj}"
    else:
        action = f"attribute {predicate} of {subject} is {_quote(value)}"
    if image_id is None and frame == "unknown":
        lead = f"Across views, {action}."
    elif image_id is None:
        lead = f"Across views, in frame {frame}, {action}."
    else:
        lead = f"In frame {frame}, for {image_id}, {action}."
    text = f"{lead} Support: {', '.join(support)}. Uncertainty: {uncertainty}."
    if note is not None:
        text += f" Note: {_quote(note)}."
    return text


def atom_text(atom: tuple) -> str:
    """One flat sentence for the atom, without a leading `[id]` bracket."""
    kind = atom[0]
    if kind == "query_frame":
        _, frame, basis = atom
        if basis is None:
            return f"Query frame: {frame}."
        return f"Query frame: {frame} (basis: {_quote_if_needed(basis)})."
    if kind == "temporal_order":
        _, status, order, basis = atom
        if status == "unknown":
            return "Temporal order: unknown."
        return f"Temporal order: {', '.join(order)} (basis: {_quote_if_needed(basis)})."
    if kind == "camera":
        _, camera_id, image_id = atom
        return f"Camera {camera_id} took {image_id}."
    if kind == "entity":
        _, entity_id, entity_type, label, status, uncertainty = atom
        article = "an object" if entity_type == "object" else f"a {entity_type}"
        quoted = _quote(label)
        if status is None:
            return f"{entity_id} is {article} labeled {quoted}."
        return (
            f"{entity_id} is {article} labeled {quoted}; "
            f"it is the same entity across views (uncertainty {uncertainty})."
        )
    if kind == "observation":
        _, _, entity_id, image_id, where, description = atom
        return f"In {image_id} ({where}), {entity_id} is observed: {_quote(description)}."
    if kind == "identity_link":
        _, _, left, right, status, uncertainty = atom
        return (
            f"Identity link between {left} and {right} is {status} "
            f"(uncertainty {uncertainty})."
        )
    if kind == "claim":
        return _claim_text(atom)
    if kind == "unknown":
        return f"Unknown: {_quote_if_needed(atom[2])}."
    raise ValueError(f"unknown atom kind {kind!r}")


def atoms(inv: dict) -> frozenset:
    """Fact atoms of a ws-1 inventory. Unknowns are numbered `u1..` by position."""
    found = [
        _query_frame_atom(inv["query_frame"]["frame"], inv["query_frame"]["basis"]),
        _temporal_atom(
            inv["temporal_order"]["status"],
            inv["temporal_order"]["order"],
            inv["temporal_order"]["basis"],
        ),
    ]
    for camera in inv["cameras"]:
        found.append(_camera_atom(camera["id"], camera["image_id"]))
    for entity in inv["entities"]:
        cross = entity["cross_view"]
        if cross is None:
            status, uncertainty = None, None
        else:
            status, uncertainty = cross["status"], cross["uncertainty"]
        found.append(_entity_atom(entity["id"], entity["type"], entity["label"], status, uncertainty))
    for obs in inv["observations"]:
        found.append(
            _observation_atom(obs["id"], obs["entity"], obs["image_id"], obs["where"], obs["description"])
        )
    for link in inv["identity_links"]:
        left, right = link["entities"]
        found.append(_link_atom(link["id"], left, right, link["status"], link["uncertainty"]))
    for claim in inv["claims"]:
        found.append(
            _claim_atom(
                claim["id"],
                claim["kind"],
                claim["subject"],
                claim["predicate"],
                claim["object"],
                claim["value"],
                claim["frame"],
                claim["image_id"],
                claim["support"],
                claim["uncertainty"],
                claim["note"],
            )
        )
    for index, text in enumerate(inv["unknowns"], start=1):
        found.append(_unknown_atom(f"u{index}", text))
    return frozenset(found)


_VALIDATOR = None


def _validator() -> Draft202012Validator:
    global _VALIDATOR
    if _VALIDATOR is None:
        schema = json.loads(contracts.SCHEMA_PATH.read_text(encoding="utf-8"))
        _VALIDATOR = Draft202012Validator(schema)
    return _VALIDATOR


def _schema_error(document: dict) -> InventoryError | None:
    errors = sorted(
        _validator().iter_errors(document),
        key=lambda error: ([str(part) for part in error.absolute_path], error.message),
    )
    if not errors:
        return None
    error = errors[0]
    path = "/".join(str(part) for part in error.absolute_path) or "(root)"
    return InventoryError("schema", f"{path}: {error.message}")


def _require_unique(items: list, label: str) -> None:
    seen: set[str] = set()
    for item in items:
        ident = item["id"]
        if ident in seen:
            raise InventoryError("duplicate_id", f"duplicate {label} id {ident}")
        seen.add(ident)


def _check_frame(frame: str, where: str, camera_ids: set[str], entity_ids: set[str], allowed: set[str]) -> None:
    if frame == "unknown":
        return
    kind, _, ident = frame.partition(":")
    if kind == "image_plane" and ident in allowed:
        return
    if kind == "camera" and ident in camera_ids:
        return
    if kind == "entity" and ident in entity_ids:
        return
    raise InventoryError("unresolved_ref", f"{where} frame {frame} does not resolve")


def _requires_image_id(claim: dict) -> bool:
    predicate = claim["predicate"]
    if claim["kind"] == "relation" and predicate in _CAMERA_MOTION | _OBJECT_MOTION:
        return True
    return claim["kind"] == "attribute" and predicate == "state"


def _check_motion(claim: dict, entity_ids: set[str], camera_ids: set[str]) -> None:
    if claim["kind"] != "relation":
        return
    predicate = claim["predicate"]
    subject = claim["subject"]
    obj = claim["object"]
    cid = claim["id"]
    if predicate in _CAMERA_MOTION:
        if subject not in camera_ids or obj not in camera_ids or subject == obj:
            raise InventoryError(
                "motion_endpoints",
                f"claim {cid} camera motion requires two different cameras, found {subject} -> {obj}",
            )
    elif predicate in _OBJECT_MOTION:
        obj_ok = obj in camera_ids or (obj in entity_ids and obj != subject)
        if subject not in entity_ids or not obj_ok:
            raise InventoryError(
                "motion_endpoints",
                (
                    f"claim {cid} object motion requires an entity subject and a camera "
                    f"or different entity, found {subject} -> {obj}"
                ),
            )


def _check_semantics(inv: dict, n_images: int) -> list:
    repairs: list = []
    if isinstance(n_images, bool) or not isinstance(n_images, int) or n_images < 1:
        raise InventoryError("camera_count", f"n_images must be a positive integer, found {n_images!r}")
    cameras = inv["cameras"]
    entities = inv["entities"]
    observations = inv["observations"]
    links = inv["identity_links"]
    claims = inv["claims"]
    if len(cameras) != n_images:
        raise InventoryError("camera_count", f"expected {n_images} cameras, found {len(cameras)}")
    for index, camera in enumerate(cameras, start=1):
        expected_id = f"cam{index}"
        expected_image = f"image_{index}"
        if camera["id"] != expected_id or camera["image_id"] != expected_image:
            raise InventoryError(
                "camera_mapping",
                (
                    f"camera {index} must be {expected_id} for {expected_image}, "
                    f"found {camera['id']} for {camera['image_id']}"
                ),
            )
    _require_unique(cameras, "camera")
    _require_unique(entities, "entity")
    _require_unique(observations, "observation")
    _require_unique(links, "identity link")
    _require_unique(claims, "claim")

    allowed = {f"image_{index}" for index in range(1, n_images + 1)}
    for obs in observations:
        if obs["image_id"] not in allowed:
            raise InventoryError(
                "image_range",
                f"observation {obs['id']} image_id {obs['image_id']} is outside 1..{n_images}",
            )
    for claim in claims:
        image_id = claim["image_id"]
        if image_id is not None and image_id not in allowed:
            raise InventoryError(
                "image_range",
                f"claim {claim['id']} image_id {image_id} is outside 1..{n_images}",
            )

    camera_ids = {camera["id"] for camera in cameras}
    entity_ids = {entity["id"] for entity in entities}
    observation_ids = {obs["id"] for obs in observations}
    nodes = entity_ids | camera_ids
    for obs in observations:
        if obs["entity"] not in nodes:
            raise InventoryError(
                "unresolved_ref",
                f"observation {obs['id']} entity {obs['entity']} does not resolve",
            )
    for link in links:
        for entity_id in link["entities"]:
            if entity_id not in entity_ids:
                raise InventoryError(
                    "unresolved_ref",
                    f"identity link {link['id']} entity {entity_id} does not resolve",
                )
    for claim in claims:
        cid = claim["id"]
        if claim["subject"] not in nodes:
            raise InventoryError("unresolved_ref", f"claim {cid} subject {claim['subject']} does not resolve")
        if claim["kind"] == "relation" and claim["object"] not in nodes:
            raise InventoryError("unresolved_ref", f"claim {cid} object {claim['object']} does not resolve")
        for support_id in claim["support"]:
            if support_id not in observation_ids:
                raise InventoryError(
                    "unresolved_ref",
                    f"claim {cid} support {support_id} does not resolve",
                )
        _check_frame(claim["frame"], f"claim {cid}", camera_ids, entity_ids, allowed)
    _check_frame(inv["query_frame"]["frame"], "query_frame", camera_ids, entity_ids, allowed)

    images_by_entity: dict[str, set[str]] = {}
    for obs in observations:
        images_by_entity.setdefault(obs["entity"], set()).add(obs["image_id"])
    for entity in entities:
        seen = images_by_entity.get(entity["id"], set())
        if len(seen) >= 2 and entity["cross_view"] is None:
            # Reusing one entity id in two images is the identity assertion.
            # The pilot model often left this redundant object null.
            applied = {
                "status": PARSER_POLICY["cross_view_repair"]["status"],
                "uncertainty": PARSER_POLICY["cross_view_repair"]["uncertainty"],
            }
            entity["cross_view"] = applied
            repairs.append({
                "entity_id": entity["id"],
                "field": "cross_view",
                "applied": dict(applied),
                "uncertainty_source": PARSER_POLICY["cross_view_repair"]["uncertainty_source"],
            })
        if len(seen) < 2 and entity["cross_view"] is not None:
            raise InventoryError(
                "cross_view",
                f"entity {entity['id']} is observed in {len(seen)} images but cross_view is set",
            )

    for claim in claims:
        frame = claim["frame"]
        if frame.startswith("image_plane:"):
            expected = frame.split(":", 1)[1]
            if claim["image_id"] != expected:
                raise InventoryError(
                    "image_id",
                    (
                        f"claim {claim['id']} frame {frame} requires image_id {expected}, "
                        f"found {claim['image_id']}"
                    ),
                )
        elif _requires_image_id(claim) and claim["image_id"] is None:
            raise InventoryError(
                "image_id",
                f"claim {claim['id']} predicate {claim['predicate']} requires image_id",
            )
        _check_motion(claim, entity_ids, camera_ids)
        if claim["predicate"] == "other" and not isinstance(claim["note"], str):
            raise InventoryError("note", f"claim {claim['id']} predicate other requires note")

    temporal = inv["temporal_order"]
    if temporal["status"] == "supported":
        expected_order = [f"image_{index}" for index in range(1, n_images + 1)]
        order = list(temporal["order"])
        if len(order) != len(expected_order) or sorted(order) != sorted(expected_order):
            raise InventoryError(
                "temporal_order",
                f"supported order is not a permutation of {', '.join(expected_order)}",
            )
    return repairs


def _decode_object(payload: str) -> dict:
    try:
        value, end = json.JSONDecoder().raw_decode(payload)
    except json.JSONDecodeError as exc:
        raise InventoryError("json", f"invalid JSON: {exc.msg}") from exc
    if payload[end:].strip():
        raise InventoryError("format", "surrounding text after the JSON object")
    if not isinstance(value, dict):
        raise InventoryError("not_object", f"JSON value is not an object ({type(value).__name__})")
    return value


def _payload_from_fence(text: str) -> str:
    first_nl = text.find("\n")
    if first_nl == -1:
        raise InventoryError("format", "malformed fenced code block")
    opener = text[:first_nl].rstrip("\r")
    if _OPENER.fullmatch(opener) is None:
        raise InventoryError("format", "malformed fenced code block")
    if text.endswith("\n```"):
        body = text[first_nl + 1:-4]
    elif text.endswith("```"):
        body = text[first_nl + 1:-3]
    else:
        raise InventoryError("format", "expected exactly one fenced code block with no surrounding text")
    if re.search(r"(?m)^```", body):
        raise InventoryError("format", "expected exactly one fenced code block")
    return body.strip()


def _extract_document(text: str) -> dict:
    if not isinstance(text, str):
        raise InventoryError("format", "inventory response must be text")
    stripped = text.strip().lstrip("\ufeff")
    if not stripped:
        raise InventoryError("format", "empty inventory response")
    if stripped.startswith("```"):
        payload = _payload_from_fence(stripped)
    elif _looks_like_json_value(stripped):
        payload = stripped
    else:
        raise InventoryError("format", "response is not a bare JSON object or one fenced code block")
    return _decode_object(payload)


def _looks_like_json_value(text: str) -> bool:
    if text[0] in '{["-' or text[0].isdigit():
        return True
    return text in ("null", "true", "false")


def parse_inventory(text: str, n_images: int) -> dict:
    """Parse and validate a model inventory response.

    Accepts a bare JSON object, or exactly one fenced code block with an
    optional language tag. Any other surrounding text fails. Schema validation
    runs before the semantic checks. A shared entity id with null ``cross_view``
    is completed from ``parser_policy``; ``parse_inventory_with_repairs`` lists
    those completions.
    """
    document, _repairs = parse_inventory_with_repairs(text, n_images)
    return document


def parse_inventory_with_repairs(text: str, n_images: int) -> tuple[dict, list]:
    """Return ``(inventory, repairs)``.

    Each repair names the entity whose ``cross_view`` was filled. ``medium``
    uncertainty in that object is ``parser_default``, not a model judgment.
    """
    document = _extract_document(text)
    schema_error = _schema_error(document)
    if schema_error is not None:
        raise schema_error
    repairs = _check_semantics(document, n_images)
    return document, repairs
