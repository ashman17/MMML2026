"""Flat and graph serializations of a ws-1 inventory.

Both renderers are deterministic functions of `inventory.atoms`. They emit
only those atoms: no inverse or transitive claims. Quoted strings use
`json.dumps` escaping. Plain basis strings and unknowns stay unquoted so the
Section 8.4 examples match; a string is quoted when it needs JSON escaping or
contains `|` or `)`.

Across-view attributes whose frame is `unknown` omit the frame, matching 8.4.
The parser treats that omission as frame `unknown`. Relation edges always name
their frame, including `unknown`. Attribute lines nest under the subject;
relation lines are grouped under "Edges by frame".
"""

from __future__ import annotations

import json
import re

from idea4_world_state.inventory import (
    _camera_atom,
    _claim_atom,
    _entity_atom,
    _link_atom,
    _observation_atom,
    _query_frame_atom,
    _quote,
    _quote_if_needed,
    _temporal_atom,
    _unknown_atom,
    atom_id,
    atom_sort_key,
    atom_text,
    atoms,
)

FLAT_HEADER = "Evidence extracted by a model from the images (may be incomplete or wrong):"
GRAPH_HEADER = "World state graph extracted by a model from the images (may be incomplete or wrong):"

_KINDS = (
    "query_frame",
    "temporal_order",
    "camera",
    "entity",
    "observation",
    "identity_link",
    "claim",
    "unknown",
)
_FLAT_SUPPORT = re.compile(
    r"(o[1-9]\d*(?:, o[1-9]\d*)*)\. Uncertainty: (low|medium|high)\."
)
_GRAPH_SUPPORT = re.compile(r"o[1-9]\d*(?:,o[1-9]\d*)*")
_IMAGE_ID = re.compile(r"image_[1-9]\d*")
_ORDER = re.compile(r"image_[1-9]\d*(?:, image_[1-9]\d*)*")


def _grouped(atom_set: frozenset) -> dict[str, list]:
    grouped = {kind: [] for kind in _KINDS}
    for atom in sorted(atom_set, key=atom_sort_key):
        grouped[atom[0]].append(atom)
    return grouped


def _only(grouped: dict[str, list], kind: str) -> tuple:
    items = grouped[kind]
    if len(items) != 1:
        raise ValueError(f"expected one {kind} atom, found {len(items)}")
    return items[0]


def _finish(lines: list[str]) -> str:
    return "\n".join(lines) + "\n"


def _graph_query(atom: tuple) -> str:
    _, frame, basis = atom
    if basis is None:
        return f"Query frame: {frame}"
    return f"Query frame: {frame} | basis: {_quote_if_needed(basis)}"


def _graph_temporal(atom: tuple) -> str:
    _, status, order, basis = atom
    if status == "unknown":
        return "Temporal order: unknown"
    return f"Temporal order: {', '.join(order)} | basis: {_quote_if_needed(basis)}"


def _graph_camera(atom: tuple) -> str:
    _, camera_id, image_id = atom
    return f"{camera_id} [camera] {image_id}"


def _graph_entity(atom: tuple) -> str:
    _, entity_id, entity_type, label, status, uncertainty = atom
    line = f"{entity_id} [{entity_type}] {_quote(label)}"
    if status is not None:
        line += f" | cross-view: {status} (uncertainty {uncertainty})"
    return line


def _graph_observation(atom: tuple) -> str:
    _, observation_id, _, image_id, where, description = atom
    return f"{observation_id} @{image_id} {where}: {_quote(description)}"


def _graph_attribute(atom: tuple) -> str:
    _, claim_id, _, _, predicate, _, value, frame, image_id, support, uncertainty, note = atom
    loc = "across views" if image_id is None else image_id
    parts = [f"{predicate} = {_quote(value)}", loc]
    if frame != "unknown":
        parts.append(f"frame {frame}")
    parts.append("support " + ",".join(support))
    parts.append(f"uncertainty {uncertainty}")
    if note is not None:
        parts.append(f"note {_quote(note)}")
    return f"{claim_id} " + " | ".join(parts)


def _graph_relation(atom: tuple) -> str:
    _, claim_id, _, subject, predicate, obj, _, _, image_id, support, uncertainty, note = atom
    loc = "across views" if image_id is None else image_id
    line = (
        f"{claim_id} {subject} -{predicate}-> {obj} | {loc} | "
        f"support {','.join(support)} | uncertainty {uncertainty}"
    )
    if note is not None:
        line += f" | note {_quote(note)}"
    return line


def _graph_link(atom: tuple) -> str:
    _, link_id, left, right, status, uncertainty = atom
    return f"{link_id} {left} ~ {right} | {status} | uncertainty {uncertainty}"


def _graph_unknown(atom: tuple) -> str:
    return f"{atom[1]} {_quote_if_needed(atom[2])}"


def render_flat(inv: dict) -> str:
    """Render the flat evidence list. Ends with a single newline."""
    grouped = _grouped(atoms(inv))
    lines = [
        FLAT_HEADER,
        atom_text(_only(grouped, "query_frame")),
        atom_text(_only(grouped, "temporal_order")),
        " ".join(atom_text(atom) for atom in grouped["camera"]),
    ]
    for kind in ("entity", "observation", "identity_link", "claim", "unknown"):
        for atom in grouped[kind]:
            lines.append(f"[{atom_id(atom)}] {atom_text(atom)}")
    return _finish(lines)


def render_graph(inv: dict) -> str:
    """Render the entity/relation graph. Ends with a single newline."""
    grouped = _grouped(atoms(inv))
    lines = [
        GRAPH_HEADER,
        _graph_query(_only(grouped, "query_frame")),
        _graph_temporal(_only(grouped, "temporal_order")),
        "Nodes:",
    ]
    observations: dict[str, list] = {}
    for atom in grouped["observation"]:
        observations.setdefault(atom[2], []).append(atom)
    attributes: dict[str, list] = {}
    for atom in grouped["claim"]:
        if atom[2] == "attribute":
            attributes.setdefault(atom[3], []).append(atom)

    def emit(node_id: str, node_line: str) -> None:
        lines.append(f"  {node_line}")
        for obs in observations.get(node_id, ()):
            lines.append(f"    {_graph_observation(obs)}")
        for claim in attributes.get(node_id, ()):
            lines.append(f"    {_graph_attribute(claim)}")

    for camera in grouped["camera"]:
        emit(camera[1], _graph_camera(camera))
    for entity in grouped["entity"]:
        emit(entity[1], _graph_entity(entity))

    if grouped["identity_link"]:
        lines.append("Identity links:")
        for link in grouped["identity_link"]:
            lines.append(f"  {_graph_link(link)}")
    else:
        lines.append("Identity links: none")

    relations = [atom for atom in grouped["claim"] if atom[2] == "relation"]
    if relations:
        lines.append("Edges by frame:")
        order: list[str] = []
        by_frame: dict[str, list] = {}
        for atom in relations:
            frame = atom[7]
            if frame not in by_frame:
                order.append(frame)
                by_frame[frame] = []
            by_frame[frame].append(atom)
        for frame in order:
            lines.append(f"  {frame}")
            for atom in by_frame[frame]:
                lines.append(f"    {_graph_relation(atom)}")
    else:
        lines.append("Edges by frame: none")

    if grouped["unknown"]:
        lines.append("Unknowns:")
        for atom in grouped["unknown"]:
            lines.append(f"  {_graph_unknown(atom)}")
    else:
        lines.append("Unknowns: none")
    return _finish(lines)


def _read_json_string(text: str, index: int) -> tuple[str, int]:
    if index >= len(text) or text[index] != '"':
        raise ValueError(f"expected quoted string in {text!r}")
    end = index + 1
    while end < len(text):
        if text[end] == "\\":
            end += 2
            continue
        if text[end] == '"':
            raw = text[index:end + 1]
            try:
                value = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid quoted string {raw!r}") from exc
            return value, end + 1
        end += 1
    raise ValueError(f"unterminated string in {text!r}")


def _parse_exact_json_string(raw: str) -> str:
    value, end = _read_json_string(raw, 0)
    if end != len(raw):
        raise ValueError(raw)
    return value


def _parse_maybe_quoted(raw: str) -> str:
    if raw.startswith('"'):
        return _parse_exact_json_string(raw)
    return raw


def _split_flat_note(line: str) -> tuple[str, str | None]:
    marker = ". Note: "
    found = None
    start = 0
    while True:
        pos = line.find(marker, start)
        if pos < 0:
            break
        raw = line[pos + len(marker):]
        if raw.endswith("."):
            try:
                value = _parse_exact_json_string(raw[:-1])
            except ValueError:
                value = None
            if value is not None:
                found = (line[:pos] + ".", value)
        start = pos + len(marker)
    if found is None:
        return line, None
    return found


def _split_flat_support(line: str) -> tuple[str, tuple[str, ...], str]:
    marker = " Support: "
    found = None
    start = 0
    while True:
        pos = line.find(marker, start)
        if pos < 0:
            break
        matched = _FLAT_SUPPORT.fullmatch(line[pos + len(marker):])
        if matched:
            found = (line[:pos], tuple(matched.group(1).split(", ")), matched.group(2))
        start = pos + len(marker)
    if found is None:
        raise ValueError(line)
    return found


def _parse_flat_lead(lead: str) -> tuple:
    attribute_patterns = (
        re.compile(
            r"In frame (?P<frame>\S+), for (?P<image>image_[1-9]\d*), "
            r"attribute (?P<pred>[a-z_]+) of (?P<subj>\S+) is "
        ),
        re.compile(
            r"Across views, in frame (?P<frame>\S+), "
            r"attribute (?P<pred>[a-z_]+) of (?P<subj>\S+) is "
        ),
        re.compile(r"Across views, attribute (?P<pred>[a-z_]+) of (?P<subj>\S+) is "),
    )
    for pattern in attribute_patterns:
        match = pattern.match(lead)
        if not match:
            continue
        try:
            value, end = _read_json_string(lead, match.end())
        except ValueError:
            continue
        if lead[end:] != ".":
            continue
        frame = match.group("frame") if "frame" in pattern.groupindex else "unknown"
        image_id = match.group("image") if "image" in pattern.groupindex else None
        return ("attribute", match["subj"], match["pred"], None, value, frame, image_id)
    relation_patterns = (
        re.compile(
            r"In frame (?P<frame>\S+), for (?P<image>image_[1-9]\d*), "
            r"(?P<subj>\S+) is (?P<pred>[a-z_]+) (?P<obj>\S+)\."
        ),
        re.compile(
            r"Across views, in frame (?P<frame>\S+), "
            r"(?P<subj>\S+) is (?P<pred>[a-z_]+) (?P<obj>\S+)\."
        ),
        re.compile(r"Across views, (?P<subj>\S+) is (?P<pred>[a-z_]+) (?P<obj>\S+)\."),
    )
    for pattern in relation_patterns:
        match = pattern.fullmatch(lead)
        if not match:
            continue
        frame = match.group("frame") if "frame" in pattern.groupindex else "unknown"
        image_id = match.group("image") if "image" in pattern.groupindex else None
        return ("relation", match["subj"], match["pred"], match["obj"], None, frame, image_id)
    raise ValueError(lead)


def _parse_flat_query(line: str) -> tuple:
    match = re.fullmatch(r"Query frame: (\S+)(?: \(basis: (.*)\))?\.", line)
    if not match:
        raise ValueError(line)
    basis = None if match.group(2) is None else _parse_maybe_quoted(match.group(2))
    return _query_frame_atom(match.group(1), basis)


def _parse_flat_temporal(line: str) -> tuple:
    if line == "Temporal order: unknown.":
        return _temporal_atom("unknown", None, None)
    match = re.fullmatch(
        r"Temporal order: (image_[1-9]\d*(?:, image_[1-9]\d*)*) \(basis: (.*)\)\.",
        line,
    )
    if not match:
        raise ValueError(line)
    return _temporal_atom("supported", tuple(match.group(1).split(", ")), _parse_maybe_quoted(match.group(2)))


def _parse_flat_cameras(line: str) -> list[tuple]:
    found = list(re.finditer(r"Camera (cam\d+) took (image_\d+)\.", line))
    if not found:
        raise ValueError(line)
    rebuilt = " ".join(f"Camera {item.group(1)} took {item.group(2)}." for item in found)
    if rebuilt != line:
        raise ValueError(line)
    return [_camera_atom(item.group(1), item.group(2)) for item in found]


def _parse_flat_entity(line: str) -> tuple:
    match = re.match(r"\[(e\d+)\] (e\d+) is (an object|a region) labeled ", line)
    if not match or match.group(1) != match.group(2):
        raise ValueError(line)
    label, end = _read_json_string(line, match.end())
    tail = line[end:]
    entity_type = "object" if match.group(3) == "an object" else "region"
    if tail == ".":
        return _entity_atom(match.group(1), entity_type, label, None, None)
    cross = re.fullmatch(
        r"; it is the same entity across views \(uncertainty (low|medium|high)\)\.",
        tail,
    )
    if not cross:
        raise ValueError(line)
    return _entity_atom(match.group(1), entity_type, label, "same", cross.group(1))


def _parse_flat_observation(line: str) -> tuple:
    match = re.match(r"\[(o\d+)\] In (image_\d+) \(([a-z_]+)\), (\S+) is observed: ", line)
    if not match:
        raise ValueError(line)
    description, end = _read_json_string(line, match.end())
    if line[end:] != ".":
        raise ValueError(line)
    return _observation_atom(match.group(1), match.group(4), match.group(2), match.group(3), description)


def _parse_flat_link(line: str) -> tuple:
    match = re.fullmatch(
        r"\[(i\d+)\] Identity link between (\S+) and (\S+) is (tentative|different) "
        r"\(uncertainty (low|medium|high)\)\.",
        line,
    )
    if not match:
        raise ValueError(line)
    return _link_atom(match.group(1), match.group(2), match.group(3), match.group(4), match.group(5))


def _parse_flat_claim(line: str) -> tuple:
    match = re.match(r"\[(c\d+)\] ", line)
    if not match:
        raise ValueError(line)
    body, note = _split_flat_note(line[match.end():])
    lead, support, uncertainty = _split_flat_support(body)
    kind, subject, predicate, obj, value, frame, image_id = _parse_flat_lead(lead)
    return _claim_atom(
        match.group(1),
        kind,
        subject,
        predicate,
        obj,
        value,
        frame,
        image_id,
        support,
        uncertainty,
        note,
    )


def _parse_flat_unknown(line: str) -> tuple:
    match = re.match(r"\[(u\d+)\] Unknown: ", line)
    if not match:
        raise ValueError(line)
    raw = line[match.end():]
    if not raw.endswith("."):
        raise ValueError(line)
    payload = raw[:-1]
    text = _parse_maybe_quoted(payload) if payload.startswith('"') else payload
    return _unknown_atom(match.group(1), text)


def parse_flat(text: str) -> frozenset:
    """Parse a flat serialization back into fact atoms."""
    lines = text.splitlines()
    if len(lines) < 4 or lines[0] != FLAT_HEADER:
        raise ValueError("flat serialization header missing")
    found = [
        _parse_flat_query(lines[1]),
        _parse_flat_temporal(lines[2]),
        *_parse_flat_cameras(lines[3]),
    ]
    for line in lines[4:]:
        if line.startswith("[e"):
            found.append(_parse_flat_entity(line))
        elif line.startswith("[o"):
            found.append(_parse_flat_observation(line))
        elif line.startswith("[i"):
            found.append(_parse_flat_link(line))
        elif line.startswith("[c"):
            found.append(_parse_flat_claim(line))
        elif line.startswith("[u"):
            found.append(_parse_flat_unknown(line))
        else:
            raise ValueError(f"unrecognized flat line: {line}")
    return frozenset(found)


def _parse_location(loc: str) -> str | None:
    if loc == "across views":
        return None
    if _IMAGE_ID.fullmatch(loc):
        return loc
    raise ValueError(loc)


def _parse_support_token(part: str) -> tuple[str, ...]:
    if not part.startswith("support "):
        raise ValueError(part)
    ids = part[len("support "):]
    if not _GRAPH_SUPPORT.fullmatch(ids):
        raise ValueError(part)
    return tuple(ids.split(","))


def _parse_uncertainty_token(part: str) -> str:
    match = re.fullmatch(r"uncertainty (low|medium|high)", part)
    if not match:
        raise ValueError(part)
    return match.group(1)


def _split_graph_note(tail: str) -> tuple[str, str | None]:
    marker = " | note "
    if marker not in tail:
        return tail, None
    head, raw = tail.split(marker, 1)
    return head, _parse_exact_json_string(raw)


def _parse_attribute_tail(tail: str) -> tuple:
    tail, note = _split_graph_note(tail)
    if not tail.startswith(" | "):
        raise ValueError(tail)
    parts = tail[3:].split(" | ")
    image_id = _parse_location(parts[0])
    index = 1
    frame = "unknown"
    if index < len(parts) and parts[index].startswith("frame "):
        frame = parts[index][len("frame "):]
        index += 1
    if index + 2 != len(parts):
        raise ValueError(tail)
    support = _parse_support_token(parts[index])
    uncertainty = _parse_uncertainty_token(parts[index + 1])
    return image_id, frame, support, uncertainty, note


def _parse_relation_tail(tail: str) -> tuple:
    tail, note = _split_graph_note(tail)
    if not tail.startswith(" | "):
        raise ValueError(tail)
    parts = tail[3:].split(" | ")
    if len(parts) != 3:
        raise ValueError(tail)
    return (
        _parse_location(parts[0]),
        _parse_support_token(parts[1]),
        _parse_uncertainty_token(parts[2]),
        note,
    )


def _parse_graph_query(line: str) -> tuple:
    prefix = "Query frame: "
    if not line.startswith(prefix):
        raise ValueError(line)
    rest = line[len(prefix):]
    marker = " | basis: "
    if marker in rest:
        frame, raw = rest.split(marker, 1)
        basis = _parse_maybe_quoted(raw)
    else:
        frame, basis = rest, None
    if not frame or any(char.isspace() for char in frame):
        raise ValueError(line)
    return _query_frame_atom(frame, basis)


def _parse_graph_temporal(line: str) -> tuple:
    if line == "Temporal order: unknown":
        return _temporal_atom("unknown", None, None)
    marker = " | basis: "
    if marker not in line:
        raise ValueError(line)
    left, raw = line.split(marker, 1)
    prefix = "Temporal order: "
    if not left.startswith(prefix):
        raise ValueError(line)
    order_text = left[len(prefix):]
    if not _ORDER.fullmatch(order_text):
        raise ValueError(line)
    return _temporal_atom("supported", tuple(order_text.split(", ")), _parse_maybe_quoted(raw))


def _parse_graph_node(body: str) -> tuple:
    camera = re.fullmatch(r"(cam\d+) \[camera\] (image_\d+)", body)
    if camera:
        return _camera_atom(camera.group(1), camera.group(2))
    match = re.match(r"(e\d+) \[(object|region)\] ", body)
    if not match:
        raise ValueError(body)
    label, end = _read_json_string(body, match.end())
    tail = body[end:]
    if tail == "":
        return _entity_atom(match.group(1), match.group(2), label, None, None)
    cross = re.fullmatch(r" \| cross-view: (\S+) \(uncertainty (low|medium|high)\)", tail)
    if not cross:
        raise ValueError(body)
    return _entity_atom(match.group(1), match.group(2), label, cross.group(1), cross.group(2))


def _parse_graph_observation(body: str, entity_id: str) -> tuple:
    match = re.match(r"(o\d+) @(image_\d+) ([a-z_]+): ", body)
    if not match:
        raise ValueError(body)
    description, end = _read_json_string(body, match.end())
    if end != len(body):
        raise ValueError(body)
    return _observation_atom(match.group(1), entity_id, match.group(2), match.group(3), description)


def _parse_graph_attribute(body: str, subject: str) -> tuple:
    match = re.match(r"(c\d+) ([a-z_]+) = ", body)
    if not match:
        raise ValueError(body)
    value, end = _read_json_string(body, match.end())
    image_id, frame, support, uncertainty, note = _parse_attribute_tail(body[end:])
    return _claim_atom(
        match.group(1),
        "attribute",
        subject,
        match.group(2),
        None,
        value,
        frame,
        image_id,
        support,
        uncertainty,
        note,
    )


def _parse_graph_child(body: str, parent: str) -> tuple:
    if body.startswith("o"):
        return _parse_graph_observation(body, parent)
    if body.startswith("c"):
        return _parse_graph_attribute(body, parent)
    raise ValueError(body)


def _parse_graph_link(body: str) -> tuple:
    match = re.fullmatch(
        r"(i\d+) (\S+) ~ (\S+) \| (tentative|different) \| uncertainty (low|medium|high)",
        body,
    )
    if not match:
        raise ValueError(body)
    return _link_atom(match.group(1), match.group(2), match.group(3), match.group(4), match.group(5))


def _parse_graph_edge(body: str, frame: str) -> tuple:
    match = re.match(r"(c\d+) (\S+) -([a-z_]+)-> (\S+)", body)
    if not match:
        raise ValueError(body)
    image_id, support, uncertainty, note = _parse_relation_tail(body[match.end():])
    return _claim_atom(
        match.group(1),
        "relation",
        match.group(2),
        match.group(3),
        match.group(4),
        None,
        frame,
        image_id,
        support,
        uncertainty,
        note,
    )


def _parse_graph_unknown(line: str) -> tuple:
    match = re.fullmatch(r"  (u\d+) (.*)", line)
    if not match:
        raise ValueError(line)
    return _unknown_atom(match.group(1), _parse_maybe_quoted(match.group(2)))


def _parse_nodes(lines: list[str], idx: int, found: list) -> int:
    current = None
    while idx < len(lines) and not lines[idx].startswith("Identity links"):
        line = lines[idx]
        if line.startswith("    "):
            if current is None:
                raise ValueError(line)
            found.append(_parse_graph_child(line[4:], current))
        elif line.startswith("  "):
            atom = _parse_graph_node(line[2:])
            found.append(atom)
            current = atom[1]
        else:
            raise ValueError(line)
        idx += 1
    return idx


def _parse_links(lines: list[str], idx: int, found: list) -> int:
    line = lines[idx]
    if line == "Identity links: none":
        return idx + 1
    if line != "Identity links:":
        raise ValueError(line)
    idx += 1
    while idx < len(lines) and lines[idx].startswith("  "):
        found.append(_parse_graph_link(lines[idx][2:]))
        idx += 1
    return idx


def _parse_edges(lines: list[str], idx: int, found: list) -> int:
    line = lines[idx]
    if line == "Edges by frame: none":
        return idx + 1
    if line != "Edges by frame:":
        raise ValueError(line)
    idx += 1
    frame = None
    while idx < len(lines) and not lines[idx].startswith("Unknowns"):
        line = lines[idx]
        if line.startswith("    "):
            if frame is None:
                raise ValueError(line)
            found.append(_parse_graph_edge(line[4:], frame))
        elif line.startswith("  "):
            frame = line[2:]
        else:
            raise ValueError(line)
        idx += 1
    return idx


def _parse_unknowns(lines: list[str], idx: int, found: list) -> int:
    line = lines[idx]
    if line == "Unknowns: none":
        return idx + 1
    if line != "Unknowns:":
        raise ValueError(line)
    idx += 1
    while idx < len(lines):
        found.append(_parse_graph_unknown(lines[idx]))
        idx += 1
    return idx


def parse_graph(text: str) -> frozenset:
    """Parse a graph serialization back into fact atoms."""
    lines = text.splitlines()
    if len(lines) < 4 or lines[0] != GRAPH_HEADER:
        raise ValueError("graph serialization header missing")
    found = [
        _parse_graph_query(lines[1]),
        _parse_graph_temporal(lines[2]),
    ]
    if lines[3] != "Nodes:":
        raise ValueError(lines[3])
    idx = _parse_nodes(lines, 4, found)
    idx = _parse_links(lines, idx, found)
    idx = _parse_edges(lines, idx, found)
    idx = _parse_unknowns(lines, idx, found)
    if idx != len(lines):
        raise ValueError("trailing graph text")
    return frozenset(found)
