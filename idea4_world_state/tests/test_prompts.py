"""Prompt layout, answer parsing, and prompt hashes."""

import ast
import hashlib
import inspect
import io
import json
import re

import pytest
from jsonschema import Draft202012Validator
from PIL import Image

from idea3_consistency.mmsi_consistency import POST_PROMPT, build_question, extract, parse_question
from idea4_visual_reasoning.render import normalized_image
from idea4_world_state import contracts, prompts
from idea4_world_state.backend import MockBackend, image_bytes_from_part
from idea4_world_state.calllog import cache_key
from idea4_world_state.inventory import atoms, parse_inventory
from idea4_world_state.serialize import parse_flat, parse_graph, render_flat, render_graph

GOLD_SENTINEL = "GOLD_SENTINEL_9f3a2c"
RATIONALE_SENTINEL = "RATIONALE_SENTINEL_7b1e4d"
STEM = "Café  view: which object is nearest the doorway?\n"
OPTIONS = (
    "OPTTOKEN_alpha_7741xq",
    "OPTTOKEN_bravo_2290xq",
    "OPTTOKEN_charlie_8834xq",
    "OPTTOKEN_delta_1106xq",
)
QUESTION = build_question(STEM, OPTIONS)
MAX_SIDE = 32
_LIMIT_PHRASES = (
    "8 entities",
    "32 observations",
    "8 identity links",
    "24 claims",
    "8 unknowns",
    "label 60",
    "description 120",
    "value 60",
    "note 120",
    "basis 160",
    "unknown 200",
)

# text, letter, pattern index. Index follows extract: double backticks, single
# backticks, braces, then a bare letter.
_ANSWER_CASES = [
    ("``A``", "A", 0),
    ("``B``", "B", 0),
    ("``C``", "C", 0),
    ("``D``", "D", 0),
    ("`A`", "A", 1),
    ("`B`", "B", 1),
    ("`C`", "C", 1),
    ("{A}", "A", 2),
    ("{D}", "D", 2),
    ("B", "B", 3),
    ("The answer is C.", "C", 3),
    ("", None, None),
    (None, None, None),
    ("   ", None, None),
    (False, None, None),
    (0, None, None),
    ("The Camera Shows A Red Object Clearly.", None, None),
    ("I choose D", "D", 3),
    ("Option A is best", None, None),
    ("See A then B", "B", 3),
    ("``A`` and also `B`", "A", 0),
    ("prefix `C` then {D}", "C", 1),
    ("use {B} not C", "B", 2),
    ("ABCD", None, None),
    ("A.", "A", 3),
    ("(C)", "C", 3),
    ("answer ``D`` please", "D", 0),
    ("no letter here", None, None),
    ("a", None, None),
    ("``E``", None, None),
    ("{a}", None, None),
    ("A  is", "A", 3),
    ("final\nB", "B", 3),
    ("```A```", "A", 0),
    ("``A`", "A", 1),
    ("Camera A faces B.", "B", 3),
    ("X A Y", None, None),
    ("A-B", "A", 3),
    ("Choose A", "A", 3),
    ("the letter is {B}", "B", 2),
    ("xx `B` yy", "B", 1),
    ("Answer: ``A``", "A", 0),
    ("```\nA\n```", "A", 3),
    ("I think A is wrong but B", "B", 3),
]


def _jpeg(index, size):
    image = Image.new("RGB", size, ((index * 37) % 256, (index * 67) % 256, (index * 97) % 256))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    return buffer.getvalue()


def _images(count):
    return [_jpeg(index, (64 + index, 40 + index)) for index in range(1, count + 1)]


def _inventory(name):
    return json.loads((contracts.FIXTURES_DIR / name).read_text(encoding="utf-8"))


def _only_user(messages):
    assert isinstance(messages, list)
    assert len(messages) == 1
    message = messages[0]
    assert set(message) == {"role", "content"}
    assert message["role"] == "user"
    assert isinstance(message["content"], list)
    return message["content"]


def _strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)


def _blob(messages):
    return "\n".join(_strings(messages))


def _canonical_schema():
    """Model-facing schema: full file with top-level annotation keys removed."""
    schema = json.loads(contracts.SCHEMA_PATH.read_text(encoding="utf-8"))
    facing = {
        key: value
        for key, value in schema.items()
        if key not in ("$schema", "title", "description")
    }
    return json.dumps(facing, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _sha(payload):
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _assert_png_parts(content, raws, max_side):
    assert len(content) >= 2 * len(raws)
    prefix = content[: 2 * len(raws)]
    for index, raw in enumerate(raws, start=1):
        label, part = prefix[2 * index - 2], prefix[2 * index - 1]
        assert label == {"type": "text", "text": f"Image {index}:"}
        assert set(part) == {"type", "image_url"}
        assert set(part["image_url"]) == {"url"}
        assert part["image_url"]["url"].startswith("data:image/png;base64,")
        assert image_bytes_from_part(part) == normalized_image(raw, max_side)
    labels = [part["text"] for part in prefix if part["type"] == "text"]
    assert labels == [f"Image {index}:" for index in range(1, len(raws) + 1)]
    assert prefix == prompts.image_parts(raws, max_side)


def test_sentinels_stay_out_of_every_message():
    # Gold and rationale are planted on the row and are not passed to builders.
    row = {
        "question": QUESTION,
        "answer": GOLD_SENTINEL,
        "thought": RATIONALE_SENTINEL,
        "images": _images(2),
    }
    inventory = _inventory("valid_full.json")
    assert GOLD_SENTINEL not in row["question"]
    assert RATIONALE_SENTINEL not in row["question"]
    stem = parse_question(row["question"])[0]
    for option in OPTIONS:
        assert option not in stem
        assert option in row["question"]

    built = {
        "raw": prompts.raw_messages(row["images"], row["question"], MAX_SIDE),
        "extract": prompts.extract_messages(row["images"], row["question"], MAX_SIDE),
        "flat": prompts.flat_messages(row["images"], row["question"], inventory, MAX_SIDE),
        "graph": prompts.graph_messages(row["images"], row["question"], inventory, MAX_SIDE),
    }
    extract_blob = _blob(built["extract"])
    for option in OPTIONS:
        assert option not in extract_blob
    assert "Options:" not in extract_blob
    assert POST_PROMPT not in extract_blob
    for messages in built.values():
        blob = _blob(messages)
        assert GOLD_SENTINEL not in blob
        assert RATIONALE_SENTINEL not in blob
        _only_user(messages)


def test_question_text_is_byte_identical_and_post_prompt_is_once():
    images = _images(2)
    inventory = _inventory("valid_empty.json")
    built = [
        prompts.raw_messages(images, QUESTION, MAX_SIDE),
        prompts.flat_messages(images, QUESTION, inventory, MAX_SIDE),
        prompts.graph_messages(images, QUESTION, inventory, MAX_SIDE),
    ]
    expected = QUESTION + POST_PROMPT
    for messages in built:
        text = _only_user(messages)[-1]["text"]
        assert text == expected
        assert text.encode("utf-8").endswith(POST_PROMPT.encode("utf-8"))
        assert text.count(POST_PROMPT) == 1
        assert "Café  view" in text

    bare = "not a parsed question"
    for messages in (
        prompts.raw_messages(images, bare, MAX_SIDE),
        prompts.flat_messages(images, bare, inventory, MAX_SIDE),
        prompts.graph_messages(images, bare, inventory, MAX_SIDE),
    ):
        assert _only_user(messages)[-1]["text"] == bare + POST_PROMPT


@pytest.mark.parametrize(
    "question",
    [
        "",
        "no options here",
        "What?\nOptions: A: a, B: b, C: c",
        "What?\nOptions: B: b, A: a, C: c, D: d",
        "What?\nOptions: A: a, B: b, C: c, D: d, A: again",
    ],
)
def test_extract_requires_four_options(question):
    with pytest.raises(ValueError, match="exactly 4 options"):
        prompts.extract_messages(_images(1), question, MAX_SIDE)


@pytest.mark.parametrize("count", [2, 3, 10])
def test_image_count_and_order_are_preserved(count):
    raws = _images(count)
    inventory = _inventory("valid_full.json")
    identity = MockBackend().model_identity()
    builders = [
        ("raw_answer", prompts.raw_messages(raws, QUESTION, MAX_SIDE)),
        ("extract", prompts.extract_messages(raws, QUESTION, MAX_SIDE)),
        ("flat_answer", prompts.flat_messages(raws, QUESTION, inventory, MAX_SIDE)),
        ("graph_answer", prompts.graph_messages(raws, QUESTION, inventory, MAX_SIDE)),
    ]
    for kind, messages in builders:
        content = _only_user(messages)
        _assert_png_parts(content, raws, MAX_SIDE)
        image_parts = [part for part in content if part.get("type") == "image_url"]
        labels = [part["text"] for part in content if part.get("text", "").startswith("Image ")]
        assert len(image_parts) == count
        assert labels == [f"Image {index}:" for index in range(1, count + 1)]
        key = cache_key(messages, 64, kind, identity, 1)
        assert len(key) == 64
        int(key, 16)
        wide = normalized_image(raws[0], MAX_SIDE)
        assert wide != normalized_image(raws[0], MAX_SIDE + 8)


@pytest.mark.parametrize("name", ["valid_full.json", "valid_empty.json"])
def test_flat_and_graph_evidence_matches_fixture(name):
    inventory = _inventory(name)
    images = _images(2)
    flat = _only_user(prompts.flat_messages(images, QUESTION, inventory, MAX_SIDE))
    graph = _only_user(prompts.graph_messages(images, QUESTION, inventory, MAX_SIDE))
    assert flat[-2] == {"type": "text", "text": render_flat(inventory)}
    assert graph[-2] == {"type": "text", "text": render_graph(inventory)}
    assert flat[-1]["text"] == QUESTION + POST_PROMPT
    assert graph[-1]["text"] == QUESTION + POST_PROMPT
    assert len(flat) == 6
    assert len(graph) == 6


def test_extraction_instruction_states_limits_schema_and_example():
    schema = _canonical_schema()
    instruction = prompts.EXTRACTION_INSTRUCTION
    disk = prompts._INSTRUCTION_PATH.read_text(encoding="utf-8")
    before, placeholder, after = disk.partition("{{INVENTORY_SCHEMA_JSON}}")
    assert placeholder == "{{INVENTORY_SCHEMA_JSON}}"
    assert "{{INVENTORY_SCHEMA_JSON}}" not in before + after
    assert instruction == before + schema + after
    prose = instruction.replace(schema, " ")
    for phrase in _LIMIT_PHRASES:
        assert phrase in prose
    assert len(prose.split()) < 1100

    stem = parse_question(QUESTION)[0].strip()
    text = _only_user(prompts.extract_messages(_images(2), QUESTION, MAX_SIDE))[-1]["text"]
    assert text == "Question stem:\n" + stem + "\n" + instruction
    assert stem == "Café  view: which object is nearest the doorway?"

    fences = re.findall(r"(?m)^```json\n(.*?)\n```$", instruction, flags=re.DOTALL)
    assert len(fences) == 2
    separate, shared = (parse_inventory(example, 2) for example in fences)
    assert {entity["id"] for entity in separate["entities"]} == {"e1", "e2"}
    assert all(entity["cross_view"] is None for entity in separate["entities"])
    assert {obs["image_id"] for obs in separate["observations"]} == {"image_1", "image_2"}
    assert separate["claims"] == []
    assert separate["unknowns"]
    shared_entity = shared["entities"][0]
    assert shared_entity["cross_view"] == {"status": "same", "uncertainty": "medium"}
    assert {obs["image_id"] for obs in shared["observations"]} == {"image_1", "image_2"}
    assert shared["claims"] == []
    assert "nothing established from the image" not in instruction


def test_embedded_schema_omits_developer_metadata():
    raw = json.loads(contracts.SCHEMA_PATH.read_text(encoding="utf-8"))
    instruction = prompts.EXTRACTION_INSTRUCTION
    disk = prompts._INSTRUCTION_PATH.read_text(encoding="utf-8")
    before, placeholder, after = disk.partition("{{INVENTORY_SCHEMA_JSON}}")
    assert placeholder == "{{INVENTORY_SCHEMA_JSON}}"
    embedded = instruction[len(before):len(instruction) - len(after)]
    for needle in ("WP2", "WP5", "semantic validator", '"title"', '"$schema"'):
        assert needle not in instruction
    assert raw["description"] not in instruction

    parsed = json.loads(embedded)
    assert set(parsed) == set(raw) - {"$schema", "title", "description"}
    assert "$defs" in parsed
    validator = Draft202012Validator(parsed)
    for name in ("valid_full.json", "valid_empty.json"):
        validator.validate(_inventory(name))

    full = json.dumps(raw, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    stale = before + full + after
    extract_hash = prompts.prompt_hashes()["extract"]
    assert extract_hash == _sha({"layout": prompts.LAYOUT_VERSION, "instruction": instruction})
    assert extract_hash != _sha({"layout": prompts.LAYOUT_VERSION, "instruction": stale})


def test_answer_patterns_match_extract_source():
    source = inspect.getsource(extract)
    match = re.search(r"for pat in \[(.+?)\]:", source, flags=re.DOTALL)
    assert match is not None
    patterns = tuple(ast.literal_eval(part) for part in match.group(1).split(", "))
    assert patterns == prompts.ANSWER_PATTERNS


@pytest.mark.parametrize("text,letter,rule", _ANSWER_CASES)
def test_parse_answer_agrees_with_extract(text, letter, rule):
    assert prompts.parse_answer(text) == (letter, rule)
    assert extract(text) == letter
    if letter is None:
        assert rule is None
    else:
        assert rule in range(len(prompts.ANSWER_PATTERNS))


def test_prompt_hashes_follow_the_contract_and_change_when_inputs_change(monkeypatch):
    base = prompts.prompt_hashes()
    assert prompts.prompt_hashes() == base
    assert set(base) == {"raw", "extract", "flat", "graph"}
    serializer_hash = hashlib.sha256(prompts._serialize_py_bytes()).hexdigest()
    assert base["raw"] == _sha({"layout": prompts.LAYOUT_VERSION, "post_prompt": prompts.POST_PROMPT})
    assert base["extract"] == _sha({
        "layout": prompts.LAYOUT_VERSION,
        "instruction": prompts.EXTRACTION_INSTRUCTION,
    })
    assert base["flat"] == _sha({
        "layout": prompts.LAYOUT_VERSION,
        "post_prompt": prompts.POST_PROMPT,
        "serializer": "flat",
        "serialize_py_sha256": serializer_hash,
    })
    assert base["graph"] == _sha({
        "layout": prompts.LAYOUT_VERSION,
        "post_prompt": prompts.POST_PROMPT,
        "serializer": "graph",
        "serialize_py_sha256": serializer_hash,
    })
    assert base["flat"] != base["graph"]

    post = prompts.POST_PROMPT
    monkeypatch.setattr(prompts, "POST_PROMPT", post + " café")
    changed = prompts.prompt_hashes()
    assert changed["raw"] != base["raw"]
    assert changed["flat"] != base["flat"]
    assert changed["graph"] != base["graph"]
    assert changed["extract"] == base["extract"]
    assert changed["raw"] == _sha({"layout": prompts.LAYOUT_VERSION, "post_prompt": post + " café"})
    escaped = json.dumps(
        {"layout": prompts.LAYOUT_VERSION, "post_prompt": post + " café"},
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
    )
    assert changed["raw"] != hashlib.sha256(escaped.encode("utf-8")).hexdigest()
    monkeypatch.setattr(prompts, "POST_PROMPT", post)

    instruction = prompts.EXTRACTION_INSTRUCTION
    monkeypatch.setattr(prompts, "EXTRACTION_INSTRUCTION", instruction + " extra rule")
    changed = prompts.prompt_hashes()
    assert changed["extract"] != base["extract"]
    assert changed["raw"] == base["raw"]
    assert changed["flat"] == base["flat"]
    assert changed["graph"] == base["graph"]
    monkeypatch.setattr(prompts, "EXTRACTION_INSTRUCTION", instruction)

    monkeypatch.setattr(prompts, "LAYOUT_VERSION", "ws-layout-test")
    changed = prompts.prompt_hashes()
    assert all(changed[key] != base[key] for key in base)
    monkeypatch.setattr(prompts, "LAYOUT_VERSION", "ws-layout-1")


def test_extract_v2_does_not_reuse_the_copied_empty_example():
    """Pilot revision 1: visible objects stay in the inventory when relations are unknown."""
    live = prompts._INSTRUCTION_PATH
    preserved = live.with_name("extract_v1.txt")
    previous = live.with_name("extract_v2.txt")
    rejected = live.with_name("extract_v4.txt")
    assert live.name == "extract_v3.txt"
    assert rejected.is_file()
    rejected_text = rejected.read_text(encoding="utf-8")
    assert "left_of" in rejected_text
    rejected_fences = re.findall(r"(?m)^```json\n(.*?)\n```$", rejected_text, flags=re.DOTALL)
    related = parse_inventory(rejected_fences[-1].replace("{{INVENTORY_SCHEMA_JSON}}", ""), 2)
    assert related["claims"][0]["predicate"] == "left_of"
    assert related["claims"][0]["support"] == ["o1", "o2"]
    assert preserved.is_file() and previous.is_file()
    old = preserved.read_text(encoding="utf-8")
    assert "nothing established from the image" in old
    assert "nothing established from the image" not in prompts.EXTRACTION_INSTRUCTION
    v1_instruction = old.replace("{{INVENTORY_SCHEMA_JSON}}", prompts._canonical_schema_json())
    assert prompts.prompt_hashes()["extract"] != _sha({
        "layout": prompts.LAYOUT_VERSION,
        "instruction": v1_instruction,
    })

    example = re.findall(
        r"(?m)^```json\n(.*?)\n```$",
        prompts.EXTRACTION_INSTRUCTION,
        flags=re.DOTALL,
    )[1]
    inventory = parse_inventory(example, 2)
    found = atoms(inventory)
    assert parse_flat(render_flat(inventory)) == found == parse_graph(render_graph(inventory))

    rejected = (
        "the letter S is on the wall in image_2, but its direction relative to the "
        "painting in image_1 cannot be determined without knowing the spatial "
        "relationship between the two images"
    )
    assert len(rejected) == 178
    held = json.loads(example)
    held["unknowns"] = [rejected]
    assert parse_inventory(json.dumps(held), 2)["unknowns"] == [rejected]
    held["unknowns"] = ["x" * 201]
    with pytest.raises(Exception) as caught:
        parse_inventory(json.dumps(held), 2)
    assert getattr(caught.value, "rule", None) == "schema"


def test_prompt_hash_serializer_bytes_change(monkeypatch):
    base = prompts.prompt_hashes()
    monkeypatch.setattr(prompts, "_serialize_py_bytes", lambda: b"not-the-real-serializer")
    changed = prompts.prompt_hashes()
    assert changed["flat"] != base["flat"]
    assert changed["graph"] != base["graph"]
    assert changed["flat"] != changed["graph"]
    assert changed["raw"] == base["raw"]
    assert changed["extract"] == base["extract"]
