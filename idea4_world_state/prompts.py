"""Message builders for the raw, extract, flat, and graph conditions.

Section 8.6 layout, ``LAYOUT_VERSION`` ``ws-layout-1``. Each builder returns
exactly one user message and no system message. For every image, in order, the
content has a text part ``Image k:`` (k starting at 1) and then a PNG data-URL
image part from ``normalized_image``.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
from pathlib import Path

from idea3_consistency.mmsi_consistency import POST_PROMPT, parse_question
from idea4_visual_reasoning.render import normalized_image
from idea4_world_state import contracts, serialize

LAYOUT_VERSION = "ws-layout-1"

# Same patterns and order as idea3_consistency.mmsi_consistency.extract.
# Index 0-1 is the strict parse; 2-3 are the lenient sensitivity cases.
ANSWER_PATTERNS = (
    r"``([A-D])``",
    r"`([A-D])`",
    r"\{([A-D])\}",
    r"\b([A-D])\b(?!\s[a-zA-Z])",
)

_PNG_DATA_PREFIX = "data:image/png;base64,"
_SCHEMA_PLACEHOLDER = "{{INVENTORY_SCHEMA_JSON}}"
_INSTRUCTION_PATH = Path(__file__).resolve().parent / "prompts" / "extract_v3.txt"


# Top-level annotation keys. The description names internal validators, so the
# model sees the schema without these three keys. Nested fields stay.
_SCHEMA_KEYS_OMITTED = ("$schema", "title", "description")


def _canonical_schema_json() -> str:
    """Canonical JSON of the model-facing inventory schema.

    Drops top-level ``$schema``, ``title``, and ``description``. Keeps every
    other key, including ``$defs``.
    """
    schema = json.loads(contracts.SCHEMA_PATH.read_text(encoding="utf-8"))
    facing = {key: value for key, value in schema.items() if key not in _SCHEMA_KEYS_OMITTED}
    return json.dumps(facing, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _load_extraction_instruction() -> str:
    text = _INSTRUCTION_PATH.read_text(encoding="utf-8")
    count = text.count(_SCHEMA_PLACEHOLDER)
    if count != 1:
        raise ValueError(f"{_INSTRUCTION_PATH} must contain {_SCHEMA_PLACEHOLDER} exactly once")
    return text.replace(_SCHEMA_PLACEHOLDER, _canonical_schema_json())


EXTRACTION_INSTRUCTION = _load_extraction_instruction()


def _serialize_py_path() -> Path:
    path = Path(serialize.__file__).resolve()
    if path.suffix == ".pyc":
        path = path.with_suffix(".py")
    return path


def _serialize_py_bytes() -> bytes:
    """Raw bytes of serialize.py. Hashed into the flat and graph prompt ids."""
    return _serialize_py_path().read_bytes()


def _sha256_canonical(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _user(content: list[dict]) -> list[dict]:
    return [{"role": "user", "content": content}]


def _question_stem(question: str) -> str:
    parsed = parse_question(question)
    if parsed is None or len(parsed[1]) != 4:
        raise ValueError("question must parse into a stem and exactly 4 options")
    return parsed[0].strip()


def image_parts(images, max_side) -> list[dict]:
    """Text ``Image k:`` plus a PNG data URL for each raw image, in order.

    ``normalized_image`` applies EXIF orientation, converts to RGB, and
    shrinks the long side to ``max_side`` without cropping.
    """
    parts = []
    for index, raw in enumerate(images, start=1):
        png = normalized_image(raw, max_side)
        url = _PNG_DATA_PREFIX + base64.b64encode(png).decode("ascii")
        parts.append({"type": "text", "text": f"Image {index}:"})
        parts.append({"type": "image_url", "image_url": {"url": url}})
    return parts


def raw_messages(images, question, max_side) -> list[dict]:
    """Images, then the question text with ``POST_PROMPT`` appended once."""
    content = image_parts(images, max_side)
    content.append({"type": "text", "text": question + POST_PROMPT})
    return _user(content)


def extract_messages(images, question, max_side) -> list[dict]:
    """Images, then the stripped stem and ``EXTRACTION_INSTRUCTION``.

    Options are omitted. Raises ``ValueError`` unless ``parse_question``
    returns exactly four options.
    """
    stem = _question_stem(question)
    content = image_parts(images, max_side)
    content.append({"type": "text", "text": "Question stem:\n" + stem + "\n" + EXTRACTION_INSTRUCTION})
    return _user(content)


def flat_messages(images, question, inventory, max_side) -> list[dict]:
    """Images, the flat evidence text, then the unchanged question."""
    content = image_parts(images, max_side)
    content.append({"type": "text", "text": serialize.render_flat(inventory)})
    content.append({"type": "text", "text": question + POST_PROMPT})
    return _user(content)


def graph_messages(images, question, inventory, max_side) -> list[dict]:
    """Images, the graph evidence text, then the unchanged question."""
    content = image_parts(images, max_side)
    content.append({"type": "text", "text": serialize.render_graph(inventory)})
    content.append({"type": "text", "text": question + POST_PROMPT})
    return _user(content)


def parse_answer(text: str | None) -> tuple[str | None, int | None]:
    """Return ``(letter, pattern_index)`` using ``extract``'s patterns.

    The letter agrees with ``extract`` on every input. ``pattern_index`` is
    the 0-based index of the first matching pattern, or ``None`` when no
    pattern matches. Falsy text is read as empty, matching ``extract``.
    """
    haystack = text or ""
    for index, pattern in enumerate(ANSWER_PATTERNS):
        match = re.search(pattern, haystack)
        if match:
            return match.group(1), index
    return None, None


def prompt_hashes() -> dict[str, str]:
    """Section 8.6 ``prompt_sha256`` values for raw, extract, flat, and graph.

    Canonical JSON uses ``sort_keys``, separators ``(",", ":")``, and
    ``ensure_ascii=False``. Flat and graph also hash the raw bytes of
    ``serialize.py``.
    """
    layout = LAYOUT_VERSION
    post_prompt = POST_PROMPT
    serializer_hash = hashlib.sha256(_serialize_py_bytes()).hexdigest()
    return {
        "raw": _sha256_canonical({"layout": layout, "post_prompt": post_prompt}),
        "extract": _sha256_canonical({"layout": layout, "instruction": EXTRACTION_INSTRUCTION}),
        "flat": _sha256_canonical({
            "layout": layout,
            "post_prompt": post_prompt,
            "serializer": "flat",
            "serialize_py_sha256": serializer_hash,
        }),
        "graph": _sha256_canonical({
            "layout": layout,
            "post_prompt": post_prompt,
            "serializer": "graph",
            "serialize_py_sha256": serializer_hash,
        }),
    }
