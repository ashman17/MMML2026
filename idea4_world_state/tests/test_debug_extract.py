"""The extraction retest stays off the official profiles and the pilot cache."""

import io
import json

import pytest
from PIL import Image

from idea3_consistency.mmsi_consistency import build_question
from idea4_world_state import prompts, runner
from idea4_world_state.backend import MockBackend
from idea4_world_state.debug_extract import COPIED_EMPTY_PHRASE, DEBUG_IDS, require_dev_ids, run_extracts


def test_debug_ids_are_development_items():
    selection = runner.load_selection(runner.contracts.PREPARE_DIR)
    assert require_dev_ids(selection, list(DEBUG_IDS)) == list(DEBUG_IDS)
    with pytest.raises(runner.RunnerError):
        require_dev_ids(selection, [999999])


def test_debug_extract_uses_the_live_prompt_and_its_own_directory(tmp_path):
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), (12, 24, 36)).save(buffer, format="PNG")
    question = build_question(
        "Which object is visible?\n",
        ("alpha", "bravo", "charlie", "delta"),
    )
    item = {
        "id": 583,
        "category": "Positional Relationship (Obj.–Obj.)",
        "n_images": 1,
        "partition": "explore",
        "role": "dev",
        "question": question,
        "images": [buffer.getvalue()],
    }
    out = tmp_path / "debug"
    rows = run_extracts([item], MockBackend(), out, 32, 128)
    assert rows[0]["n_image_parts"] == 1
    assert rows[0]["status"] == "ok"
    request = next((out / "requests").glob("*.json"))
    text = request.read_text(encoding="utf-8")
    assert "flowerpot" in text
    assert COPIED_EMPTY_PHRASE not in text
    assert "alpha" not in text
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["instruction"] == prompts._INSTRUCTION_PATH.name == "extract_v3.txt"
    assert summary["prompt_sha256"]["extract"] == prompts.prompt_hashes()["extract"]
    assert (out / "cache").is_dir()
    with pytest.raises(runner.RunnerError):
        run_extracts([item], MockBackend(), out, 32, 128)
