"""Prepare sampling: pinned golden values, determinism, and contract guards."""

import hashlib
import io
import json
import re
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

from idea4_world_state import contracts
from idea4_world_state.data import PrepareError, cli_prepare, prepare

CATEGORY_ORDER = [
    "Positional Relationship (Obj.–Obj.)",
    "Positional Relationship (Obj.–Reg.)",
    "Attribute (Meas.)",
    "MSR",
    "Positional Relationship (Cam.–Cam.)",
    "Motion (Cam.)",
    "Positional Relationship (Cam.–Reg.)",
    "Motion (Obj.)",
    "Attribute (Appr.)",
    "Positional Relationship (Reg.–Reg.)",
    "Positional Relationship (Cam.–Obj.)",
]
EVAL200_COUNTS = {
    "Positional Relationship (Obj.–Obj.)": 20,
    "Positional Relationship (Obj.–Reg.)": 20,
    "Attribute (Meas.)": 20,
    "MSR": 6,
    "Positional Relationship (Cam.–Cam.)": 20,
    "Motion (Cam.)": 19,
    "Positional Relationship (Cam.–Reg.)": 19,
    "Motion (Obj.)": 19,
    "Attribute (Appr.)": 19,
    "Positional Relationship (Reg.–Reg.)": 19,
    "Positional Relationship (Cam.–Obj.)": 19,
}
EVAL100_COUNTS = {
    "Positional Relationship (Obj.–Obj.)": 10,
    "Positional Relationship (Obj.–Reg.)": 10,
    "Attribute (Meas.)": 10,
    "MSR": 6,
    "Positional Relationship (Cam.–Cam.)": 10,
    "Motion (Cam.)": 9,
    "Positional Relationship (Cam.–Reg.)": 9,
    "Motion (Obj.)": 9,
    "Attribute (Appr.)": 9,
    "Positional Relationship (Reg.–Reg.)": 9,
    "Positional Relationship (Cam.–Obj.)": 9,
}
EVAL200_HASH = "efcc9f0272a5653d7d129b9b7851d09fefd3c7cf059d8f5be4283c9a6484e76c"
EVAL100_HASH = "2ad8facdfba28f4125071357380002ff533a7f2b3d0f68d8599c87d26e25a25d"
DEV_HASH = "2e3855ec36382248f883cffddb9b1df3fd52f25f8b357a80b53d620e2394fb92"
DEV_IDS = [
    575, 583, 955, 619, 783, 809, 356, 188, 531, 528, 912, 33,
    784, 524, 319, 14, 236, 235, 734, 721, 3, 505,
]
TIMING_BY_COUNT = {"3": 78, "4": 13, "5": 72, "6": 75, "7": 77, "8": 223, "9": None, "10": None}


def _id_list_sha256(ids):
    return hashlib.sha256(json.dumps(ids).encode()).hexdigest()


def _manifest_sha256(selection):
    payload = {key: value for key, value in selection.items() if key != "manifest_sha256"}
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode()).hexdigest()


def _counts(selection, ids):
    categories = {item["id"]: item["category"] for item in selection["items"]}
    return dict(Counter(categories[row_id] for row_id in ids))


def _question(stem):
    return f"{stem}\nOptions: A: alpha, B: bravo, C: charlie, D: delta"


def _png(color):
    image = Image.new("RGB", (8, 8), color)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _write_synthetic(directory):
    """Scarce explore category, a transitive image chain, and a blocked timing item.

    Confirm Plenty groups are {1, 4}, {2}, {3}, and the chain {40, 41, 42}.
    Id 30 shares an image with {1, 4}, so the free 3-image explore question is 31.
    Scarce has one explore question and no confirm question.
    """
    blob = {index: _png((index, (index * 17) % 256, (index * 29) % 256)) for index in range(1, 30)}
    link_yz = _png((9, 9, 9))
    link_zw = _png((4, 5, 6))
    rows = [
        (1, "confirm", "Plenty", [blob[1], blob[2]]),
        (2, "confirm", "Plenty", [blob[3], blob[4]]),
        (3, "confirm", "Plenty", [blob[5], blob[6]]),
        (4, "confirm", "Plenty", [blob[1], blob[7]]),
        (10, "explore", "Plenty", [blob[8], blob[9]]),
        (11, "explore", "Plenty", [blob[10], blob[11]]),
        (12, "explore", "Plenty", [blob[12], blob[13]]),
        (20, "explore", "Scarce", [blob[14], blob[15]]),
        (30, "explore", "Plenty", [blob[1], blob[16], blob[17]]),
        (31, "explore", "Plenty", [blob[18], blob[19], blob[20]]),
        (32, "explore", "Plenty", [blob[21], blob[22], blob[23]]),
        (40, "confirm", "Plenty", [blob[24], link_yz]),
        (41, "confirm", "Plenty", [link_yz, link_zw]),
        (42, "confirm", "Plenty", [link_zw, blob[25]]),
    ]
    table = pa.table({
        "id": pa.array([row[0] for row in rows], type=pa.int64()),
        "question_type": pa.array([row[2] for row in rows]),
        "question": pa.array([_question(f"stem {row[0]}") for row in rows]),
        "images": pa.array([row[3] for row in rows], type=pa.list_(pa.binary())),
        "answer": pa.array(["A"] * len(rows)),
        "thought": pa.array(["hidden rationale"] * len(rows)),
    })
    dataset = directory / "synthetic.parquet"
    pq.write_table(table, dataset)
    explore = [row[0] for row in rows if row[1] == "explore"]
    confirm = [row[0] for row in rows if row[1] == "confirm"]
    split = directory / "split.json"
    split.write_text(json.dumps({"explore": explore, "confirm": confirm}))
    return dataset, split


def _selected_groups(selection, roles):
    return [item["image_group"] for item in selection["items"] if item["role"] in roles]


@pytest.fixture(scope="module")
def golden(tmp_path_factory):
    if not contracts.DEFAULT_DATASET.is_file():
        pytest.skip("data/MMSI_Bench.parquet is absent")
    out = tmp_path_factory.mktemp("golden_prepare")
    prepare(contracts.DEFAULT_DATASET, contracts.DEFAULT_SPLIT, out)
    selection = json.loads((out / "selection.json").read_text())
    return selection, out


def test_golden_pinned_dataset(golden):
    selection, out = golden
    assert selection["category_order"] == CATEGORY_ORDER
    assert _counts(selection, selection["profiles"]["eval200"]["ids"]) == EVAL200_COUNTS
    assert _counts(selection, selection["profiles"]["eval100"]["ids"]) == EVAL100_COUNTS
    assert selection["profiles"]["eval200"]["id_list_sha256"] == EVAL200_HASH
    assert selection["profiles"]["eval100"]["id_list_sha256"] == EVAL100_HASH
    assert selection["profiles"]["dev"]["id_list_sha256"] == DEV_HASH
    assert selection["profiles"]["eval200"]["id_list_sha256"] == _id_list_sha256(selection["profiles"]["eval200"]["ids"])
    assert selection["profiles"]["eval100"]["id_list_sha256"] == _id_list_sha256(selection["profiles"]["eval100"]["ids"])
    assert selection["profiles"]["dev"]["id_list_sha256"] == _id_list_sha256(selection["profiles"]["dev"]["ids"])
    assert selection["profiles"]["dev"]["ids"] == DEV_IDS
    assert selection["profiles"]["timing"]["by_image_count"] == TIMING_BY_COUNT
    assert selection["profiles"]["timing"]["ids"] == [78, 13, 72, 75, 77, 223]
    assert selection["shortfalls"] == {}
    assert selection["seed"] == 0
    assert selection["dataset_sha256"] == contracts.DATASET_SHA256
    assert selection["manifest_sha256"] == _manifest_sha256(selection)
    assert selection["code_sha256"] == hashlib.sha256(Path(contracts.PACKAGE_DIR / "data.py").read_bytes()).hexdigest()

    by_id = {item["id"]: item for item in selection["items"]}
    assert len(by_id) == 1000
    eval_ids = selection["profiles"]["eval200"]["ids"]
    dev_ids = selection["profiles"]["dev"]["ids"]
    timing_ids = selection["profiles"]["timing"]["ids"]
    assert eval_ids == [item_id for item_id, item in sorted(by_id.items(), key=lambda pair: pair[1]["rank"] or 0) if item["role"] == "eval"]
    assert [by_id[item_id]["rank"] for item_id in eval_ids] == list(range(1, 201))
    assert selection["profiles"]["eval100"]["ids"] == eval_ids[:100]
    assert all(by_id[item_id]["in_eval100"] for item_id in eval_ids[:100])
    assert not any(by_id[item_id]["in_eval100"] for item_id in eval_ids[100:])
    assert all(by_id[item_id]["partition"] == "confirm" and by_id[item_id]["n_images"] == 2 for item_id in eval_ids)
    assert all(by_id[item_id]["partition"] == "explore" and by_id[item_id]["role"] == "dev" for item_id in dev_ids)
    assert Counter(by_id[item_id]["category"] for item_id in dev_ids) == {category: 2 for category in CATEGORY_ORDER}
    for count, item_id in TIMING_BY_COUNT.items():
        if item_id is None:
            assert not any(item["role"] == "timing" and item["n_images"] == int(count) for item in selection["items"])
        else:
            assert by_id[item_id]["role"] == "timing"
            assert by_id[item_id]["n_images"] == int(count)
            assert by_id[item_id]["partition"] == "explore"
    raw_ids = selection["profiles"]["raw_all"]["ids"]
    assert raw_ids[:200] == eval_ids
    assert raw_ids[200:222] == dev_ids
    assert raw_ids[222:228] == timing_ids
    assert raw_ids[228:] == sorted(set(by_id) - set(eval_ids) - set(dev_ids) - set(timing_ids))
    assert len(raw_ids) == 1000 and len(set(raw_ids)) == 1000

    excluded = {entry["id"]: entry["reason"] for entry in selection["exclusions"]}
    assert set(excluded) == set(by_id) - set(eval_ids) - set(dev_ids)
    assert set(excluded.values()) <= set(contracts.EXCLUSION_REASONS)
    for item_id, reason in excluded.items():
        if by_id[item_id]["n_images"] != 2:
            assert reason == "not_two_images"
        else:
            assert reason in ("shared_image_group", "not_reached")

    page = (out / "inspection" / "index.html").read_text()
    assert "Options:" not in page
    inspected = {int(item_id) for item_id in re.findall(r'data-id="(\d+)"', page)}
    assert inspected == set(eval_ids) | set(dev_ids) | set(timing_ids)
    for item_id in inspected:
        item = by_id[item_id]
        for index in range(1, item["n_images"] + 1):
            path = out / "inspection" / "images" / f"{item_id}_{index}.jpg"
            with Image.open(path) as thumb:
                assert thumb.format == "JPEG"
                assert max(thumb.size) <= 384


def test_eval_dev_timing_do_not_share_a_group(golden):
    selection, _out = golden
    groups = _selected_groups(selection, {"eval", "dev", "timing"})
    assert len(groups) == len(set(groups))
    assert len(groups) == 200 + 22 + 6


def test_selection_json_is_deterministic(tmp_path):
    dataset, split = _write_synthetic(tmp_path)
    first = tmp_path / "first"
    second = tmp_path / "second"
    prepare(dataset, split, first, verify_pin=False)
    prepare(dataset, split, second, verify_pin=False)
    assert (first / "selection.json").read_bytes() == (second / "selection.json").read_bytes()


def test_parquet_reads_only_inference_columns(tmp_path, monkeypatch):
    recorded = []
    real_iter = pq.ParquetFile.iter_batches
    real_read = pq.read_table

    def iter_batches(self, *args, **kwargs):
        recorded.append(kwargs.get("columns"))
        return real_iter(self, *args, **kwargs)

    def read_table(*args, **kwargs):
        recorded.append(kwargs.get("columns"))
        return real_read(*args, **kwargs)

    monkeypatch.setattr(pq.ParquetFile, "iter_batches", iter_batches)
    monkeypatch.setattr(pq, "read_table", read_table)
    dataset, split = _write_synthetic(tmp_path)
    prepare(dataset, split, tmp_path / "out", verify_pin=False)
    assert recorded, "parquet was not read"
    for columns in recorded:
        assert columns is not None
        assert set(columns) <= set(contracts.INFERENCE_COLUMNS)
        assert "answer" not in columns
        assert "thought" not in columns


def test_scarce_category_records_shortfall_without_backfill(tmp_path):
    dataset, split = _write_synthetic(tmp_path)
    prepare(dataset, split, tmp_path / "out", verify_pin=False)
    selection = json.loads((tmp_path / "out" / "selection.json").read_text())
    by_id = {item["id"]: item for item in selection["items"]}
    eval_ids = selection["profiles"]["eval200"]["ids"]
    dev_ids = selection["profiles"]["dev"]["ids"]
    assert selection["shortfalls"] == {"Scarce": 1}
    assert len(eval_ids) == 4
    assert set(eval_ids) < set(by_id)
    assert all(by_id[item_id]["partition"] == "confirm" for item_id in eval_ids)
    assert {by_id[item_id]["category"] for item_id in eval_ids} == {"Plenty"}
    assert sum(item_id in (1, 4) for item_id in eval_ids) == 1
    assert 2 in eval_ids and 3 in eval_ids
    assert sum(item_id in (40, 41, 42) for item_id in eval_ids) == 1
    assert {by_id[item_id]["image_group"] for item_id in (40, 41, 42)} == {by_id[40]["image_group"]}
    assert by_id[1]["image_group"] == by_id[4]["image_group"] == by_id[30]["image_group"]
    plenty_dev = [item_id for item_id in dev_ids if by_id[item_id]["category"] == "Plenty"]
    scarce_dev = [item_id for item_id in dev_ids if by_id[item_id]["category"] == "Scarce"]
    assert len(plenty_dev) == 2
    assert set(plenty_dev) < {10, 11, 12}
    assert scarce_dev == [20]
    leftover = {10, 11, 12} - set(plenty_dev)
    reasons = {entry["id"]: entry["reason"] for entry in selection["exclusions"]}
    assert reasons[leftover.pop()] == "not_reached"
    loser = ({1, 4} - set(eval_ids)).pop()
    assert reasons[loser] == "shared_image_group"
    assert selection["profiles"]["timing"]["by_image_count"]["3"] == 31
    assert selection["profiles"]["timing"]["ids"] == [31]
    for count in ("4", "5", "6", "7", "8", "9", "10"):
        assert selection["profiles"]["timing"]["by_image_count"][count] is None
    assert reasons[30] == "not_two_images"
    assert reasons[31] == "not_two_images"
    groups = _selected_groups(selection, {"eval", "dev", "timing"})
    assert len(groups) == len(set(groups))


def test_refuses_nonempty_output(tmp_path, capsys):
    out = tmp_path / "out"
    out.mkdir()
    marker = out / "selection.json"
    marker.write_text("keep")
    with pytest.raises(SystemExit) as caught:
        cli_prepare(SimpleNamespace(dataset=tmp_path / "missing.parquet", split=tmp_path / "missing.json", out=out))
    assert caught.value.code == contracts.EXIT_CODES["configuration_or_contract"]
    assert "non-empty" in capsys.readouterr().err
    assert marker.read_text() == "keep"


def test_pin_mismatch_exits_2(tmp_path, capsys):
    dataset, split = _write_synthetic(tmp_path)
    out = tmp_path / "out"
    with pytest.raises(SystemExit) as caught:
        cli_prepare(SimpleNamespace(dataset=dataset, split=split, out=out))
    assert caught.value.code == contracts.EXIT_CODES["configuration_or_contract"]
    assert "does not match pinned" in capsys.readouterr().err
    assert not out.exists()
