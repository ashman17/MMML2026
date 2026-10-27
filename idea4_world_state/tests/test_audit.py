"""Blind audit packet, label import, and outcome lists."""

import csv
import hashlib
import json
import random
import re
import shutil
from pathlib import Path
from types import SimpleNamespace

import pyarrow.parquet as pq
import pytest
from PIL import Image

from idea4_world_state import contracts
from idea4_world_state.audit import cli_audit_import, cli_audit_make, cli_audit_outcomes
from idea4_world_state.inventory import InventoryError, atom_id, atom_sort_key, atoms, parse_inventory
from idea4_world_state.tests.audit_fixtures import (
    ERROR_INVENTORY,
    OPTION_TEXTS,
    PARQUET_ANSWER,
    PARQUET_CATEGORY,
    PARQUET_THOUGHT,
    RECORD_PRED,
    valid_inventory,
    write_audit_run,
)

POOL_SPECS = [
    {"id": "s1", "category": "spatial", "sizes": [(1200, 900), (8, 8)]},
    {"id": "s2", "category": "spatial"},
    {"id": "s10", "category": "spatial"},
    {"id": "m1", "category": "motion"},
    {"id": "m2", "category": "motion"},
    {"id": "bad-json", "category": "spatial", "inventory": "invalid"},
    {"id": "bad-missing", "category": "motion", "inventory": "missing"},
]
SMALL_SPECS = [
    {"id": "s1", "category": "spatial"},
    {"id": "m1", "category": "motion"},
]
# id, category, gold, raw, flat, graph. Each triple is (status, pred, correct).
OUTCOME_MATRIX = [
    ("p1", "spatial", "A", ("ok", "A", "1"), ("ok", "B", "0"), ("ok", "A", "1")),
    ("p2", "spatial", "B", ("ok", "C", "0"), ("ok", "B", "1"), ("answer_parse_error", "B", "0")),
    ("m1", "motion", "C", ("ok", "A", "0"), ("ok", "B", "0"), ("ok", "C", "1")),
    ("bad", "motion", "A", ("missing", "", "0"), ("ok", "A", "1"), ("ok", "B", "0")),
]
# X-Y: corrected is Y wrong and X right; introduced is Y right and X wrong.
EXPECTED_OUTCOMES = [
    ("graph-flat", "bad", "B", "A", "A", "introduced"),
    ("graph-flat", "m1", "C", "B", "C", "corrected"),
    ("graph-flat", "p1", "A", "B", "A", "corrected"),
    ("graph-flat", "p2", "B", "B", "B", "introduced"),
    ("flat-raw", "bad", "A", "", "A", "corrected"),
    ("flat-raw", "p1", "B", "A", "A", "introduced"),
    ("flat-raw", "p2", "B", "C", "B", "corrected"),
    ("graph-raw", "m1", "C", "A", "C", "corrected"),
]
def spec_draw(eligible, n, seed=0):
    """Independent reading of the Section 6 draw.

    One Random stream. Shuffle sorted category labels, then shuffle each
    category's numerically sorted ids (non-numeric ids after, as text) in
    that shuffled category order. Round-robin list.pop() until n items or
    every bucket is empty.
    """
    rng = random.Random(seed)
    by_category = {}
    for item_id, category in eligible:
        by_category.setdefault(category, []).append(item_id)
    categories = sorted(by_category)
    rng.shuffle(categories)
    buckets = []
    for category in categories:
        numeric = sorted((i for i in by_category[category] if str(i).isdigit()), key=int)
        other = sorted(i for i in by_category[category] if not str(i).isdigit())
        ids = numeric + other
        rng.shuffle(ids)
        buckets.append(ids)
    drawn = []
    while len(drawn) < n:
        progressed = False
        for bucket in buckets:
            if not bucket:
                continue
            drawn.append(bucket.pop())
            progressed = True
            if len(drawn) >= n:
                break
        if not progressed:
            break
    return drawn


def test_make_packet_hides_options_gold_and_predictions(tmp_path, capsys):
    run = _pool_run(tmp_path)
    audit_dir = tmp_path / "audit"
    cli_audit_make(SimpleNamespace(run=run, n=20, out=audit_dir))
    captured = capsys.readouterr()
    assert re.search(r"excluded 2\b", captured.out)

    packet = json.loads((audit_dir / "packet.json").read_text())
    assert packet["excluded_count"] == 2
    assert packet["seed"] == 0
    assert set(packet["drawn_ids"]) == {"s1", "s2", "s10", "m1", "m2"}
    assert "bad-json" not in packet["drawn_ids"]
    assert "bad-missing" not in packet["drawn_ids"]

    text = _packet_text(audit_dir)
    for secret in (OPTION_TEXTS[0], OPTION_TEXTS[1], OPTION_TEXTS[2], OPTION_TEXTS[3],
                   PARQUET_ANSWER, PARQUET_THOUGHT, PARQUET_CATEGORY, RECORD_PRED, ERROR_INVENTORY):
        assert secret not in text
    html = (audit_dir / "index.html").read_text()
    assert "Options:" not in html
    assert "Where is the door in item s1?" in html
    assert "Excluded for invalid inventory: 2" in html

    items = _read_csv(audit_dir / "items.csv")
    atom_rows = _read_csv(audit_dir / "atom_labels.csv")
    labels = _read_csv(audit_dir / "item_labels.csv")
    assert [row["item_id"] for row in items] == packet["drawn_ids"]
    assert {row["category"] for row in items} == {"spatial", "motion"}
    for row in items:
        digest = hashlib.sha256((run / "inventories" / f"{row['item_id']}.json").read_bytes()).hexdigest()
        assert row["inventory_sha256"] == digest
        assert packet["inventory_sha256"][row["item_id"]] == digest
    _assert_blank_labels(atom_rows, ("accuracy", "relevance", "reviewer", "notes"))
    _assert_blank_labels(labels, (
        "query_frame_correct", "identity_error", "frame_error",
        "missing_critical_evidence", "reviewer", "notes",
    ))
    document = parse_inventory((run / "inventories" / "s1.json").read_text(), 2)
    expected_ids = [atom_id(atom) for atom in sorted(atoms(document), key=atom_sort_key)]
    got_ids = [row["atom_id"] for row in atom_rows if row["item_id"] == "s1"]
    assert got_ids == expected_ids
    assert expected_ids[:4] == ["query_frame", "temporal_order", "cam1", "cam2"]
    section = _section(html, "s1")
    for number in ("A1", "A2", "A3", "A4", "A5"):
        assert f'data-atom="{number}"' in section
    assert 'data-atom="A6"' not in section
    assert "inventory note for s1, unchecked" in section
    with Image.open(audit_dir / "images" / "s1_1.jpg") as image:
        assert image.format == "JPEG"
        assert image.size == (768, 576)
    with Image.open(audit_dir / "images" / "s1_2.jpg") as image:
        assert image.size == (8, 8)


def test_draw_is_deterministic_and_ignores_records(tmp_path):
    assert contracts.SEED == 0
    run = _pool_run(tmp_path)
    eligible, excluded = _eligible(run)
    assert excluded == 2
    expected = spec_draw(eligible, 4, seed=0)
    first = tmp_path / "out1"
    second = tmp_path / "out2"
    cli_audit_make(SimpleNamespace(run=run, n=4, out=first))
    cli_audit_make(SimpleNamespace(run=run, n=4, out=second))
    assert _drawn(first) == expected
    for name in ("index.html", "items.csv", "atom_labels.csv", "item_labels.csv", "packet.json"):
        assert (first / name).read_bytes() == (second / name).read_bytes()

    for path in (run / "records").glob("*.json"):
        path.write_text(path.read_text().replace(RECORD_PRED, "CHANGED_PRED"), encoding="utf-8")
    third = tmp_path / "out3"
    cli_audit_make(SimpleNamespace(run=run, n=4, out=third))
    assert _drawn(third) == expected

    shutil.rmtree(run / "records")
    (run / "records").write_text(RECORD_PRED, encoding="utf-8")
    fourth = tmp_path / "out4"
    cli_audit_make(SimpleNamespace(run=run, n=4, out=fourth))
    assert _drawn(fourth) == expected
    assert RECORD_PRED not in _packet_text(fourth)
    assert "CHANGED_PRED" not in _packet_text(fourth)


def test_draw_sorts_ids_numerically_before_shuffling():
    from idea4_world_state.audit import _draw

    ids = ["2", "11", "3", "100", "25", "7"]
    eligible = [{"item_id": i, "category": "A"} for i in ids]
    expected = spec_draw([(i, "A") for i in ids], len(ids), seed=0)
    assert _draw(eligible, len(ids)) == expected
    lexicographic = sorted(ids)
    rng = random.Random(0)
    rng.shuffle(["A"])
    rng.shuffle(lexicographic)
    assert expected != list(reversed(lexicographic))


def test_draw_respects_n_and_exhaustion(tmp_path):
    run = _pool_run(tmp_path)
    eligible, _excluded = _eligible(run)
    one = tmp_path / "one"
    all_items = tmp_path / "all"
    cli_audit_make(SimpleNamespace(run=run, n=1, out=one))
    cli_audit_make(SimpleNamespace(run=run, n=100, out=all_items))
    assert _drawn(one) == spec_draw(eligible, 1, seed=0)
    assert len(_drawn(one)) == 1
    assert _drawn(all_items) == spec_draw(eligible, 100, seed=0)
    assert _drawn(all_items) == spec_draw(eligible, len(eligible), seed=0)
    assert len(_drawn(all_items)) == len(eligible)


def test_invalid_inventories_are_excluded(tmp_path, capsys):
    run = _pool_run(tmp_path)
    (run / "inventories" / "extra.json").write_text(
        json.dumps(valid_inventory("extra")) + "\n", encoding="utf-8"
    )
    audit_dir = tmp_path / "audit"
    cli_audit_make(SimpleNamespace(run=run, n=20, out=audit_dir))
    assert re.search(r"excluded 2\b", capsys.readouterr().out)
    packet = json.loads((audit_dir / "packet.json").read_text())
    assert packet["excluded_count"] == 2
    assert "extra" not in packet["drawn_ids"]
    assert "bad-json" not in packet["drawn_ids"]
    assert "bad-missing" not in packet["drawn_ids"]
    assert ERROR_INVENTORY not in _packet_text(audit_dir)


def test_integer_dataset_ids_match_run_config(tmp_path):
    specs = [
        {"id": 11, "category": "spatial", "stem": "Integer id stem for 11"},
        {"id": 12, "category": "spatial"},
        {"id": 13, "category": "motion"},
    ]
    run, _dataset = write_audit_run(tmp_path / "case", specs, int_ids=True)
    config = json.loads((run / "run_config.json").read_text())
    assert config["ids"] == [11, 12, 13]
    audit_dir = tmp_path / "audit"
    cli_audit_make(SimpleNamespace(run=run, n=3, out=audit_dir))
    packet = json.loads((audit_dir / "packet.json").read_text())
    assert set(packet["drawn_ids"]) == {"11", "12", "13"}
    text = _packet_text(audit_dir)
    assert "Integer id stem for 11" in text
    assert OPTION_TEXTS[0] not in text
    assert (run / "inventories" / "11.json").is_file()


def test_refuses_nonempty_output_and_bad_config(tmp_path, capsys):
    run, _dataset = write_audit_run(tmp_path / "case", SMALL_SPECS)
    out = tmp_path / "audit"
    out.mkdir()
    marker = out / "keep.txt"
    marker.write_text("keep")
    with pytest.raises(SystemExit) as caught:
        cli_audit_make(SimpleNamespace(run=run, n=20, out=out))
    assert caught.value.code == 2
    assert "non-empty" in capsys.readouterr().err
    assert marker.read_text() == "keep"

    file_out = tmp_path / "file-out"
    file_out.write_text("x")
    with pytest.raises(SystemExit) as caught:
        cli_audit_make(SimpleNamespace(run=run, n=20, out=file_out))
    assert caught.value.code == 2
    assert "not a directory" in capsys.readouterr().err
    assert file_out.read_text() == "x"

    fresh = tmp_path / "negative"
    with pytest.raises(SystemExit) as caught:
        cli_audit_make(SimpleNamespace(run=run, n=-1, out=fresh))
    assert caught.value.code == 2
    assert "non-negative" in capsys.readouterr().err
    assert not fresh.exists()

    config_path = run / "run_config.json"
    config = json.loads(config_path.read_text())
    config["dataset_sha256"] = "0" * 64
    config_path.write_text(json.dumps(config), encoding="utf-8")
    mismatched = tmp_path / "mismatched"
    with pytest.raises(SystemExit) as caught:
        cli_audit_make(SimpleNamespace(run=run, n=20, out=mismatched))
    assert caught.value.code == 2
    assert "sha256" in capsys.readouterr().err
    assert not mismatched.exists()

    config = json.loads(config_path.read_text())
    config["dataset_sha256"] = hashlib.sha256(Path(config["dataset_path"]).read_bytes()).hexdigest()
    del config["items"][0]["n_images"]
    config_path.write_text(json.dumps(config), encoding="utf-8")
    missing_images = tmp_path / "missing-images"
    with pytest.raises(SystemExit) as caught:
        cli_audit_make(SimpleNamespace(run=run, n=20, out=missing_images))
    assert caught.value.code == 2
    assert "n_images" in capsys.readouterr().err
    assert not missing_images.exists()


def test_parquet_reads_only_inference_columns(tmp_path, monkeypatch):
    run, analysis = _outcome_case(tmp_path)
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
    packet = run / "audit"
    cli_audit_make(SimpleNamespace(run=run, n=20, out=packet))
    labels = tmp_path / "labels"
    _fill(packet, labels, None)
    cli_audit_import(SimpleNamespace(packet=packet, labels=labels))
    cli_audit_outcomes(SimpleNamespace(run=run, analysis=analysis, out=tmp_path / "outcomes"))
    assert len(recorded) >= 2
    for columns in recorded:
        assert columns is not None
        assert set(columns) <= set(contracts.INFERENCE_COLUMNS)
        assert "answer" not in columns
        assert "thought" not in columns


@pytest.mark.parametrize(
    ("mutation", "needle"),
    [
        ("drop", "missing"),
        ("duplicate", "duplicate"),
        ("extra", "unexpected"),
        ("atom_text", "atom_text"),
        ("accuracy", "outside"),
        ("relevance", "outside"),
        ("query_frame", "outside"),
        ("identity", "outside"),
        ("reviewer_atom", "reviewer"),
        ("reviewer_item", "reviewer"),
    ],
)
def test_import_rejects_bad_labels(tmp_path, capsys, mutation, needle):
    _run, packet = _small_packet(tmp_path)
    labels = tmp_path / "labels"
    _fill(packet, labels, mutation)
    with pytest.raises(SystemExit) as caught:
        cli_audit_import(SimpleNamespace(packet=packet, labels=labels))
    assert caught.value.code == 2
    assert needle in capsys.readouterr().err
    assert not (packet / "imported.json").exists()


def test_import_rejects_changed_inventory(tmp_path, capsys):
    run, packet = _small_packet(tmp_path)
    labels = tmp_path / "labels"
    _fill(packet, labels, None)
    path = run / "inventories" / "s1.json"
    path.write_text(path.read_text().replace("unchecked", "changed"), encoding="utf-8")
    with pytest.raises(SystemExit) as caught:
        cli_audit_import(SimpleNamespace(packet=packet, labels=labels))
    assert caught.value.code == 2
    assert "hash" in capsys.readouterr().err
    assert not (packet / "imported.json").exists()


def test_import_accepts_external_labels_without_prefilling(tmp_path, capsys):
    _run, packet = _small_packet(tmp_path)
    with pytest.raises(SystemExit) as caught:
        cli_audit_import(SimpleNamespace(packet=packet, labels=packet))
    assert caught.value.code == 2
    assert "outside" in capsys.readouterr().err
    assert not (packet / "imported.json").exists()

    labels = tmp_path / "labels"
    _fill(packet, labels, None)
    before_atom = (labels / "atom_labels.csv").read_bytes()
    before_item = (labels / "item_labels.csv").read_bytes()
    blank_atoms = (packet / "atom_labels.csv").read_bytes()
    blank_items = (packet / "item_labels.csv").read_bytes()
    cli_audit_import(SimpleNamespace(packet=packet, labels=labels))
    assert (labels / "atom_labels.csv").read_bytes() == before_atom
    assert (labels / "item_labels.csv").read_bytes() == before_item
    assert (packet / "atom_labels.csv").read_bytes() == blank_atoms
    assert (packet / "item_labels.csv").read_bytes() == blank_items
    _assert_blank_labels(_read_csv(packet / "atom_labels.csv"), ("accuracy", "relevance", "reviewer", "notes"))
    imported = json.loads((packet / "imported.json").read_text())
    assert imported["atom_labels_sha256"] == hashlib.sha256(before_atom).hexdigest()
    assert imported["item_labels_sha256"] == hashlib.sha256(before_item).hexdigest()
    assert imported["atom_count"] == len(_read_csv(labels / "atom_labels.csv"))
    assert imported["item_count"] == len(_read_csv(labels / "item_labels.csv"))
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", imported["imported_at"])


def test_import_accepts_inplace_packet(tmp_path):
    _run, packet = _small_packet(tmp_path)
    _fill(packet, packet, None)
    before_atom = (packet / "atom_labels.csv").read_bytes()
    before_item = (packet / "item_labels.csv").read_bytes()
    cli_audit_import(SimpleNamespace(packet=packet, labels=packet))
    assert (packet / "atom_labels.csv").read_bytes() == before_atom
    assert (packet / "item_labels.csv").read_bytes() == before_item
    assert (packet / "imported.json").is_file()
    assert _read_csv(packet / "atom_labels.csv")[0]["accuracy"] == "supported"


def test_outcomes_refuses_before_import(tmp_path, capsys):
    run, analysis = _outcome_case(tmp_path)
    out = tmp_path / "audit_outcomes"
    with pytest.raises(SystemExit) as caught:
        cli_audit_outcomes(SimpleNamespace(run=run, analysis=analysis, out=out))
    assert caught.value.code == 2
    assert "imported.json" in capsys.readouterr().err
    assert not out.exists()

    packet = run / "audit"
    cli_audit_make(SimpleNamespace(run=run, n=20, out=packet))
    with pytest.raises(SystemExit) as caught:
        cli_audit_outcomes(SimpleNamespace(run=run, analysis=analysis, out=out))
    assert caught.value.code == 2
    assert "imported.json" in capsys.readouterr().err
    assert not out.exists()

    (packet / "imported.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(SystemExit) as caught:
        cli_audit_outcomes(SimpleNamespace(run=run, analysis=analysis, out=out))
    assert caught.value.code == 2
    assert "imported.json" in capsys.readouterr().err
    assert not out.exists()


def test_outcomes_match_hand_built_item_outcomes(tmp_path, capsys):
    run, analysis = _outcome_case(tmp_path)
    packet = run / "audit"
    cli_audit_make(SimpleNamespace(run=run, n=20, out=packet))
    make_text = _packet_text(packet)
    assert OPTION_TEXTS[0] not in make_text
    assert PARQUET_ANSWER not in make_text
    assert RECORD_PRED not in make_text
    assert "bad" not in json.loads((packet / "packet.json").read_text())["drawn_ids"]

    labels = tmp_path / "labels"
    _fill(packet, labels, None)
    cli_audit_import(SimpleNamespace(packet=packet, labels=labels))
    out = tmp_path / "audit_outcomes"
    cli_audit_outcomes(SimpleNamespace(run=run, analysis=analysis, out=out))
    rows = _read_csv(out / "outcomes.csv")
    assert list(csv.DictReader((out / "outcomes.csv").open(encoding="utf-8")).fieldnames) == list(
        contracts.AUDIT_OUTCOME_COLUMNS
    )
    got = [
        (row["comparison"], row["item_id"], row["x_pred"], row["y_pred"], row["gold"], row["direction"])
        for row in rows
    ]
    assert got == EXPECTED_OUTCOMES
    for row in rows:
        assert row["primary_error_type"] == ""
        assert row["reviewer"] == ""
        assert row["notes"] == ""
    for error_type in contracts.PRIMARY_ERROR_TYPES:
        if error_type == "other":
            continue
        assert error_type not in (out / "outcomes.csv").read_text()
    html = (out / "index.html").read_text()
    assert OPTION_TEXTS[0] in html
    assert "Options:" in html
    assert "Gold: A" in html
    assert "Gold: C" in html
    assert "No schema-valid inventory." in html
    assert "inventory note for p1, unchecked" in html
    assert PARQUET_ANSWER not in html
    assert PARQUET_THOUGHT not in html
    assert RECORD_PRED not in html
    assert (out / "images" / "bad_1.jpg").is_file()
    assert not (packet / "images" / "bad_1.jpg").exists()
    produced = sorted(path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*") if path.is_file())
    assert "case/run/audit/index.html" in produced
    assert "case/run/audit/imported.json" in produced
    assert "audit_outcomes/outcomes.csv" in produced
    assert "audit_outcomes/index.html" in produced

    with pytest.raises(SystemExit) as caught:
        cli_audit_outcomes(SimpleNamespace(run=run, analysis=analysis, out=out))
    assert caught.value.code == 2
    assert "non-empty" in capsys.readouterr().err
    assert got == [
        (row["comparison"], row["item_id"], row["x_pred"], row["y_pred"], row["gold"], row["direction"])
        for row in _read_csv(out / "outcomes.csv")
    ]


def _pool_run(tmp_path):
    run, _dataset = write_audit_run(tmp_path / "case", POOL_SPECS)
    return run


def _small_packet(tmp_path):
    run, _dataset = write_audit_run(tmp_path / "case", SMALL_SPECS)
    packet = tmp_path / "audit"
    cli_audit_make(SimpleNamespace(run=run, n=20, out=packet))
    return run, packet


def _outcome_case(tmp_path):
    specs = [
        {"id": item_id, "category": category, "inventory": "missing" if item_id == "bad" else "valid"}
        for item_id, category, _gold, _raw, _flat, _graph in OUTCOME_MATRIX
    ]
    run, _dataset = write_audit_run(tmp_path / "case", specs, run_id="outcome-run")
    analysis = tmp_path / "case" / "analysis"
    analysis.mkdir()
    rows = []
    for condition in ("graph", "raw", "flat"):
        for item_id, category, gold, raw, flat, graph in reversed(OUTCOME_MATRIX):
            status, pred, correct = {"raw": raw, "flat": flat, "graph": graph}[condition]
            rows.append({
                "id": item_id,
                "category": category,
                "condition": condition,
                "status": status,
                "pred": pred,
                "gold": gold,
                "correct": correct,
            })
    _write_csv(analysis / "item_outcomes.csv", (
        "id", "category", "condition", "status", "pred", "gold", "correct",
    ), rows)
    return run, analysis


def _eligible(run):
    config = json.loads((run / "run_config.json").read_text())
    meta = {str(item["id"]): item for item in config["items"]}
    eligible = []
    excluded = 0
    for raw in config["ids"]:
        item_id = str(raw)
        path = run / "inventories" / f"{item_id}.json"
        ok = path.is_file()
        if ok:
            try:
                parse_inventory(path.read_text(encoding="utf-8"), meta[item_id]["n_images"])
            except InventoryError:
                ok = False
        if ok:
            eligible.append((item_id, meta[item_id]["category"]))
        else:
            excluded += 1
    return eligible, excluded


def _fill(packet, dest, mutation):
    dest.mkdir(parents=True, exist_ok=True)
    atom_rows = _read_csv(packet / "atom_labels.csv")
    item_rows = _read_csv(packet / "item_labels.csv")
    for row in atom_rows:
        row["accuracy"] = "supported"
        row["relevance"] = "relevant"
        row["reviewer"] = "reviewer-1"
        row["notes"] = ""
    for row in item_rows:
        row["query_frame_correct"] = "yes"
        row["identity_error"] = "no"
        row["frame_error"] = "no"
        row["missing_critical_evidence"] = "no"
        row["reviewer"] = "reviewer-1"
        row["notes"] = "checked"
    if mutation == "drop":
        atom_rows = atom_rows[1:]
    elif mutation == "duplicate":
        atom_rows = atom_rows + [dict(atom_rows[0])]
    elif mutation == "extra":
        atom_rows = atom_rows + [{
            "item_id": atom_rows[0]["item_id"],
            "atom_id": "not-an-atom",
            "atom_text": "extra",
            "accuracy": "supported",
            "relevance": "relevant",
            "reviewer": "reviewer-1",
            "notes": "",
        }]
    elif mutation == "atom_text":
        atom_rows[0]["atom_text"] = "rewritten atom"
    elif mutation == "accuracy":
        atom_rows[0]["accuracy"] = "nope"
    elif mutation == "relevance":
        atom_rows[0]["relevance"] = "sometimes"
    elif mutation == "query_frame":
        item_rows[0]["query_frame_correct"] = "true"
    elif mutation == "identity":
        item_rows[0]["identity_error"] = "cannot_judge"
    elif mutation == "reviewer_atom":
        atom_rows[0]["reviewer"] = "   "
    elif mutation == "reviewer_item":
        item_rows[0]["reviewer"] = ""
    elif mutation is not None:
        raise AssertionError(mutation)
    _write_csv(dest / "atom_labels.csv", contracts.AUDIT_ATOM_COLUMNS, atom_rows)
    _write_csv(dest / "item_labels.csv", contracts.AUDIT_ITEM_LABEL_COLUMNS, item_rows)


def _drawn(audit_dir):
    return json.loads((audit_dir / "packet.json").read_text())["drawn_ids"]


def _packet_text(audit_dir):
    names = ("index.html", "items.csv", "atom_labels.csv", "item_labels.csv", "packet.json")
    return "\n".join((audit_dir / name).read_text(encoding="utf-8") for name in names)


def _section(document, item_id):
    marker = f'data-item-id="{item_id}"'
    start = document.index(marker)
    start = document.rfind("<section", 0, start)
    end = document.find("</section>", start)
    return document[start:end]


def _assert_blank_labels(rows, columns):
    assert rows
    for row in rows:
        for column in columns:
            assert row[column] == ""


def _read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path, fieldnames, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
