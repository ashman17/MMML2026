"""Render saved item 170 evidence over original benchmark images.

Run from the repository root with the 11777-project Python environment.
Marker positions are manual illustration anchors, not model localizations.
The lines visualize observation membership in shared entity IDs; they are
not additional identity_links, spatial claims, or computed correspondences.
"""
from __future__ import annotations

import io
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/11777-evidence-demo-mpl")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch, FancyBboxPatch
from PIL import Image
import pyarrow.parquet as pq

from idea4_world_state.inventory import parse_inventory_with_repairs
from idea4_world_state.serialize import parse_flat, parse_graph, render_graph


def main():
    runs = ROOT / "runs/idea4_world_state"
    run = runs / "eval200"
    out = Path(__file__).resolve().parent / "figures"
    out.mkdir(exist_ok=True)
    inv_path = run / "inventories/170.json"
    inv = json.loads(inv_path.read_text())
    record = json.loads((run / "records/170__graph.json").read_text())
    key = record["call_keys"]["extract"]
    cache = json.loads((runs / "_cache_v3" / key[:2] / f"{key}.json").read_text())
    parsed, repairs = parse_inventory_with_repairs(cache["text"], n_images=2)
    assert parsed == inv and repairs == []
    flat = (run / "serializations/170.flat.txt").read_text()
    graph = (run / "serializations/170.graph.txt").read_text()
    assert parse_flat(flat) == parse_graph(graph)
    assert render_graph(inv) == graph
    assert inv["claims"] == [] and inv["identity_links"] == []

    row = pq.read_table(
        ROOT / "data/MMSI_Bench.parquet",
        columns=["id", "images", "question", "answer"],
        filters=[("id", "=", 170)],
    ).to_pylist()[0]
    images = [Image.open(io.BytesIO(blob)).convert("RGB") for blob in row["images"]]
    assert len(images) == 2
    predictions = {}
    for condition in ("raw", "flat", "graph"):
        folder = runs / "raw_all" if condition == "raw" else run
        rec = json.loads((folder / f"records/170__{condition}.json").read_text())
        assert not rec["synthetic"] and rec["status"] == "ok"
        predictions[condition] = rec["pred"]
    assert predictions == {"raw": "D", "flat": "D", "graph": "B"}
    assert row["answer"] == "B"

    # Positions are chosen by visual inspection of the original photographs.
    # The inventory itself supplies only coarse `where` labels.
    positions = {
        "o1": (0.54, 0.52), "o2": (0.25, 0.24), "o3": (0.72, 0.75),
        "o4": (0.54, 0.52), "o5": (0.25, 0.24), "o6": (0.72, 0.75),
    }
    colors = {"e1": "#e6a23a", "e2": "#24b3cf", "e3": "#b687df"}
    names = {"e1": "Bottle", "e2": "Tray", "e3": "Gripper"}
    node_y = {"e1": 0.650, "e2": 0.790, "e3": 0.475}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "svg.fonttype": "none"})
    fig = plt.figure(figsize=(14, 9.4), facecolor="white")
    ink = "#203344"
    fig.text(0.035, 0.958, "From two views to an entity-grouped evidence inventory",
             fontsize=18, fontweight="bold", color=ink)
    fig.text(0.035, 0.921,
             "Item 170  |  In which direction is the camera rotating?  "
             "A: Left   B: Down   C: Right   D: Up", fontsize=11, color=ink)

    axes = []
    for i, x in enumerate((0.035, 0.565)):
        ax = fig.add_axes([x, 0.422, 0.40, 0.447])
        ax.imshow(images[i], extent=[0, 1, 1, 0])
        # The axes have the source 4:3 aspect; normalized coordinates must
        # fill those axes rather than forcing a square image.
        ax.set_aspect("auto")
        ax.set_xlim(0, 1)
        ax.set_ylim(1, 0)
        ax.axis("off")
        ax.set_title(f"Image {i+1}", loc="left", fontsize=11,
                     fontweight="bold", pad=7, color=ink)
        axes.append(ax)

    # Draw connectors first so observations remain legible over them.
    for obs in inv["observations"]:
        idx = int(obs["image_id"].split("_")[1]) - 1
        eid = obs["entity"]
        connector = ConnectionPatch(
            xyA=positions[obs["id"]], coordsA="data", axesA=axes[idx],
            xyB=(0.5, node_y[eid]), coordsB=fig.transFigure,
            color=colors[eid], lw=1.8, linestyle=(0, (4, 3)),
            alpha=0.95, zorder=3, clip_on=False,
        )
        fig.add_artist(connector)
    for obs in inv["observations"]:
        idx = int(obs["image_id"].split("_")[1]) - 1
        color = colors[obs["entity"]]
        x, y = positions[obs["id"]]
        ax = axes[idx]
        ax.scatter([x], [y], s=165, color=color, edgecolor="white",
                   linewidth=1.7, zorder=5)
        ax.annotate(obs["id"], (x, y), xytext=(10, -17),
                    textcoords="offset points", fontsize=10,
                    fontweight="bold", color="white", zorder=6,
                    bbox=dict(boxstyle="round,pad=.24", fc="#203344", ec="none", alpha=.9))
    for eid, y in node_y.items():
        fig.text(0.5, y, f"{eid}\n{names[eid]}", ha="center", va="center",
                 fontsize=10, fontweight="bold", color=ink, zorder=7,
                 bbox=dict(boxstyle="round,pad=.48", fc="white",
                           ec=colors[eid], lw=2))
    fig.text(0.5, 0.854, "Shared\nentity IDs", ha="center", va="center",
             fontsize=10, color=ink)
    fig.text(0.035, 0.387,
             "Dashed lines: observations grouped under the same entity "
             "(model cross-view status: same; uncertainty: medium).",
             fontsize=10, color=ink)
    fig.text(0.035, 0.364,
             "Markers are manually placed for illustration; the extractor "
             "provides coarse locations, not pixel coordinates.",
             fontsize=9, color="#536675")

    # One entity is sufficient to expose the actual schema and grouping.
    for x, width in ((0.035, 0.443), (0.505, 0.46)):
        fig.add_artist(FancyBboxPatch(
            (x, 0.071), width, 0.263, transform=fig.transFigure,
            boxstyle="round,pad=.010", facecolor="#f3f6f9",
            edgecolor="#d7e0e8", linewidth=0.8, zorder=0))
    fig.text(0.047, 0.309, "Extracted JSON: selected fields for e1",
             fontsize=11, fontweight="bold", color=ink)
    json_lines = [
        '"entities": [{"id": "e1", "type": "object",',
        '  "label": "glicerine bottle",',
        '  "cross_view": {"status": "same",',
        '                 "uncertainty": "medium"}}]',
        '"observations": [',
        '  {"id": "o1", "entity": "e1",',
        '   "image_id": "image_1", "where": "center", ...},',
        '  {"id": "o4", "entity": "e1",',
        '   "image_id": "image_2", "where": "center", ...}]',
        '"identity_links": [], "claims": []',
    ]
    fig.text(0.047, 0.283, "\n".join(json_lines), va="top",
             family="DejaVu Sans Mono", fontsize=9, linespacing=1.22, color=ink)
    fig.text(0.517, 0.309, "Graph presentation: same evidence atoms",
             fontsize=11, fontweight="bold", color=ink)
    graph_lines = [
        'e1 [object] "glicerine bottle"',
        '  cross-view: same (uncertainty medium)',
        '  o1 @image_1 center:',
        '    "glicerine bottle held by a robotic gripper"',
        '  o4 @image_2 center:',
        '    "glicerine bottle held by a robotic gripper"',
        'Edges by frame: none',
    ]
    fig.text(0.517, 0.283, "\n".join(graph_lines), va="top",
             family="DejaVu Sans Mono", fontsize=9, linespacing=1.22, color=ink)
    fig.text(0.517, 0.148, "Recorded unknown: camera rotation direction is not supported.",
             fontsize=9, color="#536675")
    fig.text(0.517, 0.106, "Raw: D (Up)     Flat: D (Up)     Graph: B (Down)     Gold: B",
             fontsize=10, fontweight="bold", color=ink)
    fig.text(0.035, 0.027,
             "Selected success case. Correct answering does not establish "
             "that grouping caused the correction; no spatial-relation edges were extracted.",
             fontsize=9, color="#536675")
    stem = out / "item170_evidence_organization"
    fig.savefig(stem.with_suffix(".png"), dpi=220, facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), facecolor="white")
    plt.close(fig)
    (out / "item170_inventory.json").write_text(json.dumps(inv, indent=2) + "\n")
    (out / "item170_model_extraction.txt").write_text(cache["text"] + "\n")
    (out / "item170_graph.txt").write_text(graph)
    (out / "item170_flat.txt").write_text(flat)
    metadata = {
        "item_id": 170, "source": "data/MMSI_Bench.parquet",
        "image_sizes": [list(im.size) for im in images],
        "inventory_source": str(inv_path.relative_to(ROOT)),
        "extraction_cache_key": key, "parser_repairs": repairs,
        "predictions": predictions, "gold": row["answer"],
        "selection": "Outcome-selected Graph-only success; descriptive example.",
        "positions": positions,
        "positions_source": "Manually placed after visual inspection, not model output.",
        "lines": "Observation membership in shared entity IDs; not spatial-relation edges.",
        "schema_excerpt": "Selected fields only; ellipses omit observation descriptions.",
        "graph_excerpt": "Entity e1 only; cross-view metadata placed on its own line for layout.",
        "checks": "Raw extraction matches inventory; no repairs; Flat and Graph atoms match.",
    }
    (out / "item170_demo_provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"figure": str(stem.with_suffix('.png')), **metadata}, indent=2))


if __name__ == "__main__":
    main()
