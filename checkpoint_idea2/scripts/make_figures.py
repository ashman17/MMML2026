#!/usr/bin/env python3
"""Generate the checkpoint's plots using archived metrics and selected RGB frames."""

import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import ticker

ROOT = Path(__file__).resolve().parents[1]
METRICS = json.loads((ROOT / "evidence/metrics.json").read_text())
INK = "#182a31"
MUTED = "#455b62"
GRID = "#dce6e3"
TEAL = "#0d746b"
ORANGE = "#ad4d27"
GOLD = "#ffd279"
RAW = "#61727b"
BLUE = "#416181"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "text.color": INK,
    "axes.labelcolor": INK, "axes.edgecolor": GRID, "axes.spines.top": False,
    "axes.spines.right": False, "axes.spines.left": False, "xtick.color": MUTED,
    "ytick.color": INK, "xtick.major.size": 0, "ytick.major.size": 0,
    "axes.axisbelow": True, "grid.color": GRID, "svg.fonttype": "none",
    "pdf.fonttype": 42, "savefig.facecolor": "white",
})


def save(fig, name):
    for ext in ("pdf", "svg"):
        fig.savefig(ROOT / "figures" / f"{name}.{ext}")
    plt.close(fig)


def mmsi_headlines():
    base = METRICS["mmsi"]["conditions"]
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.28), sharey=True)
    fig.subplots_adjust(left=.335, right=.91, top=.85, bottom=.23, wspace=.22)
    lines = [
        ("Images + question", ("raw",), RAW, "o"),
        ("+ pose text", ("vggt_heuristic",), BLUE, "s"),
        ("Adapted + pose text", ("boot_lora_pose_text_s42", "boot_lora_pose_text_s123"), TEAL, "D"),
    ]
    for ax, (subset, title) in zip(axes, [("all_1000", "All questions · 1,000"),
                                        ("msr_198", "Multi-image reasoning · 198")]):
        for y, (_, keys, color, marker) in zip([2, 1, 0], lines):
            if len(keys) == 2:
                vals = [base[key]["subsets"][subset] * 100 for key in keys]
                ax.vlines(vals, y - .13, y + .13, color=color, lw=1.0)
                ax.plot(vals, [y + .13, y - .13], marker, color=color, ms=4.8, linestyle="none")
                for value, shift in zip(vals, [.15, -.15]):
                    ax.text(value + .49, y + shift, f"{value:.1f}", va="center", color=color, fontsize=9)
            else:
                value = base[keys[0]]["subsets"][subset] * 100
                ax.plot(value, y, marker, color=color, ms=5.5)
                ax.text(value + .5, y, f"{value:.1f}", va="center", color=color, fontsize=9)
        ax.set_xlim(24, 41.2)
        ax.set_ylim(-.43, 2.43)
        ax.set_xticks([25, 30, 35, 40])
        ax.set_yticks([2, 1, 0], [line[0] for line in lines], fontsize=9)
        ax.grid(axis="x")
        ax.set_title(title, loc="left", fontsize=9.5, fontweight="bold", pad=8)
    fig.supxlabel("Accuracy (%)", y=.025, fontsize=9)
    save(fig, "mmsi_headlines")


def mmsi_categories():
    b = METRICS["mmsi"]["conditions"]
    raw = b["raw"]["by_category"]
    names = {
        "Attribute (Appr.)": "Appearance", "Attribute (Meas.)": "Measurement",
        "MSR": "Multi-image reasoning", "Motion (Cam.)": "Camera motion", "Motion (Obj.)": "Object motion",
    }
    fig, ax = plt.subplots(figsize=(7.7, 4.24))
    fig.subplots_adjust(left=.278, right=.98, top=.89, bottom=.15)
    conditions = [
        ("vggt_heuristic", BLUE, "s", "Pose text"), ("boot_lora_pose_text_s42", TEAL, "o", "Adapted + pose · seed 42"),
        ("boot_lora_pose_text_s123", ORANGE, "^", "Adapted + pose · seed 123"),
    ]
    ax.axvline(0, color=RAW, linewidth=1)
    for offset, (condition, color, marker, label) in zip((.2, 0., -.2), conditions):
        values = [100 * (b[condition]["by_category"][key]["acc"] - row["acc"]) for key, row in raw.items()]
        ax.scatter(values, [i + offset for i in reversed(range(len(raw)))], label=label, color=color,
                   marker=marker, s=27, zorder=3)
    labels = [f"{names.get(key, key.removeprefix('Positional Relationship '))} · {row['n']}" for key, row in raw.items()]
    ax.set_yticks(list(reversed(range(len(raw)))), labels)
    ax.set_ylim(-.6, len(raw) - .4)
    ax.grid(axis="y", linewidth=.7)
    ax.set_xlabel("Accuracy difference from image-only Qwen3-VL (percentage points)")
    ax.legend(loc="lower left", bbox_to_anchor=(-.012, 1.01), ncols=3, frameon=False, fontsize=9)
    save(fig, "mmsi_categories")


def frame_reader():
    reader = METRICS["reader"]
    rows = [
        ("Images + question", "verified", "images_only"),
        ("Coordinates for both frames", "verified", "both_frames"),
        ("Routed + threshold applied", "verified", "routed_clamped"),
        ("Routed + threshold applied", "predicted_v2", "routed_clamped"),
    ]
    fig, ax = plt.subplots(figsize=(6.3, 2.28))
    fig.subplots_adjust(left=.368, right=.88, top=.78, bottom=.255)
    for y, (label, evidence, key) in zip([3, 2, 1, 0], rows):
        datum = reader[evidence][key]
        color = ORANGE if evidence == "predicted_v2" else TEAL if key != "images_only" else RAW
        ax.hlines(y, 0, datum["paired"], color=color, linewidth=2)
        marker = "s" if evidence == "predicted_v2" else "o"
        ax.plot(datum["paired"], y, marker, color=color, ms=6.5)
        ax.text(datum["paired"] + .72, y, f"{datum['paired']}/{datum['n']}", color=color, va="center",
                fontsize=9.3, fontweight="bold")
    ax.set_yticks([3, 2, 1, 0], [x[0] for x in rows], fontsize=9)
    ax.set_xticks([0, 6, 12, 18, 24])
    ax.set_xlim(0, 25.8)
    ax.set_ylim(-.4, 3.4)
    ax.grid(axis="x")
    ax.set_xlabel("Both questions correct on the same pair", fontsize=9, labelpad=5)
    ax.text(0, 1.2, "●  Verified coordinates", transform=ax.transAxes, color=TEAL, fontsize=9.1)
    ax.text(.60, 1.2, "■  Predicted v2", transform=ax.transAxes, color=ORANGE, fontsize=9.1)
    save(fig, "reader_interface")


def motion_bands():
    motion = METRICS["motion"]["bands"]
    fig, ax = plt.subplots(figsize=(6.25, 2.4))
    fig.subplots_adjust(left=.30, right=.91, top=.92, bottom=.25)
    for y, (name, datum) in zip([2, 1, 0], motion.items()):
        ax.barh(y, datum["n"], height=.53, color=GRID)
        ax.barh(y, datum["false_still"], height=.53, color=ORANGE)
        ax.text(datum["false_still"] + .27, y, f"{datum['false_still']}/{datum['n']}",
                color=INK, va="center", fontsize=9.4, fontweight="bold")
    ax.set_yticks([2, 1, 0], ["Small", "Moderate", "Large"])
    ax.set_xticks([0, 3, 6, 9, 12])
    ax.set_xlim(0, 12.25)
    ax.grid(axis="x", alpha=.7)
    ax.set_xlabel("Moving targets predicted to be still", fontsize=9, labelpad=5)
    save(fig, "motion_bands")


def displacements():
    with (ROOT / "evidence/fresh_v2_displacements.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    fig, axes = plt.subplots(1, 2, figsize=(7.55, 3.42))
    fig.subplots_adjust(left=.107, right=.98, bottom=.20, top=.90, wspace=.29)
    extent = math.ceil(max(abs(float(row["gold_dx"])) for row in rows) * 10) / 10
    for ax, lim, title in zip(axes, (extent, .24), ("All 48 fresh scenes", "Near the still threshold")):
        ax.axvspan(-.05, .05, color=GRID, alpha=.5)
        ax.axhspan(-.05, .05, color=GRID, alpha=.5)
        ax.plot([-lim, lim], [-lim, lim], "--", color=RAW, lw=1)
        for false, color, marker in [(False, TEAL, "o"), (True, ORANGE, "^")]:
            selected = [row for row in rows if (row["false_still"] == "True") == false]
            ax.scatter([float(row["gold_dx"]) for row in selected],
                       [float(row["aligned_dx"]) for row in selected], color=color, marker=marker,
                       s=25, edgecolor="white", linewidth=.35, zorder=3)
        ax.set(xlim=(-lim, lim), ylim=(-lim, lim), aspect="equal")
        ax.xaxis.set_major_locator(ticker.MaxNLocator(5))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
        ax.set_title(title, loc="left", fontsize=10, fontweight="bold")
        ax.set_xlabel("Verified displacement", fontsize=9)
    axes[0].set_ylabel("Scale-aligned prediction", fontsize=9)
    fig.text(.22, .032, "▲  False still at the deployed threshold", color=ORANGE, fontsize=8.3)
    fig.text(.64, .032, "●  Other outcomes", color=TEAL, fontsize=8.3)
    save(fig, "displacement_audit")


def threshold_tradeoff():
    s = METRICS["motion"]["fresh"]
    transitions = METRICS["motion"]["threshold_transitions"]
    fig, axes = plt.subplots(1, 2, figsize=(7.75, 2.63), gridspec_kw={"width_ratios": [2.2, 1]})
    fig.subplots_adjust(left=.248, right=.95, bottom=.19, top=.87, wspace=.46)
    for y, label, key in [(2, "Both answers correct", "pair"), (1, "Moving called still", "false_still"),
                          (0, "True-still recall", "still_tp")]:
        n = {"pair": 48, "false_still": 36, "still_tp": 12}[key]
        x, z = s["v2"][key], s["candidate"][key]
        axes[0].plot([100 * x / n, 100 * z / n], [y, y], color=GRID, lw=3)
        for val, col, mk, yoff in [(x, BLUE, "o", .16), (z, ORANGE, "^", -.16)]:
            p = val / n * 100
            axes[0].plot(p, y, mk, color=col, ms=5.7)
            axes[0].text(p, y + yoff, f"{val}/{n}", ha="center",
                         va="bottom" if yoff > 0 else "top", color=col, fontsize=8.1)
    axes[0].set(xlim=(0, 100), ylim=(-.55, 2.55))
    axes[0].set_yticks([2, 1, 0], ["Both correct", "False still", "Still recall"])
    axes[0].set_xticks([0, 25, 50, 75, 100])
    axes[0].set_xlabel("Share of relevant cases (%)", fontsize=9)
    axes[0].grid(axis="x")
    fig.text(.25, .93, "●  0.05 deployed", color=BLUE, fontsize=9)
    fig.text(.49, .93, "▲  0.02 candidate", color=ORANGE, fontsize=9)
    for y, label, key, c in [(1, "Pair wins", "wins", TEAL), (0, "Pair losses", "losses", ORANGE)]:
        count = len(transitions[key])
        axes[1].barh(y, count, height=.52, color=c)
        axes[1].text(count + .2, y, str(count), va="center", color=INK, fontsize=10)
    axes[1].set_yticks([1, 0], ["Wins", "Losses"])
    axes[1].set_xticks([0, 3, 6])
    axes[1].set(xlim=(0, 7), ylim=(-.6, 1.65))
    axes[1].set_xlabel("Changes on 48 pairs", fontsize=9)
    axes[1].grid(axis="x")
    save(fig, "threshold_tradeoff")


if __name__ == "__main__":
    mmsi_headlines()
    mmsi_categories()
    frame_reader()
    motion_bands()
    displacements()
    threshold_tradeoff()
