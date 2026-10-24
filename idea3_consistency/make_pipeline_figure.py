"""Overview figure for the Idea 3 report section: midterm baseline -> failure analysis -> final plan.
Usage: python make_pipeline_figure.py   # writes fig_idea3_pipeline.pdf / .png next to this file
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

BLUE, ORANGE, GREEN, GREY = "#4C78A8", "#F58518", "#54A24B", "#555555"
plt.rcParams.update({"font.size": 7, "font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans"})

fig, ax = plt.subplots(figsize=(7.0, 3.3))
ax.set_xlim(0, 100); ax.set_ylim(0, 48); ax.axis("off")


def box(x, y, w, h, text, color, fill=True, size=7, weight="normal", align="center"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=0.8",
                                fc=color + ("22" if fill else "00"), ec=color, lw=0.9))
    tx = x + w / 2 if align == "center" else x + 1
    ax.text(tx, y + h / 2, text, ha=align, va="center", fontsize=size, weight=weight,
            color="#222222", linespacing=1.35)


def arrow(x0, y0, x1, y1, color=GREY):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=7,
                                 lw=0.9, color=color, shrinkA=0, shrinkB=0))


def panel(x, w, title, color):
    ax.add_patch(FancyBboxPatch((x, 7), w, 39, boxstyle="round,pad=0.3,rounding_size=1.2",
                                fc="none", ec=color, lw=1.2, ls="--"))
    ax.text(x + w / 2, 44.2, title, ha="center", va="center", fontsize=6.4, weight="bold", color=color)


# Panel 1: midterm baseline
panel(1, 31, "Midterm baseline", BLUE)
box(3, 37, 27, 4, "Input $x$ = (images, question, options)", BLUE, size=5.6)
variants = ["orig", "mirror", "reorder", "shuffle"]
for i, v in enumerate(variants):
    box(3 + i * 7, 27.5, 6, 6.5, v, BLUE, fill=False, size=5.4)
    arrow(16.5, 37, 6 + i * 7, 34.2)
    arrow(6 + i * 7, 27.5, 16.5, 24.4)
box(3, 20, 27, 4.2, "VLM $f$: Qwen2.5-VL-7B (+ InternVL3-8B)", BLUE, size=5.1)
arrow(10, 20, 10, 17.4); arrow(23, 20, 23, 17.4)
box(3, 9, 13, 8.2, "Transformation\nvote\n30.1% (SC 28.7%)", BLUE, size=5.3)
box(17, 9, 13, 8.2, "Disagreement\nscore $s(x)$\nAUROC 0.556", BLUE, size=5.3)

# Panel 2: failure analysis
panel(35, 29.5, "Failure analysis (test, N = 700)", ORANGE)
findings = [
    ("Lexical anchoring", "after mirroring, changed answers pick\nthe same-word option 56% (chance 33%)"),
    ("Confident errors", "38.5% of errors are consistent across\nall transforms, so voting cannot fix them"),
    ("Modality ablation", "no image: 26.3% vs. 29.0% with images\n(p = 0.18); 51% of answers unchanged"),
]
for i, (head, body) in enumerate(findings):
    y = 33 - i * 11.5
    box(37, y, 25, 9, "", ORANGE, size=6)
    ax.text(49.5, y + 6.6, head, ha="center", va="center", fontsize=6.2, weight="bold")
    ax.text(49.5, y + 3.0, body, ha="center", va="center", fontsize=4.6, linespacing=1.3)

# Panel 3: final plan
panel(67, 32, "Final: improvements", GREEN)
box(69, 27, 28, 14.5, "", GREEN)
ax.text(83, 39.0, "A. Inference-time decoding (main)", ha="center", fontsize=5.5, weight="bold")
ax.text(83, 35.6, "no training, same Qwen2.5-VL-7B", ha="center", fontsize=4.8, style="italic")
ax.text(83, 32.2, r"TTA: $\Sigma_t\, \log p(\pi_t(c)\,|\,t(x))$", ha="center", fontsize=5.4)
ax.text(83, 28.9, r"$-\ \alpha\, \Sigma_t \log p(\pi_t(c)\,|\,t(x)_{\varnothing})$  (no image)",
        ha="center", fontsize=5.0)
box(69, 10, 28, 14, "", GREEN, fill=False)
ax.text(83, 21.6, "B. Extension: consistency fine-tuning", ha="center", fontsize=5.1, weight="bold")
ax.text(83, 18.4, r"$\mathcal{L} = \mathrm{CE}$", ha="center", fontsize=5.4)
ax.text(83, 15.3, r"$+\ \lambda_1 \Sigma_t\, \mathrm{KL}(p(\cdot|x)\,\|\,\pi_t\, p(\cdot|t(x)))$",
        ha="center", fontsize=5.4)
ax.text(83, 12.2, r"$+\ \lambda_2\, \mathrm{KL}(p(\cdot|x_{\varnothing})\,\|\,\mathcal{U})$",
        ha="center", fontsize=5.4)

arrow(32.6, 26, 34.4, 26, GREY)
arrow(64.6, 26, 66.4, 26, GREY)
ax.text(50, 2.6, "Evaluate baseline, A and B on the same held-out split: accuracy, flip rate, "
        "lexical anchoring, confident errors, no-image agreement", ha="center", va="center", fontsize=5.8,
        style="italic", color=GREY)

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig_idea3_pipeline")
fig.savefig(out + ".pdf", bbox_inches="tight")
fig.savefig(out + ".png", dpi=250, bbox_inches="tight")
print(f"wrote {out}.pdf/.png")
