"""Shared matplotlib setup so all figures use one style and category order."""

import os

from common import CACHE, CATEGORY_ORDER

os.environ.setdefault("MPLCONFIGDIR", str(CACHE / "mpl"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

CATEGORY_PALETTE = dict(zip(CATEGORY_ORDER, sns.color_palette("tab20", len(CATEGORY_ORDER))))


def setup():
    sns.set_theme(style="white", context="paper", font_scale=0.9)
    plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
                         "axes.titlesize": 9, "axes.labelsize": 8})
    return plt, sns


def expect(name: str, ok: bool, detail: str = "") -> bool:
    """Report a pre-stated expectation; unlike a check, a miss is a finding, not a bug."""
    print(f"[{'EXPECTED' if ok else 'SURPRISE'}] {name}" + (f" -- {detail}" if detail else ""))
    return ok
