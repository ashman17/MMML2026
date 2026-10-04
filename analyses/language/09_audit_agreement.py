"""Compare the 44-item audit sheet with the automatic tags.

Run, from this directory:
  conda run -n 11777-project python 09_audit_agreement.py

The sheet is an assisted labeling pass. Agreement here is a check of the
automatic tags against that sheet, not an independent inter-rater score.
"""

from __future__ import annotations

import re

import pandas as pd

from common import OUTPUTS

AUDIT = OUTPUTS.parent / "audit"
TYPE_MAP = {
    "object": "object",
    "room": "region_room",
    "functional area": "region_functional",
    "part": "part",
    "camera": "camera",
}
LEMMA = {
    "clothes": "clothe", "boxes": "box", "photos": "photo", "pictures": "picture",
    "chairs": "chair", "areas": "area", "walls": "wall", "doors": "door",
}
CLOSED = ("frame", "anchor", "mirror", "reorder", "spatial", "viewpoint", "comparison", "reasoning")


def _blank(value) -> str:
    if value is None or (isinstance(value, float) and value != value):
        return ""
    text = str(value).strip()
    if text.lower() in {"", "nan", "none"}:
        return ""
    if re.fullmatch(r"\d+\.0", text):
        return text[:-2]
    return text


def _yes(value) -> str:
    text = _blank(value).lower()
    if text in {"yes", "true", "1"}:
        return "yes"
    if text in {"no", "false", "0"}:
        return "no"
    return text


def _mention_heads(text: str) -> set[tuple[str, str]]:
    found = set()
    for piece in _blank(text).split(";"):
        match = re.match(r"\s*(.*?)\s*\(([^)]+)\)\s*$", piece.strip())
        if not match:
            continue
        words = re.findall(r"[a-z]+", match.group(1).lower())
        if not words:
            continue
        head = LEMMA.get(words[-1], words[-1])
        kind = TYPE_MAP.get(match.group(2).strip().lower(), match.group(2).strip().lower())
        found.add((head, kind))
    return found


def _bridge_lemmas(text: str, kinds: set[str]) -> set[str]:
    found = set()
    for piece in _blank(text).split(";"):
        if ":" not in piece:
            continue
        name, kind = piece.split(":", 1)
        if kind.strip().split()[0] not in kinds:
            continue
        words = re.findall(r"[a-z]+", name.lower())
        if words:
            found.add(LEMMA.get(words[-1], words[-1]))
    return found


def main() -> None:
    sheet = pd.read_csv(AUDIT / "response_sheet.csv")
    key = pd.read_csv(AUDIT / "key.csv")
    mentions = pd.read_csv(AUDIT / "key_mentions.csv")
    frame = sheet.merge(key, on=["audit_item", "id"], suffixes=("_sheet", "_key"))
    assert len(frame) == 44

    frame["anchor_sheet"] = frame["anchor_image_sheet"].map(_blank)
    frame["anchor_key"] = frame["anchor_image_key"].map(_blank)
    comparisons = {
        "frame": frame["frame_sheet"] == frame["frame_key"],
        "anchor": frame["anchor_sheet"] == frame["anchor_key"],
        "mirror": frame["mirror"] == frame["mirror_level"],
        "reorder": frame["reorder"] == frame["reorder_level"],
        "spatial": frame["added_spatial"].map(_yes) == frame["add_spatial"].map(_yes),
        "viewpoint": frame["added_viewpoint"].map(_yes) == frame["add_viewpoint"].map(_yes),
        "comparison": frame["added_comparison"].map(_yes) == frame["add_comparison"].map(_yes),
        "reasoning": frame["added_reasoning"].map(_yes) == frame["add_reasoning"].map(_yes),
    }

    rows = []
    for item, row in frame.set_index("audit_item").iterrows():
        human = _mention_heads(row["mentions"])
        auto_rows = mentions[mentions["audit_item"] == item]
        auto = {(str(r.head), str(r.type)) for r in auto_rows.itertuples()}
        human_heads = {head for head, _ in human}
        auto_heads = {head for head, _ in auto}
        human_bridge = _bridge_lemmas(row["bridge_notes"], {"landmark", "synonym"})
        auto_bridge = set(_blank(row["bridging_heads"]).split())
        rows.append({
            "audit_item": int(item),
            "id": int(row["id"]),
            "human_mention_heads": len(human_heads),
            "auto_mention_heads": len(auto_heads),
            "mention_head_overlap": len(human_heads & auto_heads),
            "human_bridge": len(human_bridge),
            "auto_bridge": len(auto_bridge),
            "bridge_overlap": len(human_bridge & auto_bridge),
        })
    detail = pd.DataFrame(rows)
    summary_rows = []
    print(f"Items: {len(frame)}")
    for name in CLOSED:
        matched = int(comparisons[name].sum())
        print(f"  {name:12} {matched}/44")
        summary_rows.append({"field": name, "agree": matched, "n": 44})
        misses = frame.loc[~comparisons[name], ["audit_item", "id"]]
        for miss in misses.itertuples(index=False):
            item = int(miss.audit_item)
            if name == "frame":
                left, right = frame.loc[frame.audit_item == item, "frame_sheet"].iloc[0], frame.loc[frame.audit_item == item, "frame_key"].iloc[0]
            elif name == "anchor":
                left, right = frame.loc[frame.audit_item == item, "anchor_sheet"].iloc[0], frame.loc[frame.audit_item == item, "anchor_key"].iloc[0]
            elif name == "mirror":
                left, right = frame.loc[frame.audit_item == item, "mirror"].iloc[0], frame.loc[frame.audit_item == item, "mirror_level"].iloc[0]
            elif name == "reorder":
                left, right = frame.loc[frame.audit_item == item, "reorder"].iloc[0], frame.loc[frame.audit_item == item, "reorder_level"].iloc[0]
            elif name == "spatial":
                left, right = _yes(frame.loc[frame.audit_item == item, "added_spatial"].iloc[0]), _yes(frame.loc[frame.audit_item == item, "add_spatial"].iloc[0])
            elif name == "viewpoint":
                left, right = _yes(frame.loc[frame.audit_item == item, "added_viewpoint"].iloc[0]), _yes(frame.loc[frame.audit_item == item, "add_viewpoint"].iloc[0])
            elif name == "comparison":
                left, right = _yes(frame.loc[frame.audit_item == item, "added_comparison"].iloc[0]), _yes(frame.loc[frame.audit_item == item, "add_comparison"].iloc[0])
            else:
                left, right = _yes(frame.loc[frame.audit_item == item, "added_reasoning"].iloc[0]), _yes(frame.loc[frame.audit_item == item, "add_reasoning"].iloc[0])
            print(f"    {item:02d} sheet {left or 'blank'} | auto {right or 'blank'}")

    human_mentions = int(detail["human_mention_heads"].sum())
    auto_mentions = int(detail["auto_mention_heads"].sum())
    mention_overlap = int(detail["mention_head_overlap"].sum())
    human_bridge = int(detail["human_bridge"].sum())
    auto_bridge = int(detail["auto_bridge"].sum())
    bridge_overlap = int(detail["bridge_overlap"].sum())
    print(f"\nMention heads: sheet {human_mentions}, auto {auto_mentions}, overlap {mention_overlap}")
    print(f"  sheet heads found by auto: {mention_overlap}/{human_mentions}")
    print(f"  auto heads also on the sheet: {mention_overlap}/{auto_mentions}")
    print(f"Bridge lemmas (sheet landmark or synonym vs auto heads):")
    print(f"  sheet {human_bridge}, auto {auto_bridge}, overlap {bridge_overlap}")
    print(f"  sheet lemmas found by auto: {bridge_overlap}/{human_bridge}")
    print(f"  auto lemmas also on the sheet: {bridge_overlap}/{auto_bridge}")

    pd.DataFrame(summary_rows).to_csv(OUTPUTS / "09_audit_agreement.csv", index=False)
    detail.to_csv(OUTPUTS / "09_audit_mention_bridge.csv", index=False)


if __name__ == "__main__":
    main()
