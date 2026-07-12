#!/usr/bin/env python3
"""Generate an editable SVG map of the annotation feature hierarchy.

Reads schema/channel_template.json (the constant-shape channel set) and lays out every
channel as a taxonomy poster: six feature-class cards, each grouped by subclass, with one
chip per channel. The output is a plain, flat SVG (presentation attributes, real <text>,
grouped per card, no CSS/filters) so it imports cleanly into PowerPoint / Illustrator /
Inkscape and can be recolored or rearranged. Regenerate after the channel set changes:

    python3 tools/build_feature_map.py            # -> analysis/figures/feature_map.svg

Pure standard library.
"""
from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "analysis" / "figures" / "feature_map.svg"

# Class order + a modern, distinct hue per class, and a one-line description.
CLASSES = OrderedDict([
    ("visual",    ("#6366f1", "What the frame looks like")),
    ("audio",     ("#06b6d4", "What the soundtrack sounds like")),
    ("language",  ("#f59e0b", "What is said (from the transcript)")),
    ("social",    ("#ec4899", "Who is present and how they relate")),
    ("situation", ("#10b981", "Where/when; event structure")),
    ("affect",    ("#ef4444", "Depicted emotion across channels")),
])
SUBLABEL = {  # prettier subclass names
    "low_level_static": "low-level static", "high_level_static": "high-level semantics",
    "faces_bodies_gaze": "faces / bodies / gaze", "dynamic_motion": "motion",
    "saliency_aesthetics_depth": "saliency / depth", "action": "action",
    "low_level": "low-level acoustics", "high_level": "audio events / scenes",
    "speech": "speech", "lexical": "lexical", "syntax": "syntax",
    "depicted": "depicted affect", "(direct)": "",
}

# ---- geometry ----
MARGIN, TOP = 42, 168
CARD_W, GAPX = 452, 26
NCOL = 3
PAD = 16
HEADER_H = 58
SUB_GAP, CHIP_H, CHIP_GAP, LINE_GAP = 12, 26, 8, 8
CHAR_W, CHIP_PAD = 6.7, 20
FONT = "'Segoe UI','Helvetica Neue',Arial,sans-serif"


def mix(hex_a, hex_b, t):
    a = [int(hex_a[i:i+2], 16) for i in (1, 3, 5)]
    b = [int(hex_b[i:i+2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(a[i]+(b[i]-a[i])*t):02x}" for i in range(3))


def load_tree():
    t = json.loads((ROOT / "schema" / "channel_template.json").read_text())
    tree = OrderedDict()
    for cls in CLASSES:
        tree[cls] = OrderedDict()
    for c in sorted(t["channels"], key=lambda c: c["path"]):
        p = c["path"].split("/")
        cls, sub, leaf = p[0], (p[1] if len(p) > 2 else "(direct)"), p[-1]
        tree.setdefault(cls, OrderedDict()).setdefault(sub, []).append(leaf)
    # order subclasses by size (desc) for tidy packing
    for cls in tree:
        tree[cls] = OrderedDict(sorted(tree[cls].items(), key=lambda kv: -len(kv[1])))
    return tree, t.get("n_channels", sum(len(l) for s in tree.values() for l in s.values()))


def chip_w(text):
    return max(46, round(len(text) * CHAR_W) + CHIP_PAD)


def top_round_rect(w, h, r):
    return f"M0,{h} V{r} Q0,0 {r},0 H{w-r} Q{w},0 {w},{r} V{h} Z"


def render_card(cls, subs):
    """Render one class card at local origin (0,0); return (height, svg)."""
    base, desc = CLASSES[cls]
    n = sum(len(v) for v in subs.values())
    inner_w = CARD_W - 2 * PAD
    body = []
    y = HEADER_H + 14
    for sub, leaves in subs.items():
        label = SUBLABEL.get(sub, sub.replace("_", " "))
        if label:
            body.append(f'<text x="{PAD}" y="{y+12}" font-family="{FONT}" font-size="11.5" '
                        f'font-weight="700" letter-spacing="0.6" fill="{mix(base,"#000000",0.15)}">'
                        f'{escape(label.upper())}</text>')
            y += SUB_GAP + 12
        cx, cy = PAD, y
        chip_fill, chip_txt = mix(base, "#ffffff", 0.86), mix(base, "#000000", 0.45)
        for leaf in leaves:
            txt = leaf.replace("_", " ")
            w = chip_w(txt)
            if cx + w > PAD + inner_w and cx > PAD:      # wrap
                cx = PAD; cy += CHIP_H + LINE_GAP
            body.append(
                f'<g><rect x="{cx}" y="{cy}" width="{w}" height="{CHIP_H}" rx="7" '
                f'fill="{chip_fill}" stroke="{mix(base,"#ffffff",0.6)}" stroke-width="1"/>'
                f'<text x="{cx+w/2:.1f}" y="{cy+CHIP_H/2+4:.1f}" text-anchor="middle" '
                f'font-family="{FONT}" font-size="12.5" fill="{chip_txt}">{escape(txt)}</text></g>')
            cx += w + CHIP_GAP
        y = cy + CHIP_H + SUB_GAP + 4
    h = y + 10
    svg = [f'<rect x="0" y="0" width="{CARD_W}" height="{h}" rx="16" fill="#ffffff" '
           f'stroke="#e5e7eb" stroke-width="1"/>',
           f'<path d="{top_round_rect(CARD_W, HEADER_H, 16)}" fill="{base}"/>',
           f'<text x="{PAD}" y="26" font-family="{FONT}" font-size="18" font-weight="700" '
           f'fill="#ffffff">{escape(cls.upper())}</text>',
           f'<text x="{CARD_W-PAD}" y="26" text-anchor="end" font-family="{FONT}" '
           f'font-size="13" font-weight="600" fill="#ffffff" opacity="0.92">{n} channels</text>',
           f'<text x="{PAD}" y="44" font-family="{FONT}" font-size="12.5" fill="#ffffff" '
           f'opacity="0.9">{escape(desc)}</text>']
    return h, "".join(svg + body)


def main():
    tree, total = load_tree()
    cards = [(cls, *render_card(cls, subs)) for cls, subs in tree.items() if subs]
    cards.sort(key=lambda c: -c[1])                      # tallest first for balanced packing
    col_h = [TOP] * NCOL
    placed = []
    for cls, h, svg in cards:
        i = min(range(NCOL), key=lambda k: col_h[k])     # shortest column
        x = MARGIN + i * (CARD_W + GAPX)
        placed.append((x, col_h[i], svg))
        col_h[i] += h + GAPX
    W = MARGIN * 2 + NCOL * CARD_W + (NCOL - 1) * GAPX
    H = max(col_h) + 30

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
             f'width="{W}" height="{H}" font-family="{FONT}">',
             f'<rect x="0" y="0" width="{W}" height="{H}" fill="#f8fafc"/>',
             f'<text x="{MARGIN}" y="66" font-family="{FONT}" font-size="30" font-weight="800" '
             f'fill="#0f172a">Narrative Feature Annotation Map</text>',
             f'<text x="{MARGIN}" y="98" font-family="{FONT}" font-size="15.5" fill="#475569">'
             f'{total} computational annotation channels across six feature classes, on a shared '
             f'1&#8202;Hz time grid.</text>',
             f'<text x="{MARGIN}" y="122" font-family="{FONT}" font-size="12.5" fill="#94a3b8">'
             f'Each chip is one channel (a value series + provenance). Channels that do not apply '
             f'to a stimulus are stored as explicit nulls. Generated from schema/channel_template.json.</text>']
    for x, y, svg in placed:
        parts.append(f'<g transform="translate({x},{y})">{svg}</g>')
    parts.append("</svg>")
    OUT.write_text("\n".join(parts))
    print(f"wrote {OUT}  ({total} channels, {W}x{H})")


if __name__ == "__main__":
    main()
