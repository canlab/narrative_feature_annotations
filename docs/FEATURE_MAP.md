# Feature map

A visual map of **all 95 annotation channels and their hierarchical organization** — the
six feature classes, each grouped by subclass, with one chip per channel.

![Narrative Feature Annotation Map](../analysis/figures/feature_map.svg)

## Editable graphic

The map is a plain, flat **SVG** built for editing:
[`analysis/figures/feature_map.svg`](../analysis/figures/feature_map.svg).

- Open it in **PowerPoint**, **Illustrator**, **Inkscape**, or any browser. In PowerPoint,
  *Insert → Pictures*, then right-click → *Convert to Shape* to recolor, move, or relabel
  individual cards, chips, and text.
- It uses real text elements, per-card groups, and inline colors (no CSS or filters), so
  it survives import cleanly and recolors easily.
- Regenerate it whenever the channel set changes:
  ```bash
  python3 tools/build_feature_map.py     # reads schema/channel_template.json
  ```

## Related

- The **designed semantic hierarchy** behind these channels (the taxonomy the pipeline is
  organized around): [feature hierarchy](scoping_review/01_hierarchy.md).
- What each pass actually computes: [pipeline status](design/PHASE2_STATUS.md).
- The on-disk layout of these channels in each `.h5`: [annotation format](design/ANNOTATION_FORMAT.md).
