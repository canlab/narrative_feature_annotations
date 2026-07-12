#!/usr/bin/env python3
"""Assemble the documentation book source (book/src/) for mdBook.

All project documentation already lives as canonical Markdown under docs/ (+ README.md,
the analysis figures, and the interactive web app). This script MIRRORS those files into
book/src/ with their directory structure preserved — so every existing relative link
(inter-doc links and ../analysis/figures/*.png) keeps resolving with zero rewriting, and
mdBook converts .md links to .html automatically. It then writes the book's landing page
(introduction.md), the interactive-browser page (browser.md), and the table of contents
(SUMMARY.md). book/src/ is fully generated — regenerate it any time the docs change:

    python3 tools/build_book.py && mdbook build book

Pure standard library; no third-party dependencies.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "book" / "src"

# (source path relative to ROOT, destination relative to SRC). Directories are copied
# whole; the mirror keeps relative links (docs<->docs, docs->../analysis/figures) intact.
MIRROR = [
    ("README.md", "README.md"),
    ("docs", "docs"),                       # all .md + walkthrough.m
    ("analysis/figures", "analysis/figures"),
    ("analysis/web", "analysis/web"),        # index.html + segments.json + README (browser)
]

# The curated table of contents. Each entry: (indent, title, src-relative path or None).
# A None path renders as a non-linked section header. Paths point at the MIRRORED files.
TOC = [
    ("prefix", "Introduction", "introduction.md"),
    ("part", "Overview", None),
    (0, "Project overview", "project_overview.md"),
    (0, "Contents & user guide", "docs/CONTENTS.md"),
    ("part", "Phase 1 — Scoping Review", None),
    (0, "Scoping review — index", "docs/scoping_review/scoping_index.md"),
    (0, "Overview & executive summary", "docs/scoping_review/00_overview.md"),
    (0, "Semantic feature hierarchy", "docs/scoping_review/01_hierarchy.md"),
    (0, "Visual features", "docs/scoping_review/02_visual.md"),
    (0, "Audio features", "docs/scoping_review/03_audio.md"),
    (0, "Language features", "docs/scoping_review/04_language.md"),
    (0, "Social features", "docs/scoping_review/05_social.md"),
    (0, "Situation features", "docs/scoping_review/06_situation.md"),
    (0, "Affect features", "docs/scoping_review/07_affect.md"),
    (0, "Redundancy & parsimony", "docs/scoping_review/08_redundancy.md"),
    (0, "Recommendations — best-in-class set", "docs/scoping_review/09_recommendations.md"),
    ("part", "Phase 2 — Pipeline & Format", None),
    (0, "Annotation output format (spec)", "docs/design/ANNOTATION_FORMAT.md"),
    (0, "Core feature-set proposal (frozen)", "docs/design/PHASE2_CORE_SET_PROPOSAL.md"),
    (0, "Pipeline status — the extractors", "docs/design/PHASE2_STATUS.md"),
    (0, "Deployment feasibility", "docs/design/DEPLOYMENT_FEASIBILITY.md"),
    (0, "Implementation plan", "docs/design/IMPLEMENTATION_PLAN.md"),
    ("part", "Phase 3 — Corpus", None),
    (0, "Corpus annotation", "docs/design/PHASE3_CORPUS.md"),
    (0, "Adding movies & stories", "docs/ADDING_MOVIES.md"),
    (0, "External stimuli (HCP, CamCAN)", "docs/EXTERNAL_STIMULI.md"),
    ("part", "Phase 4 — Analysis & Dissemination", None),
    (0, "Analysis, tools & figures", "docs/design/PHASE4_ANALYSIS.md"),
    (0, "Review paper (draft)", "docs/REVIEW_PAPER.md"),
    (0, "Interactive segment browser", "browser.md"),
    (1, "Browser app README", "analysis/web/app_readme.md"),
    ("part", "Appendix", None),
    (0, "Deploying this site", "docs/DEPLOYING.md"),
]

# mdBook renders any `README.md`/`index.md` to `index.html` but rewrites inbound `.md`
# links to `<name>.html`, which mismatches — and a rendered README would also clobber the
# web app's own index.html. So rename these two mirrored files away from README and rewrite
# every inbound link to match. (old on-disk name -> new name, both relative to SRC.)
RENAME = {
    "README.md": "project_overview.md",     # else it maps to index.html, displacing the intro
    "docs/scoping_review/README.md": "docs/scoping_review/scoping_index.md",
    "analysis/web/README.md": "analysis/web/app_readme.md",
}
# Link-target substrings to rewrite across all mirrored markdown (path tail is enough;
# any ../ prefix is preserved because only the tail is replaced).
LINK_REWRITE = {
    "scoping_review/README.md": "scoping_review/scoping_index.md",
    "analysis/web/README.md": "analysis/web/app_readme.md",
}


def corpus_facts() -> dict:
    """Pull headline numbers from the generated derivatives so the intro stays current."""
    f = {"n_stimuli": 83, "total_minutes": 470.6, "n_channels": 95,
         "audiovisual": 53, "audio_only": 29, "text_only": 1}
    stats = ROOT / "analysis" / "corpus_stats.json"
    if stats.exists():
        s = json.loads(stats.read_text())
        f["n_stimuli"] = s.get("n_stimuli", f["n_stimuli"])
        f["total_minutes"] = s.get("total_minutes", f["total_minutes"])
        for k in ("audiovisual", "audio_only", "text_only"):
            f[k] = s.get("by_modality", {}).get(k, f[k])
    tmpl = ROOT / "schema" / "channel_template.json"
    if tmpl.exists():
        f["n_channels"] = json.loads(tmpl.read_text()).get("n_channels", f["n_channels"])
    return f


def write_introduction(f: dict) -> None:
    hours = f["total_minutes"] / 60
    (SRC / "introduction.md").write_text(f"""# Narrative Feature Extraction

*Computational annotation of movies and stories for cognitive science and
naturalistic neuroimaging.*

This site is the entry point for browsing every part of the project: the Phase 1
**scoping review**, the annotation **format specification** and **pipeline**, the
**how-to guides** for growing the corpus, the draft **review paper** with its figures,
and the interactive **segment browser**.

## What the project does

A human supplies a movie or an audio/text story; the pipeline returns a hierarchical,
semantically organized, **second-by-second** set of annotations — visual, audio,
language, social, situational, and affective — produced by a curated set of best-in-class
models. Features that do not apply to a stimulus (e.g. visual features for an audio-only
story) are returned as explicit nulls, so every annotation shares one **constant shape**
and the corpus stacks into rectangular matrices for analysis.

## At a glance

- **{f['n_stimuli']} stimuli** annotated (~{hours:.1f} h): {f['audiovisual']} audiovisual,
  {f['audio_only']} audio-only, {f['text_only']} text-only.
- **{f['n_channels']} channels** per stimulus across six feature classes, on a common 1 Hz grid.
- Local-first models on Apple-Silicon GPU/CPU; HDF5 + JSON output with a MATLAB reader.

## How this book is organized

| Part | Contents |
|------|----------|
| **Overview** | Project overview and the full [contents & user guide](docs/CONTENTS.md). |
| **Phase 1 — Scoping review** | Survey of computational annotation tools per feature class, the semantic hierarchy, redundancy analysis, and best-in-class recommendations. |
| **Phase 2 — Pipeline & format** | The annotation [format spec](docs/design/ANNOTATION_FORMAT.md), the frozen feature set, and the extractors that run. |
| **Phase 3 — Corpus** | How the corpus was assembled and [how to add your own movies/stories](docs/ADDING_MOVIES.md). |
| **Phase 4 — Analysis & dissemination** | Corpus analysis and tools, the [review paper](docs/REVIEW_PAPER.md), and the [interactive browser](browser.md). |

## Start here

- **[Review paper (draft)](docs/REVIEW_PAPER.md)** — the models/algorithms behind each
  annotation and the empirical structure of the annotation space across the corpus.
- **[Contents & user guide](docs/CONTENTS.md)** — the full map of tools, datasets,
  derivatives, and how to load / view / inspect them (with MATLAB recipes).
- **[Interactive segment browser](browser.md)** — rank stimulus segments by any
  combination of features.

> The source repository holds the pipeline (`src/nfe/`), the MATLAB reader (`matlab/`),
> and the annotated corpus derivatives. This documentation book is generated from the
> canonical Markdown in `docs/` by `tools/build_book.py`.
""")


def write_browser() -> None:
    readme = ROOT / "analysis" / "web" / "README.md"
    extra = ""
    if readme.exists():
        body = readme.read_text().split("\n", 1)[1].strip()   # drop the H1
        extra = "\n## Notes from the app README\n\n" + body + "\n"
    (SRC / "browser.md").write_text(f"""# Interactive segment browser

A static, in-browser tool that ranks every stimulus segment by any combination of
annotation features and plays the matching moment. It reads a precomputed index
(`analysis/web/segments.json`) and runs entirely client-side — no server logic.

**▶ [Open the segment browser](analysis/web/index.html)**

Pick any set of features, toggle each **High** or **Low**, and segments are ranked by
their mean z-score across the chosen features (with a features-covered count when a
segment lacks some channels). Interpretable vector channels are expanded per component,
so — for example — EmoNet's 20 emotion categories and the eight facial expressions are
individually searchable. Rankings are face-valid: high optical flow surfaces action
scenes, high word-rate the dialogue clips, EmoNet "Aesthetic Appreciation" the
beach-sunset clips.

> **Playback & media.** The ranking interface works anywhere. Actually *playing* a clip
> needs the source media served from the same origin — so playback works when you serve
> the project locally, but not on the public GitHub Pages site, where the licensed media
> are not hosted. To use it fully, serve the repository root with `python3 tools/serve.py`
> (a small Range-capable server, so video **seeking** works — the built-in
> `python -m http.server` cannot seek) and open `analysis/web/index.html`.
{extra}""")


def write_summary() -> None:
    lines = ["# Summary", ""]
    for indent, title, path in TOC:
        if indent == "part":
            lines += ["", f"# {title}", ""]
        elif indent == "prefix":                       # prefix chapter: plain link, becomes index.html
            lines.append(f"[{title}]({path})")
        elif path is None:
            lines.append(f"- {title}")
        else:
            lines.append(f"{'  ' * indent}- [{title}]({path})")
    (SRC / "SUMMARY.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    if SRC.exists():
        shutil.rmtree(SRC)
    SRC.mkdir(parents=True)
    for rel_src, rel_dst in MIRROR:
        s, d = ROOT / rel_src, SRC / rel_dst
        if not s.exists():
            print(f"  WARN missing source: {rel_src}")
            continue
        d.parent.mkdir(parents=True, exist_ok=True)
        if s.is_dir():
            shutil.copytree(s, d, dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns(".DS_Store", "__pycache__"))
        else:
            shutil.copy2(s, d)
    # Rename README/index-named chapters so mdBook doesn't map them to index.html.
    for old, new in RENAME.items():
        op, np_ = SRC / old, SRC / new
        if op.exists():
            op.rename(np_)
    # Rewrite inbound links to the renamed files across every mirrored markdown page.
    for md in SRC.rglob("*.md"):
        text = md.read_text()
        new_text = text
        for a, b in LINK_REWRITE.items():
            new_text = new_text.replace(a, b)
        if new_text != text:
            md.write_text(new_text)

    facts = corpus_facts()
    write_introduction(facts)
    write_browser()
    write_summary()
    n_md = sum(1 for _ in SRC.rglob("*.md"))
    print(f"wrote {SRC} ({n_md} markdown pages; corpus: {facts['n_stimuli']} stimuli, "
          f"{facts['n_channels']} channels)")


if __name__ == "__main__":
    main()
