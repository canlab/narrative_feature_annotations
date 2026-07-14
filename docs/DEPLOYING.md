# Deploying the documentation site (GitHub Pages)

The documentation book is published to GitHub Pages from this repository at
**<https://canlab.github.io/narrative_feature_annotations/>**. A GitHub Actions workflow
(`.github/workflows/deploy-book.yml`) rebuilds and republishes it on every push to `main`.

## What lives in git vs. Dropbox

This project mixes small, shareable text with large, copyrighted media. The split is:

| Kept in **git / GitHub** | Kept in **Dropbox only** (git-ignored) |
|--------------------------|----------------------------------------|
| Pipeline code (`src/`, `matlab/`, `tools/`) | Stimulus **media** — the movies and audio (`data/**/*.mp4,*.wav,…`, ~11 GB) |
| Docs (`docs/`, `README.md`) and the book config (`book/book.toml`) | Annotation **derivatives** (`annotations/`, ~217 MB of `.h5` + sidecars) |
| Schema (`schema/`), figures (`analysis/figures/`), search index (`analysis/web/`) | |
| Corpus **metadata** (`data/manifest.csv`, `data/**/SOURCES.md`, lexicon CSVs, text story) | Reference **PDFs** (`PDFs/`, ~100 MB) — publisher/preprint PDFs of the papers behind each model |

Why the media, annotations, and reference PDFs stay out of git:

- **Media** exceed GitHub's 100 MB per-file limit and are **copyrighted** (the `spacetop`
  clips, the *Narratives* audio, and *Kung Fury* are third-party IP). Only the CC-BY
  Blender open films are freely redistributable.
- **Annotations** are large derivatives (and their ASR transcripts reproduce film
  dialogue). They are regenerable from the media, so they are kept in Dropbox and can be
  published separately (e.g. OSF/Zenodo with a DOI) if a citable dataset is wanted.
- **Reference PDFs** (`PDFs/`) are copyrighted publisher/preprint copies kept for offline
  convenience; the docs link to the public arXiv/DOI URLs instead, so the site needs no PDFs.

The published site does **not** need either: it is built from the docs, the figures, and
the interactive search index, all of which are committed.

## One-time setup

```bash
cd "<this folder>"

# 1. Initialize git and keep Dropbox out of git's internals (see "Dropbox" below).
git init
xattr -w com.dropbox.ignored 1 .git      # macOS: stop Dropbox syncing the .git folder

# 2. Stage everything; .gitignore already excludes media, annotations, .venv, book build.
git add -A
git status                                 # sanity check: no .mp4/.wav/.h5 should appear

# 3. First commit.
git commit -m "Narrative Feature Extraction: pipeline, docs, and documentation book"

# 4. Create the GitHub repo and push (org: canlab).
git branch -M main
gh repo create canlab/narrative_feature_annotations --source=. --remote=origin --push --public
#   …or, if the repo already exists:
#   git remote add origin https://github.com/canlab/narrative_feature_annotations.git
#   git push -u origin main
```

Then enable Pages **once**: repo **Settings → Pages → Build and deployment → Source:
"GitHub Actions"**. The next push (or a manual run of the *Deploy documentation book*
workflow) publishes the site. GitHub Pages requires a **public** repo on the free plan.

## Updating the site

Edit the Markdown under `docs/` (or `README.md`), commit, and push — the workflow
regenerates the book and redeploys. To preview locally first:

```bash
brew install mdbook                  # one-time; or: cargo install mdbook
python3 tools/build_book.py          # assemble book/src/ from docs/
mdbook serve book                    # live preview at http://localhost:3000
```

`book/src/` and `book/book/` are generated (git-ignored); never edit them by hand.

## Regenerating the annotations (not in the repo)

Because `annotations/` is Dropbox-only, a fresh `git clone` has the code and docs but not
the derivatives. To rebuild them you need the media (also Dropbox-only) in place, then:

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
PYTHONPATH=src .venv/bin/python -m nfe.batch --manifest data/manifest.csv \
    --out annotations/corpus --template schema/channel_template.json \
    --vision --audio-hl --events
# then refresh derivatives the site uses:
PYTHONPATH=src .venv/bin/python tools/build_search_index.py    # analysis/web/segments.json
# (MATLAB) refreshAnalysis("annotations/corpus")               # analysis/figures + stats
```

See [`ADDING_MOVIES.md`](ADDING_MOVIES.md) for the full refresh workflow.

## Dropbox + git

This working folder lives in Dropbox. That is fine — GitHub Pages does not care where the
local folder is — with one caveat: **Dropbox continuously syncs the `.git` folder**, which
across multiple machines (or if it syncs mid-commit) can create "conflicted copy" files
inside `.git` and corrupt the repo. Mitigations:

- Tell Dropbox to ignore `.git` (the `xattr` command above), so Dropbox leaves git alone.
- The arrangement is otherwise complementary: **Dropbox** holds the large media +
  annotations; **GitHub** holds the code + docs. Don't sync the *repo* across machines via
  Dropbox — use `git clone`/`pull`/`push` for that and let Dropbox carry only the media.
