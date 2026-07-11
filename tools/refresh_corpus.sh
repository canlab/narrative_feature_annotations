#!/usr/bin/env bash
# Re-run the corpus pipeline after adding media to data/movies/.
# Regenerates the manifest, annotates any NEW stimuli (resumable — existing .h5 are
# skipped), and rebuilds the web search index. Then run the MATLAB analysis step
# (refreshAnalysis) and the doc update — see docs/ADDING_MOVIES.md.
#
# Usage:
#   tools/refresh_corpus.sh                 # default passes (vision + audio-hl + events)
#   tools/refresh_corpus.sh --reason        # extra flags are forwarded to nfe.batch
#   tools/refresh_corpus.sh --no-skip       # re-annotate everything (e.g. after model changes)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
export PYTHONPATH=src

echo "==> 1/3 rebuild manifest"
$PY tools/build_manifest.py

echo "==> 2/3 annotate corpus (new stimuli only unless --no-skip)"
$PY -m nfe.batch --manifest data/manifest.csv --out annotations/corpus \
    --template schema/channel_template.json --vision --audio-hl --events "$@"

echo "==> 3/3 rebuild web search index"
$PY tools/build_search_index.py --seglen 5

cat <<'EOF'

Python refresh complete. Next:
  1) In MATLAB:   addpath matlab; refreshAnalysis("annotations/corpus")
     (regenerates analysis/figures/* and analysis/corpus_stats.json)
  2) Ask Claude to update the docs from analysis/corpus_stats.json, OR follow the
     "Agent runbook" in docs/ADDING_MOVIES.md.
EOF
