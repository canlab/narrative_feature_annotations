# Reusable prompt — "summarize the project now"

When the user says **"summarize the project now"** (or asks for a project summary), produce a
short narrative review of the whole project and its current annotation state, and write it —
together with the feature table and 1–3 figures — into **`docs/feature_summary_table.docx`**
(Word format, for import into Google Docs).

## Audience & tone
Expert scientists who are **not** necessarily AI/ML experts. Convey **what** was done, the
**current state** of the annotations, and — importantly — **why** it matters (the scientific
motivation), clearly and fairly. Be accurate; do not overstate. Narrative ≤ **half a page**.

## What the summary must contain (in this order, in the .docx)
1. **Narrative review** (two short paragraphs, ≤ ½ page):
   - *What & why:* naturalistic neuroimaging needs quantitative, second-by-second descriptions
     of dynamic movie/story stimuli; the scoping review of best-in-class visual/auditory/language
     models; the selection criteria (large training data, **open weights** for reproducibility,
     best-in-class validated performance); the selected extractors (interpretable classical-ML /
     signal-processing features **and** large foundation/transformer models, with concrete
     examples such as DINOv2, SigLIP2, CLAP, VideoMAE, EmoNet); the six broad domains (visual,
     auditory, language, social, emotional, situational) organized low-/mid-/high-level (Table 1).
   - *Application & structure:* how many feature-variable time series were extracted from how many
     clips/stories and how many seconds (1 Hz); the interactive viewer for QC; the within-domain
     exploratory factor analysis and what it showed — some model outputs are **irreducible**
     (first principal factor explains only a few %; e.g., action posteriors, DINOv2, SigLIP2,
     AudioSet, EmoNet), others **highly reducible** (first component > 50%; e.g., chroma, sentiment).
2. **Table 1** — the class × level feature-summary table.
3. **Figures** from `matlab/figures/` (from `plotFactorScores` and
   `plot_tsne_umap_annotations`), placed **after the table**, each with a brief caption. Pick the
   ones that show the annotations most clearly and beautifully (defaults: `01_factor_timeseries`,
   `04_correlation_matrix`, `05_tsne`, and `08_umap_annotations` — the full 2,768-variable
   annotation space colored by domain, with a legend).

## How to generate it (mechanics)
All headline numbers are pulled live so the prose stays correct as the corpus grows:
- `schema/channel_template.json` → channel and variable counts.
- `analysis/corpus_stats.json` → stimulus counts, modality split, total minutes.
- `analysis/factor_reducibility.json` → per-category first-PC % (written by the MATLAB factor
  analysis; regenerate if the corpus changed — see below).

Steps:
1. If the corpus or factors changed, refresh inputs first:
   - In MATLAB: `F = readAnnotationCorpusFull("annotations/corpus"); F = extractCategoryFactors(F, "Save","analysis/extracted_factors.mat");`
     then `plotFactorScores();` (writes the figures and, via the helper, the reducibility numbers).
     If `analysis/factor_reducibility.json` is stale, recompute the per-category first-PC % and
     rewrite it.
2. **Review and, if needed, rewrite the narrative** in `tools/build_project_summary.py` so it is
   accurate and fair for the current state (this is the judgment step — do not just run the script
   blindly; update the prose, examples, and figure choices as the project evolves).
3. Run: `python3 tools/build_project_summary.py` → writes `docs/feature_summary_table.docx`.
4. Verify (optional): convert to PDF and eyeball that the narrative is ≤ ½ page, the table fits
   one page, and the figures + captions render.

The table itself is built by `tools/build_feature_summary.py` (reused by the summary builder).
