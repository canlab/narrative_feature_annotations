# Narratives dataset — spoken-story audio stimuli

Audio story stimuli from the **"Narratives" fMRI dataset** (Nastase et al., 2021),
OpenNeuro **ds002345**. 29 naturalistic spoken stories (~5.3 h total, ~3–56 min each)
downloaded from the dataset's `stimuli/` folder on OpenNeuro's public S3 bucket. In this
project they are treated as **audio-only** stimuli (ASR transcribes them; language / audio /
affect / event passes run; visual channels are `NaN`).

## Attribution & citation
Nastase, S. A., Liu, Y.-F., Hillman, H., Zadbood, A., Hasenfratz, L., Keshavarzian, N.,
Chen, J., Honey, C. J., … Hasson, U. (2021). *The "Narratives" fMRI dataset for evaluating
models of naturalistic language comprehension.* **Scientific Data** 8:250. OpenNeuro ds002345.

## License / terms — IMPORTANT
- The **dataset** is released as **CC0** on OpenNeuro.
- The **audio stimulus clips themselves are NOT public domain.** Per the dataset's
  `stimuli/README`, they were compiled from various sources and are shared **strictly for
  non-profit, non-commercial scholarly research under "fair use" / "fair dealing."** Several
  clips are copyrighted TV/film audio (e.g., *bigbang*, *friends*, *himym*, *seinfeld*,
  *vinny*, *upintheair*, *sherlock*, *merlin*). Use here (computational annotation for
  cognitive-neuroscience research) is within that intended scope. **Do not redistribute** the
  media. Rights holders may request removal via sam.nastase@gmail.com.

## Contents (as downloaded)
- **Natural spoken narratives** (The Moth–style and lab-recorded): `pieman`, `piemanpni`,
  `tunnel`, `lucy`, `prettymouth`, `milkywayoriginal`, `slumlordreach`, `notthefall` /
  `notthefallintact`, `21styear`, `forgot`, `bronx`, `black`, `shame`, `santa`,
  `shapesphysical`, `shapessocial`, `sherlock`, `merlin`.
- **Short TV/film audio clips** (copyrighted): `bigbang`, `friends`, `himym`, `seinfeld`,
  `vinny`, `upintheair`.
- **Scrambled / paraphrased control variants** (degraded or manipulated speech — you may want
  to exclude these from naturalistic analyses): `milkywaysynonyms`, `milkywayvodka`,
  `notthefalllongscram`, `notthefallshortscram`.

Time-stamped word/phoneme transcripts and Gentle forced alignments are also available in
ds002345 (not downloaded here — the pipeline's ASR re-transcribes; grab them from OpenNeuro
if you prefer ground-truth transcripts).

Source: <https://openneuro.org/datasets/ds002345> · paper: <https://doi.org/10.1038/s41597-021-01033-3>
