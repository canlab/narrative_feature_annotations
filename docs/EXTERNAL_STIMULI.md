# Obtaining external neuroimaging stimuli (Narratives, HCP, CamCAN)

Three widely-used naturalistic-fMRI stimulus sets that enrich this corpus and enable
comparison with a large body of published brain data. The **Narratives** spoken-story
collection is openly available and has **already been downloaded and integrated**; the
**HCP 7T movie** clips and the **CamCAN** Hitchcock clip require data-access requests. This
page gives provenance/licensing for each and how to (re)obtain and integrate them.

| Stimulus | What it is | Openly downloadable? | Channel | Status |
|----------|-----------|----------------------|---------|--------|
| Narratives | 29 spoken-story audio clips (~5.3 h) | ✅ dataset CC0; stimuli shared for non-commercial research (fair use) | OpenNeuro ds002345 | **integrated** |
| HCP 7T movies | 4 concatenated `.mp4`s of CC Vimeo clips + Hollywood excerpts | ❌ (free account + data-use terms; Hollywood parts copyrighted) | ConnectomeDB | request |
| CamCAN movie | 8-min edit of Hitchcock's *Bang! You're Dead* | ❌ (application + agreement; copyrighted) | Cam-CAN data access | request |

---

## 1. Narratives spoken-story stimuli (obtained)

**What it is.** 29 naturalistic **spoken-story audio** clips (~3–56 min, ~5.3 h total) from
the "Narratives" fMRI dataset (Nastase et al., 2021), OpenNeuro **ds002345** — natural stories
(Pie Man, Tunnel, Slumlord, Milky Way, Sherlock, Merlin, …), several scrambled/paraphrased
control variants, and a few short TV/film audio clips. Integrated here as **audio-only**
stimuli under `data/stories/narratives/` (ASR transcribes them; audio/language/affect/event
passes run; visual channels are `NaN`).

**How it was obtained (reproducible).** Downloaded from OpenNeuro's public S3 bucket:
```bash
base="https://s3.amazonaws.com/openneuro.org/ds002345/stimuli"
for s in pieman tunnel slumlordreach ... ; do curl -fsSL -o "data/stories/narratives/${s}_audio.wav" "$base/${s}_audio.wav"; done
```
(The full file list is on S3 under `ds002345/stimuli/`; word/phoneme transcripts and Gentle
alignments are also in ds002345 if you want ground-truth transcripts instead of ASR.)

**License / terms.** The dataset is **CC0**, but the audio clips are **not public domain** —
they are shared **for non-profit, non-commercial scholarly research under "fair use"** (some
are copyrighted TV/film audio); do not redistribute. Attribution + details:
`data/stories/narratives/SOURCES.md`. Rights holders may request removal via the dataset's
contact.

**Status.** Downloaded and catalogued in the manifest; annotation runs via the standard
audio config (`--audio-hl --events --template`).

---

## 2. HCP 7T movie-watching stimuli

**What it is.** Four movie runs (`MOVIE1–MOVIE4`), each a `~12–14 min` `.mp4` concatenating
short (1–4.3 min) clips from two sources:
- **Creative-Commons Vimeo** independent films (filenames marked `CC`) — genuinely open,
- **Hollywood** film excerpts (the set published by *Cutting et al.*) — **copyrighted**,
plus a repeated Vimeo "validation" clip shown across runs. Metadata files
`HCP_7T_Movie_Info.csv` (clip origins) and `HCP_7T_Movie_Clip_Timing.csv` (start/stop of
each clip and REST block) accompany the stimuli.

**How to obtain (official).**
1. Register for a free account at **ConnectomeDB** — <https://db.humanconnectome.org>.
2. Accept the **HCP Open Access Data Use Terms**.
3. Open the **WU-Minn HCP 1200 Subjects (S1200)** project page and find the
   **"7T Movie Resources"** — download the **7T movie stimulus zip** plus
   `HCP_7T_Movie_Info.csv` and `HCP_7T_Movie_Clip_Timing.csv`.
   Direct project link: <https://db.humanconnectome.org/data/projects/HCP_1200>.
4. **Open subset (optional):** if you only need the freely-licensed portions, the `CC`-named
   clips are CC Vimeo films; `HCP_7T_Movie_Info.csv` lists each clip's origin, so those can be
   fetched individually. (This will *not* reproduce the exact HCP stimulus, which also
   includes the copyrighted Hollywood excerpts and specific concatenation/timing.)

**Licensing.** The Hollywood excerpts are copyrighted; HCP shares them under its data-use
terms for research use — do not redistribute. The `CC` clips carry their Creative-Commons
terms (attribution).

**Integrate.** Put the `.mp4`s in `data/movies/hcp/` and refresh (below).

---

## 3. CamCAN movie-watching stimulus

**What it is.** An **8-minute edited version of Alfred Hitchcock's *"Bang! You're Dead"***
(*Alfred Hitchcock Presents*, S7E2, 1961), condensed from the ~25–30 min original while
preserving the plot. It is **copyrighted** (Universal / the Hitchcock estate). The specific
8-min CamCAN edit — not just the original episode — is what matches the published data and
its extensive existing annotations.

**How to obtain (official).**
1. Apply through the **Cam-CAN data access portal** —
   <https://camcan-archive.mrc-cbu.cam.ac.uk/dataaccess/>.
2. Provide a valid **academic affiliation** and a **specific hypothesis**, and request the
   **movie-watching** data/stimulus. Notes from Cam-CAN: they reject vague "all data"
   requests and student-submitted applications — a **supervisor/PI must apply**, selecting
   only the data types justified by the proposal.
3. On approval you receive **sftp** download instructions; the Hitchcock stimulus is included.

**Licensing.** Copyrighted TV footage provided for research under the Cam-CAN agreement — do
not redistribute. (The full original episode airs on some streaming services, e.g. Roku, but
that is not the 8-min Cam-CAN edit.)

**Integrate.** Put the clip in `data/movies/camcan/` and refresh (below).

---

## Why these weren't downloaded automatically

Both require a **personal account and a data-use agreement that you must accept**, and both
contain **copyrighted** Hollywood/Hitchcock footage. I have no such accounts, and pulling the
copyrighted content from unofficial sources would not be appropriate. Obtain them through the
official channels above; then integration is one step.

## After you obtain them

```
data/movies/hcp/       ← HCP 7T MOVIE1–4 .mp4 files
data/movies/camcan/    ← the 8-min Hitchcock clip
```

Then run the standard refresh (see [`ADDING_MOVIES.md`](ADDING_MOVIES.md)) — or just tell
Claude *"refresh the dataset"*:

```bash
tools/refresh_corpus.sh                                  # annotates the new stimuli
matlab -batch "addpath matlab; refreshAnalysis('annotations/corpus')"
```

The new stimuli are annotated on the same 103-channel constant-shape template and folded into
all analyses and the search interface automatically. If you also have the HCP clip-timing CSV,
tell me and I can align annotations to the per-clip boundaries.

---

**Sources.** HCP: [7T imaging protocol](https://www.humanconnectome.org/hcp-protocols-ya-7t-imaging),
[ConnectomeDB S1200](https://db.humanconnectome.org/data/projects/HCP_1200),
[HCP-users: movie stimuli](https://groups.google.com/a/humanconnectome.org/g/hcp-users/c/Sa-Z_gfeiLo),
[7T movie clip-info wiki](https://wiki.humanconnectome.org/docs/7T%20Movie%20watching%20task%20clip%20info%20timing%20versions%20shown%20per%20individual%20subject.html).
CamCAN: [data access portal](https://camcan-archive.mrc-cbu.cam.ac.uk/dataaccess/),
[Cam-CAN protocol (Shafto et al. 2014, BMC Neurology)](https://link.springer.com/article/10.1186/s12883-014-0204-1),
[Bang! You're Dead (Hitchcock wiki)](https://the.hitchcock.zone/wiki/Alfred_Hitchcock_Presents_-_Bang!_You're_Dead).
