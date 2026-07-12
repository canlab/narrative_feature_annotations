# Segment Search — web interface

Interactive browser tool to rank movie/story segments by any combination of
computational annotations and play the matching moment.

## Launch

Serve from the **project root** with the bundled server, then open the page:

```bash
cd <project-root>
python3 tools/serve.py            # http://localhost:8000  (pass a port to change it)
# then open:  http://localhost:8000/analysis/web/index.html
```

Use `tools/serve.py`, **not** `python -m http.server`: the built-in server ignores HTTP
`Range` requests, so `<video>` cannot **seek** — the scrub bar and "play segment" won't
jump. `tools/serve.py` adds byte-range support so seeking works. Serving from the root is
also what lets the player load videos from `/data/movies/...`; opening `index.html` via
`file://` shows rankings but cannot play clips.

## Use

- Filter/scroll the feature list (grouped by class); tick features to search on.
- Toggle each picked feature **High** / **Low**.
- Optionally filter by source and choose how many results to show.
- Segments are ranked by the mean z-score of the selected features. Each result has
  **▶ Segment** (seeks to its start, auto-stops at its end) and **▶ Clip** (plays the
  whole stimulus); the player repeats these for the loaded segment.
- Selected features are plotted as a time series over the whole clip, with a marker that
  tracks playback. **Click the plot** to jump the video to that time; the video scrub bar
  works too.

## Rebuild the index

`segments.json` is generated from the annotated corpus. Regenerate after annotating
more stimuli (or to change the window length):

```bash
PYTHONPATH=src .venv/bin/python tools/build_search_index.py --seglen 5
```
