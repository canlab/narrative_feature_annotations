# Segment Search — web interface

Interactive browser tool to rank movie/story segments by any combination of
computational annotations and play the matching moment.

## Launch

Serve from the **project root** (so the app and the media files share an origin),
then open the page:

```bash
cd <project-root>
python3 -m http.server 8000
# then open:  http://localhost:8000/analysis/web/index.html
```

(Serving from the root is what lets the in-browser player load videos from
`/data/movies/...`; opening `index.html` directly via `file://` shows rankings but
cannot play clips.)

## Use

- Filter/scroll the feature list (grouped by class); tick features to search on.
- Toggle each picked feature **High** / **Low**.
- Optionally filter by source and choose how many results to show.
- Segments are ranked by the mean z-score of the selected features; click **▶ Play**
  to watch that segment (auto-seeks to its start and stops at its end).

## Rebuild the index

`segments.json` is generated from the annotated corpus. Regenerate after annotating
more stimuli (or to change the window length):

```bash
PYTHONPATH=src .venv/bin/python tools/build_search_index.py --seglen 5
```
