#!/usr/bin/env python3
"""Static file server for the project WITH HTTP Range support.

`python -m http.server` ignores the `Range` header and returns the whole file (200),
which means HTML5 <video> cannot SEEK — clicking the scrub bar or jumping to a segment
does nothing. This server answers Range requests with `206 Partial Content`, so video
seeking works. Use it (instead of `python -m http.server`) to run the segment browser:

    python3 tools/serve.py            # serves the project root on http://localhost:8000
    python3 tools/serve.py 8100       # ...on a different port
    # then open  http://localhost:8000/analysis/web/index.html

Pure standard library; no third-party dependencies.
"""
from __future__ import annotations

import functools
import os
import re
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class RangeHandler(SimpleHTTPRequestHandler):
    """SimpleHTTPRequestHandler + byte-range (206) support for seekable media."""

    def send_head(self):
        self._range = None
        path = self.translate_path(self.path)
        if os.path.isdir(path):
            return super().send_head()          # directory listing / index.html redirect
        try:
            f = open(path, "rb")
        except OSError:
            self.send_error(404, "File not found")
            return None
        try:
            fs = os.fstat(f.fileno())
            size = fs.st_size
            ctype = self.guess_type(path)
            rng = self.headers.get("Range")
            if not rng:
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(size))
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Last-Modified", self.date_time_string(fs.st_mtime))
                self.end_headers()
                return f
            m = re.match(r"bytes=(\d*)-(\d*)\s*$", rng)
            if not m:
                self.send_error(400, "Invalid Range header")
                f.close()
                return None
            g_start, g_end = m.group(1), m.group(2)
            if g_start == "":                     # suffix range: last N bytes
                n = int(g_end or 0)
                start, end = max(0, size - n), size - 1
            else:
                start = int(g_start)
                end = int(g_end) if g_end else size - 1
            end = min(end, size - 1)
            if start > end or start >= size:
                self.send_response(416)           # Range Not Satisfiable
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                f.close()
                return None
            self.send_response(206)
            self.send_header("Content-Type", ctype)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.send_header("Content-Length", str(end - start + 1))
            self.send_header("Last-Modified", self.date_time_string(fs.st_mtime))
            self.end_headers()
            f.seek(start)
            self._range = (start, end)
            return f
        except Exception:
            f.close()
            raise

    def copyfile(self, source, outputfile):
        rng = getattr(self, "_range", None)
        if rng is None:
            return super().copyfile(source, outputfile)
        start, end = rng
        remaining = end - start + 1
        while remaining > 0:
            chunk = source.read(min(64 * 1024, remaining))
            if not chunk:
                break
            try:
                outputfile.write(chunk)
            except (BrokenPipeError, ConnectionResetError):
                break                              # client seeked away / closed — fine
            remaining -= len(chunk)


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    handler = functools.partial(RangeHandler, directory=ROOT)
    with ThreadingHTTPServer(("", port), handler) as httpd:
        print(f"Serving {ROOT}\n  http://localhost:{port}/analysis/web/index.html\n"
              f"(Range requests enabled — video seeking works. Ctrl+C to stop.)")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped.")


if __name__ == "__main__":
    main()
