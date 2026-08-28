"""Localhost competition server for the pure pursuit showcase.

Deliberately built on the Python standard library only: nothing to install at
the venue, no network access required to start it. Run it with

    python src/server.py

then open http://127.0.0.1:8000 in a browser.
"""

import argparse
import json
import mimetypes
import os
import sys
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from current import CAR_WIDTH
from obstacles import scatter_course
from safe_expr import EXAMPLES, ExpressionError
from scoreboard import Scoreboard, clean_name
from simulation import PARAM_LIMITS, build_track, clamp_params, simulate

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEB_DIR = os.path.join(ROOT, "web")
MAX_BODY_BYTES = 64 * 1024

COURSE = scatter_course()
SCOREBOARD = Scoreboard()


class Handler(BaseHTTPRequestHandler):
    server_version = "PurePursuitShowcase/1.0"

    # -- helpers ---------------------------------------------------------

    def _send(self, status, body, content_type="application/json; charset=utf-8"):
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        # The page is single-origin and self-contained; keep it uncached so a
        # mid-fair edit to the UI shows up on refresh.
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _send_json(self, status, payload):
        self._send(status, json.dumps(payload))

    def _error(self, status, message):
        self._send_json(status, {"ok": False, "error": message})

    def _read_json(self):
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            raise ValueError("Invalid Content-Length header.")
        if length <= 0:
            return {}
        if length > MAX_BODY_BYTES:
            raise ValueError("Request body is too large.")
        raw = self.rfile.read(length)
        try:
            data = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise ValueError("Request body was not valid JSON.")
        if not isinstance(data, dict):
            raise ValueError("Request body must be a JSON object.")
        return data

    def _serve_static(self, path):
        relative = path.lstrip("/") or "index.html"
        target = os.path.realpath(os.path.join(WEB_DIR, relative))
        # Never serve outside web/, whatever the URL claims.
        if not target.startswith(os.path.realpath(WEB_DIR) + os.sep):
            return self._error(403, "Forbidden")
        if not os.path.isfile(target):
            return self._error(404, "Not found")
        content_type, _ = mimetypes.guess_type(target)
        with open(target, "rb") as handle:
            body = handle.read()
        self._send(200, body, content_type or "application/octet-stream")

    # -- routes ----------------------------------------------------------

    def do_GET(self):
        route = urlparse(self.path).path
        try:
            if route == "/api/health":
                return self._send_json(200, {"ok": True})
            if route == "/api/course":
                course = COURSE.to_dict()
                course["car_width"] = CAR_WIDTH
                course["car_length"] = 2.5
                return self._send_json(200, {
                    "ok": True,
                    "course": course,
                    "limits": {k: {"default": d, "min": lo, "max": hi}
                               for k, (d, lo, hi) in PARAM_LIMITS.items()},
                    "examples": [{"name": n, "equation": e} for n, e in EXAMPLES],
                })
            if route == "/api/scoreboard":
                return self._send_json(200, {
                    "ok": True,
                    "entries": SCOREBOARD.ranked(),
                    "stats": SCOREBOARD.stats(),
                })
            return self._serve_static(route)
        except Exception:
            traceback.print_exc()
            return self._error(500, "Server error.")

    do_HEAD = do_GET

    def do_POST(self):
        route = urlparse(self.path).path
        try:
            payload = self._read_json()
        except ValueError as exc:
            return self._error(400, str(exc))

        try:
            if route == "/api/preview":
                return self._handle_preview(payload)
            if route == "/api/run":
                return self._handle_run(payload)
            return self._error(404, "Not found")
        except ExpressionError as exc:
            return self._error(400, str(exc))
        except Exception:
            traceback.print_exc()
            return self._error(500, "Server error.")

    def _handle_preview(self, payload):
        equation = payload.get("equation", COURSE.default_equation)
        track = build_track(COURSE, equation)
        return self._send_json(200, {"ok": True, "track": track.to_lists()})

    def _handle_run(self, payload):
        try:
            name = clean_name(payload.get("name", ""))
        except ValueError as exc:
            return self._error(400, str(exc))

        equation = payload.get("equation", COURSE.default_equation)
        if not isinstance(equation, str):
            return self._error(400, "Equation must be text.")
        params = clamp_params(payload.get("params") or {})

        result, track, frames = simulate(COURSE, equation, params)
        entry = SCOREBOARD.record(name, result, equation, params)

        return self._send_json(200, {
            "ok": True,
            "result": result.to_dict(),
            "track": track.to_lists(),
            "frames": frames,
            "entry_id": entry["id"],
            "rank": SCOREBOARD.rank_of(entry["id"]),
            "entries": SCOREBOARD.ranked(),
            "stats": SCOREBOARD.stats(),
        })

    def log_message(self, fmt, *args):
        # Quieter than the default: one line per request, no client noise.
        sys.stderr.write("%s %s\n" % (self.command, self.path))


def main():
    parser = argparse.ArgumentParser(description="Pure pursuit competition server")
    parser.add_argument("--host", default="127.0.0.1",
                        help="bind address (default: 127.0.0.1, kiosk-only)")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    try:
        server = ThreadingHTTPServer((args.host, args.port), Handler)
    except OSError as exc:
        print(f"Could not bind {args.host}:{args.port} -- {exc}", file=sys.stderr)
        print(f"An old server is probably still running. Find and stop it with:\n"
              f"  lsof -ti tcp:{args.port} | xargs kill\n"
              f"or start this one on another port:  --port {args.port + 1}",
              file=sys.stderr)
        raise SystemExit(1)
    stats = SCOREBOARD.stats()
    # flush=True: stdout is block-buffered when redirected to a log file, and
    # the operator needs to see the URL immediately either way.
    print(f"Course: {COURSE.name}   scoreboard: {SCOREBOARD.json_path}", flush=True)
    print(f"Loaded {stats['attempts']} previous attempts "
          f"({stats['finishes']} finishes)", flush=True)
    print(f"\n  Open  http://{args.host}:{args.port}", flush=True)
    print(f"  Stop  Ctrl-C here, or from another terminal: "
          f"kill {os.getpid()}\n", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down. Scoreboard saved.")
        server.shutdown()


if __name__ == "__main__":
    main()
