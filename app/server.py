"""
Audio Proc - 2000 :: HTTP server (standard library only).

    GET  /                  control panel
    GET  /hls/stream.m3u8   processed HLS output (plus segments)
    GET  /api/state         settings + meters + source status + log
    GET  /api/devices       detected sound cards and test signals
    POST /api/settings      full or partial settings object
    POST /api/source        {"type": "hls", "url": ...} | {"type": "soundcard", "device": ...}
                            | {"type": "test", "signal": "music|tone|pink"} | {"type": null} to stop
    POST /api/reset         restore default processing settings

Environment:
    PORT (8020)  DATA_DIR (./data)  HLS_DIR (/tmp/audioproc2000-hls)
    AUTH_USER / AUTH_PASS  optional basic auth on the control panel and API
                           (the /hls/ output stays open so players can pull it)
"""
from __future__ import annotations

import base64
import json
import mimetypes
import os
import signal
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from .engine import DEFAULT_SETTINGS, Engine
from .media import TEST_SIGNALS, Processor, list_sound_cards

HERE = os.path.dirname(os.path.abspath(__file__))
STATIC = os.path.join(HERE, "static")
PORT = int(os.environ.get("PORT", "8020"))
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(HERE, "..", "data"))
HLS_DIR = os.environ.get("HLS_DIR", "/tmp/audioproc2000-hls")
AUTH_USER, AUTH_PASS = os.environ.get("AUTH_USER"), os.environ.get("AUTH_PASS")
STATE_FILE = os.path.join(DATA_DIR, "state.json")

os.makedirs(DATA_DIR, exist_ok=True)


def load_state() -> dict:
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


saved = load_state()
engine = Engine(saved.get("settings"))
proc = Processor(engine, HLS_DIR)
save_lock = threading.Lock()


def save_state():
    with save_lock:
        tmp = STATE_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"settings": engine.get_settings(), "source": proc.spec}, f, indent=2)
        os.replace(tmp, STATE_FILE)


def deep_merge(base: dict, patch: dict) -> dict:
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def validate_source(body: dict) -> dict | None:
    kind = body.get("type")
    if not kind:
        return None
    if kind == "hls":
        url = str(body.get("url", "")).strip()
        if urlparse(url).scheme not in ("http", "https"):
            raise ValueError("HLS URL must start with http:// or https://")
        return {"type": "hls", "url": url}
    if kind == "soundcard":
        devs = {d["id"]: d["name"] for d in list_sound_cards()}
        dev = body.get("device")
        if dev not in devs:
            raise ValueError("Sound card not found — refresh the device list")
        return {"type": "soundcard", "device": dev, "name": devs[dev]}
    if kind == "test":
        sig = body.get("signal", "music")
        if sig not in TEST_SIGNALS:
            raise ValueError("Unknown test signal")
        return {"type": "test", "signal": sig}
    raise ValueError(f"Unknown source type: {kind}")


class Handler(BaseHTTPRequestHandler):
    server_version = "AudioProc2000/1.0"

    def log_message(self, fmt, *args):  # keep the console quiet (HLS polling is chatty)
        pass

    # -- helpers ----------------------------------------------------------
    def _authorised(self) -> bool:
        if not (AUTH_USER and AUTH_PASS) or self.path.startswith("/hls/"):
            return True
        want = "Basic " + base64.b64encode(f"{AUTH_USER}:{AUTH_PASS}".encode()).decode()
        if self.headers.get("Authorization") == want:
            return True
        self.send_response(401)
        self.send_header("WWW-Authenticate", 'Basic realm="Audio Proc - 2000"')
        self.end_headers()
        return False

    def _json(self, obj, code=200):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        if n > 1_000_000:
            raise ValueError("request too large")
        return json.loads(self.rfile.read(n) or b"{}")

    def _file(self, path, cache="no-cache"):
        try:
            with open(path, "rb") as f:
                data = f.read()
        except (FileNotFoundError, IsADirectoryError):
            return self._json({"error": "not found"}, 404)
        ctype = {".m3u8": "application/vnd.apple.mpegurl", ".ts": "video/mp2t"}.get(
            os.path.splitext(path)[1]) or mimetypes.guess_type(path)[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", cache)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    # -- routes -----------------------------------------------------------
    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.end_headers()

    def do_GET(self):
        if not self._authorised():
            return
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            return self._file(os.path.join(STATIC, "index.html"))
        if path.startswith("/hls/"):
            name = os.path.basename(path)
            if not name.endswith((".m3u8", ".ts")):
                return self._json({"error": "not found"}, 404)
            cache = "no-cache" if name.endswith(".m3u8") else "max-age=60"
            return self._file(os.path.join(HLS_DIR, name), cache)
        if path.startswith("/static/"):
            name = os.path.basename(path)
            return self._file(os.path.join(STATIC, name), "max-age=3600")
        if path == "/api/state":
            return self._json({
                "settings": engine.get_settings(),
                "meters": engine.meters,
                "source": proc.status(),
                "log": proc.log.get()[-12:],
            })
        if path == "/api/devices":
            return self._json({
                "soundcards": list_sound_cards(),
                "tests": [{"id": k, "name": v[0]} for k, v in TEST_SIGNALS.items()],
            })
        if path == "/api/defaults":
            return self._json(DEFAULT_SETTINGS)
        self._json({"error": "not found"}, 404)

    def do_POST(self):
        if not self._authorised():
            return
        path = urlparse(self.path).path
        try:
            body = self._body()
            if path == "/api/settings":
                old = engine.get_settings()
                new = engine.set_settings(deep_merge(engine.get_settings(), body))
                if new["output"]["bitrate_k"] != old["output"]["bitrate_k"]:
                    proc.output.restart(new["output"]["bitrate_k"])
                save_state()
                return self._json({"settings": new})
            if path == "/api/reset":
                engine.set_settings(DEFAULT_SETTINGS)
                save_state()
                return self._json({"settings": engine.get_settings()})
            if path == "/api/source":
                spec = validate_source(body)
                proc.set_source(spec)
                save_state()
                return self._json({"source": proc.status()})
        except (ValueError, json.JSONDecodeError) as e:
            return self._json({"error": str(e)}, 400)
        self._json({"error": "not found"}, 404)


def main():
    proc.start()
    if saved.get("source"):
        try:
            proc.set_source(validate_source(saved["source"]))
        except ValueError as e:
            proc.log.add("system", f"could not restore last source: {e}")
    httpd = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    httpd.daemon_threads = True

    def stop(*_):
        threading.Thread(target=httpd.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    print(f"Audio Proc - 2000 listening on http://0.0.0.0:{PORT}  (HLS: /hls/stream.m3u8)", flush=True)
    httpd.serve_forever()
    proc.shutdown()
    sys.exit(0)


if __name__ == "__main__":
    main()
