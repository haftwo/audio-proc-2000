"""
Audio Proc - 2000 :: inputs, HLS output and the real-time clock.

Every input is decoded by ffmpeg to raw 48 kHz stereo float32 on stdout, so the
DSP never cares where audio came from:

    HLS / any URL ffmpeg can read   ffmpeg -i <url>
    Local sound card (ALSA)         ffmpeg -f alsa -i plughw:CARD=<id>,DEV=<n>
    Test signals                    ffmpeg -re -f lavfi -i <generator>

A clock thread pulls one block every 21.3 ms, runs the DSP and feeds a single
long-running ffmpeg encoder that writes the HLS output. If the source drops
out, silence is inserted, so the output stream never breaks.
"""
from __future__ import annotations

import collections
import os
import queue
import re
import shutil
import subprocess
import threading
import time

import numpy as np

from .engine import BLOCK, BLOCK_DUR, CHANNELS, SR, Engine

FFMPEG = os.environ.get("FFMPEG", "ffmpeg")
BLOCK_BYTES = BLOCK * CHANNELS * 4

TEST_SIGNALS = {
    "music": ("Dynamic test programme (varying level)",
              "aevalsrc=exprs='0.45*(0.55+0.45*sin(2*PI*0.2*t))*(sin(2*PI*110*t)+0.5*sin(2*PI*440*t)+0.25*sin(2*PI*1760*t)+0.1*sin(2*PI*6000*t))/1.85"
              "|0.45*(0.55+0.45*sin(2*PI*0.2*t+1))*(sin(2*PI*110*t)+0.5*sin(2*PI*440*t)+0.25*sin(2*PI*1760*t)+0.1*sin(2*PI*6000*t))/1.85'"
              ":s=48000:c=stereo"),
    "tone": ("1 kHz tone, -18 dBFS", "sine=frequency=1000:sample_rate=48000,pan=stereo|c0=c0|c1=c0"),
    "pink": ("Pink noise, -20 dBFS RMS", "anoisesrc=color=pink:amplitude=0.1:sample_rate=48000,pan=stereo|c0=c0|c1=c0"),
}


def list_sound_cards() -> list[dict]:
    """Capture-capable ALSA devices, read straight from /proc/asound (no extra libs)."""
    cards, devices = {}, []
    try:
        with open("/proc/asound/cards") as f:
            for line in f:
                m = re.match(r"\s*(\d+)\s+\[(\S+)\s*\]:\s*(.*?)\s+-\s+(.*)", line)
                if m:
                    cards[m.group(1)] = (m.group(2), m.group(4).strip())
        with open("/proc/asound/pcm") as f:
            for line in f:
                m = re.match(r"(\d+)-(\d+):\s*([^:]*):\s*([^:]*):(.*)", line)
                if not m or "capture" not in m.group(5):
                    continue
                card, dev = str(int(m.group(1))), int(m.group(2))
                cid, cname = cards.get(card, (card, f"Card {card}"))
                devices.append({
                    "id": f"plughw:CARD={cid},DEV={dev}",
                    "name": f"{cname} — {m.group(3).strip()} (hw:{card},{dev})",
                })
    except FileNotFoundError:
        pass
    return devices


class LogTail:
    def __init__(self, n=40):
        self.lines = collections.deque(maxlen=n)
        self.lock = threading.Lock()

    def add(self, who, line):
        line = line.strip()
        if line:
            with self.lock:
                self.lines.append(f"{time.strftime('%H:%M:%S')} [{who}] {line}")

    def get(self):
        with self.lock:
            return list(self.lines)


def _drain(proc, who, log: LogTail):
    for raw in iter(proc.stderr.readline, b""):
        log.add(who, raw.decode(errors="replace"))


class Source:
    """One ffmpeg decoder process feeding a bounded queue; restarts itself on failure."""

    def __init__(self, spec: dict, q: queue.Queue, log: LogTail):
        self.spec, self.q, self.log = spec, q, log
        self.stop_evt = threading.Event()
        self.proc = None
        self.state = "starting"
        self.error = ""
        self.blocks = 0
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def command(self) -> list[str]:
        kind = self.spec["type"]
        base = [FFMPEG, "-hide_banner", "-nostdin", "-loglevel", "warning"]
        if kind == "hls":
            inp = ["-user_agent", "AudioProc2000/1.0", "-i", self.spec["url"]]
        elif kind == "soundcard":
            inp = ["-f", "alsa", "-sample_rate", str(SR), "-channels", str(CHANNELS),
                   "-thread_queue_size", "1024", "-i", self.spec["device"]]
        elif kind == "test":
            gen = TEST_SIGNALS.get(self.spec.get("signal", "music"), TEST_SIGNALS["music"])[1]
            inp = ["-re", "-f", "lavfi", "-i", gen]
        else:
            raise ValueError(f"unknown source type {kind}")
        out = ["-vn", "-ac", str(CHANNELS), "-ar", str(SR), "-f", "f32le", "pipe:1"]
        return base + inp + out

    def _run(self):
        backoff = 1.0
        while not self.stop_evt.is_set():
            try:
                self.proc = subprocess.Popen(self.command(), stdout=subprocess.PIPE,
                                             stderr=subprocess.PIPE, bufsize=0)
            except Exception as e:  # ffmpeg missing etc.
                self.state, self.error = "error", str(e)
                self.log.add("input", str(e))
                self.stop_evt.wait(5)
                continue
            threading.Thread(target=_drain, args=(self.proc, "input", self.log), daemon=True).start()
            self.state = "connecting"
            buf = bytearray()
            while not self.stop_evt.is_set():
                chunk = self.proc.stdout.read(BLOCK_BYTES - len(buf))
                if not chunk:
                    break
                buf += chunk
                if len(buf) < BLOCK_BYTES:
                    continue
                block = np.frombuffer(bytes(buf), dtype=np.float32).reshape(BLOCK, CHANNELS)
                buf.clear()
                self.state, self.error, backoff = "running", "", 1.0
                self.blocks += 1
                while not self.stop_evt.is_set():
                    try:
                        self.q.put(block, timeout=0.25)   # back-pressure (HLS start-up bursts)
                        break
                    except queue.Full:
                        if self.spec["type"] == "soundcard":  # never stall a live card: drop oldest
                            try:
                                self.q.get_nowait()
                            except queue.Empty:
                                pass
            self._kill()
            if self.stop_evt.is_set():
                break
            self.state = "reconnecting"
            self.error = "input ended or failed — retrying"
            self.log.add("input", f"source exited, retrying in {backoff:.0f}s")
            self.stop_evt.wait(backoff)
            backoff = min(backoff * 2, 15)
        self.state = "stopped"

    def _kill(self):
        p, self.proc = self.proc, None
        if p and p.poll() is None:
            p.terminate()
            try:
                p.wait(2)
            except subprocess.TimeoutExpired:
                p.kill()

    def stop(self):
        self.stop_evt.set()
        self._kill()
        self.thread.join(timeout=3)


class HLSOutput:
    """Long-running AAC/HLS encoder fed with processed PCM on stdin."""

    def __init__(self, out_dir: str, log: LogTail, bitrate_k: int = 128, seg_time: int = 2, list_size: int = 6):
        self.dir, self.log = out_dir, log
        self.bitrate_k, self.seg_time, self.list_size = bitrate_k, seg_time, list_size
        self.proc = None
        self.lock = threading.Lock()

    def start(self):
        os.makedirs(self.dir, exist_ok=True)
        for f in os.listdir(self.dir):
            if f.endswith((".ts", ".m3u8", ".tmp")):
                os.remove(os.path.join(self.dir, f))
        cmd = [FFMPEG, "-hide_banner", "-nostdin", "-loglevel", "warning",
               "-f", "f32le", "-ar", str(SR), "-ac", str(CHANNELS), "-i", "pipe:0",
               "-c:a", "aac", "-b:a", f"{self.bitrate_k}k",
               "-f", "hls", "-hls_time", str(self.seg_time), "-hls_list_size", str(self.list_size),
               "-hls_flags", "delete_segments+independent_segments+temp_file+program_date_time",
               "-hls_segment_filename", os.path.join(self.dir, "seg_%06d.ts"),
               os.path.join(self.dir, "stream.m3u8")]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
        threading.Thread(target=_drain, args=(self.proc, "output", self.log), daemon=True).start()
        self.log.add("output", f"HLS encoder started (AAC {self.bitrate_k} kbps, {self.seg_time}s segments)")

    def write(self, pcm: np.ndarray):
        with self.lock:
            try:
                self.proc.stdin.write(pcm.tobytes())
            except (BrokenPipeError, OSError, AttributeError):
                self.log.add("output", "encoder pipe broke — restarting")
                self._stop_locked()
                self.start()

    def restart(self, bitrate_k=None):
        with self.lock:
            if bitrate_k:
                self.bitrate_k = int(bitrate_k)
            self._stop_locked()
            self.start()

    def _stop_locked(self):
        p, self.proc = self.proc, None
        if p:
            try:
                p.stdin.close()
            except Exception:
                pass
            try:
                p.wait(3)
            except subprocess.TimeoutExpired:
                p.kill()

    def stop(self):
        with self.lock:
            self._stop_locked()


class Processor:
    """Ties source -> engine -> HLS output together on a real-time clock."""

    def __init__(self, engine: Engine, hls_dir: str):
        self.engine = engine
        self.log = LogTail()
        self.q: queue.Queue = queue.Queue(maxsize=int(3.0 / BLOCK_DUR))   # up to ~3 s buffered
        self.source: Source | None = None
        self.spec: dict | None = None
        self.src_lock = threading.Lock()
        self.output = HLSOutput(hls_dir, self.log, engine.get_settings()["output"]["bitrate_k"])
        self.underruns = 0
        self.running = False
        if not shutil.which(FFMPEG):
            self.log.add("system", "ffmpeg not found on PATH — install ffmpeg")

    def start(self):
        self.output.start()
        self.running = True
        threading.Thread(target=self._clock, daemon=True).start()

    def _clock(self):
        silence = np.zeros((BLOCK, CHANNELS), dtype=np.float32)
        nxt = time.monotonic()
        while self.running:
            try:
                block = self.q.get_nowait()
            except queue.Empty:
                block = silence
                if self.source and self.source.state == "running":
                    self.underruns += 1
            try:
                out = self.engine.process(block)
            except Exception as e:  # never let a DSP bug kill the stream
                self.log.add("dsp", f"error: {e}")
                out = silence
            self.output.write(out)
            nxt += BLOCK_DUR
            delay = nxt - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            elif delay < -0.5:          # we fell badly behind (e.g. host stall): resync
                nxt = time.monotonic()

    def set_source(self, spec: dict | None):
        with self.src_lock:
            if self.source:
                self.source.stop()
                self.source = None
            while not self.q.empty():
                try:
                    self.q.get_nowait()
                except queue.Empty:
                    break
            self.spec = spec
            if spec:
                self.log.add("input", f"selecting {describe(spec)}")
                self.source = Source(spec, self.q, self.log)
            else:
                self.log.add("input", "source stopped")

    def status(self) -> dict:
        s = self.source
        return {
            "spec": self.spec,
            "label": describe(self.spec) if self.spec else "No source",
            "state": s.state if s else "idle",
            "error": s.error if s else "",
            "buffer_ms": round(self.q.qsize() * BLOCK_DUR * 1000),
            "underruns": self.underruns,
        }

    def shutdown(self):
        self.running = False
        self.set_source(None)
        self.output.stop()


def describe(spec: dict | None) -> str:
    if not spec:
        return "No source"
    if spec["type"] == "hls":
        return f"HLS: {spec.get('url', '')}"
    if spec["type"] == "soundcard":
        return f"Sound card: {spec.get('name') or spec.get('device')}"
    if spec["type"] == "test":
        return f"Test: {TEST_SIGNALS.get(spec.get('signal', 'music'), ('?',))[0]}"
    return spec["type"]
