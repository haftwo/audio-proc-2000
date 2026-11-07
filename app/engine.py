"""
Audio Proc - 2000 :: DSP engine

Signal chain (all stereo-linked, 48 kHz float32, block based):

    Input Gain -> Normaliser (loudness AGC) -> 5-band EQ -> Compressor -> Limiter

Everything is vectorised with numpy/scipy. The dynamics processors work on
32-sample "frames" for their control signal (gain computer + ballistics) and
interpolate the gain per sample, which keeps Python overhead tiny while still
being sample-accurate where it matters (the limiter is a true brickwall in the
sample domain).
"""
from __future__ import annotations

import copy
import math
import threading

import numpy as np
from scipy.signal import sosfilt

SR = 48000
CHANNELS = 2
BLOCK = 1024                 # frames per processing block (~21.3 ms)
FRAME = 32                   # control-rate frame for dynamics
BLOCK_DUR = BLOCK / SR

DEFAULT_SETTINGS = {
    "bypass": False,
    "gain": {"enabled": True, "db": 0.0},
    "normalise": {
        "enabled": True,
        "target_lufs": -18.0,
        "max_gain_db": 12.0,
        "max_cut_db": 12.0,
        "speed_s": 3.0,
        "gate_lufs": -45.0,
    },
    "eq": {
        "enabled": True,
        "bands": [
            {"name": "Low",      "type": "lowshelf",  "freq": 80.0,    "gain": 0.0, "q": 0.71},
            {"name": "Low-Mid",  "type": "peak",      "freq": 250.0,   "gain": 0.0, "q": 1.0},
            {"name": "Mid",      "type": "peak",      "freq": 1000.0,  "gain": 0.0, "q": 1.0},
            {"name": "High-Mid", "type": "peak",      "freq": 4000.0,  "gain": 0.0, "q": 1.0},
            {"name": "High",     "type": "highshelf", "freq": 10000.0, "gain": 0.0, "q": 0.71},
        ],
    },
    "comp": {
        "enabled": True,
        "threshold_db": -20.0,
        "ratio": 3.0,
        "attack_ms": 10.0,
        "release_ms": 150.0,
        "knee_db": 6.0,
        "makeup_db": 3.0,
    },
    "limiter": {"enabled": True, "ceiling_db": -1.0, "release_ms": 50.0},
    "output": {"bitrate_k": 128},
}

# Parameter limits, used to sanitise anything coming from the UI.
LIMITS = {
    "gain.db": (-24, 24),
    "normalise.target_lufs": (-36, -6),
    "normalise.max_gain_db": (0, 24),
    "normalise.max_cut_db": (0, 24),
    "normalise.speed_s": (0.2, 30),
    "normalise.gate_lufs": (-70, -20),
    "eq.freq": (20, 20000),
    "eq.gain": (-15, 15),
    "eq.q": (0.1, 10),
    "comp.threshold_db": (-60, 0),
    "comp.ratio": (1, 20),
    "comp.attack_ms": (0.1, 200),
    "comp.release_ms": (5, 2000),
    "comp.knee_db": (0, 24),
    "comp.makeup_db": (-12, 24),
    "limiter.ceiling_db": (-20, 0),
    "limiter.release_ms": (1, 1000),
    "output.bitrate_k": (32, 320),
}


def db_to_lin(db):
    return 10.0 ** (np.asarray(db) / 20.0)


def lin_to_db(x, floor=-120.0):
    x = np.maximum(np.asarray(x, dtype=np.float64), 1e-12)
    return np.maximum(20.0 * np.log10(x), floor)


# --------------------------------------------------------------------------
# Biquads (RBJ Audio EQ Cookbook)
# --------------------------------------------------------------------------
def biquad_sos(kind: str, f0: float, gain_db: float, q: float, sr: int = SR) -> np.ndarray:
    f0 = min(max(f0, 10.0), sr * 0.49)
    a = 10 ** (gain_db / 40.0)
    w0 = 2 * math.pi * f0 / sr
    cw, sw = math.cos(w0), math.sin(w0)
    alpha = sw / (2 * max(q, 1e-3))
    if kind == "peak":
        b0, b1, b2 = 1 + alpha * a, -2 * cw, 1 - alpha * a
        a0, a1, a2 = 1 + alpha / a, -2 * cw, 1 - alpha / a
    elif kind == "lowshelf":
        sa = 2 * math.sqrt(a) * alpha
        b0 = a * ((a + 1) - (a - 1) * cw + sa)
        b1 = 2 * a * ((a - 1) - (a + 1) * cw)
        b2 = a * ((a + 1) - (a - 1) * cw - sa)
        a0 = (a + 1) + (a - 1) * cw + sa
        a1 = -2 * ((a - 1) + (a + 1) * cw)
        a2 = (a + 1) + (a - 1) * cw - sa
    elif kind == "highshelf":
        sa = 2 * math.sqrt(a) * alpha
        b0 = a * ((a + 1) + (a - 1) * cw + sa)
        b1 = -2 * a * ((a - 1) + (a + 1) * cw)
        b2 = a * ((a + 1) + (a - 1) * cw - sa)
        a0 = (a + 1) - (a - 1) * cw + sa
        a1 = 2 * ((a - 1) - (a + 1) * cw)
        a2 = (a + 1) - (a - 1) * cw - sa
    else:
        raise ValueError(kind)
    return np.array([b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0])


# ITU-R BS.1770 K-weighting at 48 kHz (pre-filter shelf + RLB high-pass)
K_WEIGHT_SOS = np.array([
    [1.53512485958697, -2.69169618940638, 1.19839281085285, 1.0, -1.69065929318241, 0.73248077421585],
    [1.0, -2.0, 1.0, 1.0, -1.99004745483398, 0.99007225036621],
])


class LoudnessMeter:
    """Momentary (400 ms) and short-term (3 s) loudness, BS.1770 style (ungated)."""

    def __init__(self):
        self.zi = np.zeros((K_WEIGHT_SOS.shape[0], 2, CHANNELS))
        self.m_hist = np.zeros(max(1, round(0.4 / BLOCK_DUR)))
        self.s_hist = np.zeros(max(1, round(3.0 / BLOCK_DUR)))
        self.i = 0

    def push(self, x: np.ndarray) -> tuple[float, float]:
        y, self.zi = sosfilt(K_WEIGHT_SOS, x, axis=0, zi=self.zi)
        ms = float(np.sum(np.mean(y * y, axis=0)))  # sum of channel mean-squares
        self.m_hist[self.i % len(self.m_hist)] = ms
        self.s_hist[self.i % len(self.s_hist)] = ms
        self.i += 1
        return self.lufs(self.m_hist.mean()), self.lufs(self.s_hist.mean())

    @staticmethod
    def lufs(ms: float) -> float:
        return -0.691 + 10 * math.log10(max(ms, 1e-12))


# --------------------------------------------------------------------------
# Processing modules
# --------------------------------------------------------------------------
def _ramp(prev: float, frame_gains: np.ndarray, n: int) -> np.ndarray:
    """Per-sample gain: linear ramps ending at each frame's gain value."""
    pos = np.concatenate(([-1], np.arange(1, len(frame_gains) + 1) * FRAME - 1))
    val = np.concatenate(([prev], frame_gains))
    return np.interp(np.arange(n), pos, val)


class Normaliser:
    """Slow loudness AGC: steers K-weighted momentary loudness toward a target."""

    def __init__(self):
        self.meter = LoudnessMeter()
        self.gain_db = 0.0

    def process(self, x, p):
        m, _ = self.meter.push(x)
        prev = self.gain_db
        if m > p["gate_lufs"]:
            want = float(np.clip(p["target_lufs"] - m, -p["max_cut_db"], p["max_gain_db"]))
            coeff = 1.0 - math.exp(-BLOCK_DUR / max(p["speed_s"], 0.05))
            self.gain_db += (want - self.gain_db) * coeff
        # below the gate: hold current gain (don't pump up silence / noise floor)
        g = np.linspace(db_to_lin(prev), db_to_lin(self.gain_db), len(x), endpoint=True)
        return x * g[:, None]


class Equaliser:
    def __init__(self):
        self.key = None
        self.sos = None
        self.zi = None

    def _update(self, bands):
        key = tuple((b["type"], b["freq"], b["gain"], b["q"]) for b in bands)
        if key == self.key:
            return
        self.key = key
        self.sos = np.vstack([biquad_sos(b["type"], b["freq"], b["gain"], b["q"]) for b in bands])
        if self.zi is None or self.zi.shape[0] != self.sos.shape[0]:
            self.zi = np.zeros((self.sos.shape[0], 2, CHANNELS))

    def process(self, x, p):
        self._update(p["bands"])
        y, self.zi = sosfilt(self.sos, x, axis=0, zi=self.zi)
        return y


class Compressor:
    """Feed-forward, stereo-linked, soft-knee RMS compressor."""

    def __init__(self):
        self.env_db = 0.0     # current gain reduction (<= 0 dB)
        self.prev_lin = 1.0
        self.last_gr = 0.0

    @staticmethod
    def curve(level_db, t, r, w):
        level_db = np.asarray(level_db, dtype=np.float64)
        over = level_db - t
        out = level_db.copy()
        if w > 0:
            knee = np.abs(2 * over) <= w
            out[knee] = level_db[knee] + (1 / r - 1) * (over[knee] + w / 2) ** 2 / (2 * w)
            hi = 2 * over > w
        else:
            hi = over > 0
        out[hi] = t + over[hi] / r
        return out - level_db  # gain change (<= 0)

    def process(self, x, p):
        n = len(x)
        frames = x.reshape(n // FRAME, FRAME, CHANNELS)
        rms = np.sqrt(np.mean(frames * frames, axis=1)).max(axis=1)        # linked
        target = self.curve(lin_to_db(rms), p["threshold_db"], max(p["ratio"], 1.0), p["knee_db"])
        fdur = FRAME / SR
        att = math.exp(-fdur / max(p["attack_ms"] / 1000, 1e-5))
        rel = math.exp(-fdur / max(p["release_ms"] / 1000, 1e-5))
        g = np.empty(len(target))
        e = self.env_db
        for i, t in enumerate(target):
            c = att if t < e else rel
            e = c * e + (1 - c) * t
            g[i] = e
        self.env_db = e
        self.last_gr = float(g.min())
        lin = db_to_lin(g + p["makeup_db"])
        gains = _ramp(self.prev_lin, lin, n)
        self.prev_lin = float(lin[-1])
        return x * gains[:, None]


class Limiter:
    """
    Look-ahead brickwall peak limiter (one 32-sample frame of look-ahead,
    ~0.67 ms latency). Gain ramps are constructed so that every sample of
    a frame is attenuated by at least that frame's required amount, then a
    final clip guarantees the ceiling.
    """

    def __init__(self):
        self.held = np.zeros((FRAME, CHANNELS))
        self.prev = 1.0
        self.last_gr = 0.0

    def process(self, x, p):
        ceiling = float(db_to_lin(p["ceiling_db"]))
        buf = np.concatenate([self.held, x])
        nf = len(buf) // FRAME
        peaks = np.abs(buf.reshape(nf, FRAME, CHANNELS)).max(axis=(1, 2))
        req = np.minimum(1.0, ceiling / np.maximum(peaks, 1e-12))
        target = np.minimum(req[:-1], req[1:])            # look-ahead by one frame
        rel = math.exp(-(FRAME / SR) / max(p["release_ms"] / 1000, 1e-5))
        g = np.empty(len(target))
        cur = self.prev
        for i, t in enumerate(target):
            cur = t if t < cur else cur + (t - cur) * (1 - rel)
            g[i] = cur
        out = buf[:-FRAME] * _ramp(self.prev, g, len(buf) - FRAME)[:, None]
        self.prev = float(g[-1])
        self.held = buf[-FRAME:].copy()
        self.last_gr = float(lin_to_db(g.min()))
        return np.clip(out, -ceiling, ceiling)


class Ballistic:
    """Peak-programme style meter value: instant rise, linear dB fall."""

    def __init__(self, fall_db_per_s=20.0, floor=-90.0, rise_up=True):
        self.v = floor
        self.fall = fall_db_per_s * BLOCK_DUR
        self.floor = floor
        self.rise_up = rise_up

    def push(self, v):
        if self.rise_up:
            self.v = v if v > self.v else max(v, self.v - self.fall, self.floor)
        else:  # for gain reduction (negative numbers): instant fall, slow return
            self.v = v if v < self.v else min(v, self.v + self.fall)
        return self.v


# --------------------------------------------------------------------------
# Engine
# --------------------------------------------------------------------------
def _clamp(key, v):
    lo, hi = LIMITS.get(key, (-1e9, 1e9))
    return float(min(max(float(v), lo), hi))


def sanitise(settings: dict) -> dict:
    """Merge incoming settings over defaults and clamp every numeric value."""
    out = copy.deepcopy(DEFAULT_SETTINGS)
    s = settings or {}
    out["bypass"] = bool(s.get("bypass", False))
    for sect in ("gain", "normalise", "comp", "limiter", "output"):
        for k, v in (s.get(sect) or {}).items():
            if k not in out[sect]:
                continue
            out[sect][k] = bool(v) if k == "enabled" else _clamp(f"{sect}.{k}", v)
    eq = s.get("eq") or {}
    out["eq"]["enabled"] = bool(eq.get("enabled", True))
    for i, b in enumerate((eq.get("bands") or [])[:5]):
        for k in ("freq", "gain", "q"):
            if k in b:
                out["eq"]["bands"][i][k] = _clamp(f"eq.{k}", b[k])
    out["output"]["bitrate_k"] = int(out["output"]["bitrate_k"])
    return out


class Engine:
    def __init__(self, settings: dict | None = None):
        self.lock = threading.Lock()
        self.settings = sanitise(settings or {})
        self.norm = Normaliser()
        self.eq = Equaliser()
        self.comp = Compressor()
        self.lim = Limiter()
        self.in_loud = LoudnessMeter()
        self.out_loud = LoudnessMeter()
        self.m = {
            "in_peak": [Ballistic(), Ballistic()],
            "out_peak": [Ballistic(), Ballistic()],
            "comp_gr": Ballistic(fall_db_per_s=15, floor=0, rise_up=False),
            "lim_gr": Ballistic(fall_db_per_s=15, floor=0, rise_up=False),
        }
        self.meters = {}

    def set_settings(self, s: dict):
        clean = sanitise(s)
        with self.lock:
            self.settings = clean
        return clean

    def get_settings(self) -> dict:
        with self.lock:
            return copy.deepcopy(self.settings)

    def process(self, x: np.ndarray) -> np.ndarray:
        with self.lock:
            s = self.settings
        x = x.astype(np.float64, copy=False)
        in_pk = lin_to_db(np.abs(x).max(axis=0))
        in_m, in_s = self.in_loud.push(x)

        y = x
        if not s["bypass"]:
            if s["gain"]["enabled"]:
                y = y * db_to_lin(s["gain"]["db"])
            if s["normalise"]["enabled"]:
                y = self.norm.process(y, s["normalise"])
            if s["eq"]["enabled"]:
                y = self.eq.process(y, s["eq"])
            if s["comp"]["enabled"]:
                y = self.comp.process(y, s["comp"])
            else:
                self.comp.last_gr = 0.0
            if s["limiter"]["enabled"]:
                y = self.lim.process(y, s["limiter"])
            else:
                self.lim.last_gr = 0.0
                y = np.clip(y, -1.0, 1.0)

        out_pk = lin_to_db(np.abs(y).max(axis=0))
        out_m, out_s = self.out_loud.push(y)
        self.meters = {
            "in_peak": [self.m["in_peak"][c].push(float(in_pk[c])) for c in range(CHANNELS)],
            "out_peak": [self.m["out_peak"][c].push(float(out_pk[c])) for c in range(CHANNELS)],
            "in_lufs_m": round(in_m, 1), "in_lufs_s": round(in_s, 1),
            "out_lufs_m": round(out_m, 1), "out_lufs_s": round(out_s, 1),
            "norm_gain": round(self.norm.gain_db, 1) if s["normalise"]["enabled"] and not s["bypass"] else 0.0,
            "comp_gr": round(self.m["comp_gr"].push(self.comp.last_gr if not s["bypass"] else 0.0), 1),
            "lim_gr": round(self.m["lim_gr"].push(self.lim.last_gr if not s["bypass"] else 0.0), 1),
        }
        return y.astype(np.float32)
