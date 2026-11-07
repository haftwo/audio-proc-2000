"""Sanity tests for the DSP chain. Run: python3 -m tests.test_dsp (from the project root)."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from app.engine import Engine, BLOCK, SR, biquad_sos, lin_to_db, DEFAULT_SETTINGS
from scipy.signal import sosfreqz
import copy


def run(engine, sig):
    out = []
    for i in range(0, len(sig) - BLOCK + 1, BLOCK):
        out.append(engine.process(sig[i:i + BLOCK]))
    return np.concatenate(out)


def tone(freq, db, secs, sr=SR):
    t = np.arange(int(secs * sr)) / sr
    x = (10 ** (db / 20)) * np.sin(2 * np.pi * freq * t)
    return np.stack([x, x], axis=1)


def test_limiter_ceiling():
    s = copy.deepcopy(DEFAULT_SETTINGS)
    s["gain"]["db"] = 18; s["limiter"]["ceiling_db"] = -1.0
    e = Engine(s)
    y = run(e, tone(100, -3, 3))
    pk = lin_to_db(np.abs(y).max())
    assert pk <= -1.0 + 1e-6, pk
    print(f"limiter: hot input peaks at {pk:.3f} dBFS (ceiling -1.0)  OK")


def test_eq_response():
    sos = biquad_sos("peak", 1000, 6, 1.0)
    w, h = sosfreqz(sos.reshape(1, 6), worN=[1000], fs=SR)
    g = 20 * np.log10(abs(h[0]))
    assert abs(g - 6) < 0.05, g
    sos = biquad_sos("lowshelf", 80, -6, 0.71)
    _, h = sosfreqz(sos.reshape(1, 6), worN=[20], fs=SR)
    assert abs(20 * np.log10(abs(h[0])) + 6) < 0.6
    print(f"eq: +6 dB peak at 1 kHz measures {g:.2f} dB  OK")


def test_compressor_ratio():
    s = copy.deepcopy(DEFAULT_SETTINGS)
    for k in ("normalise", "eq", "limiter"):
        s[k]["enabled"] = False
    s["comp"].update(threshold_db=-30, ratio=4, knee_db=0, makeup_db=0)
    e = Engine(s)
    y = run(e, tone(1000, -10, 2))
    rms_db = lin_to_db(np.sqrt(np.mean(y[-SR // 2:, 0] ** 2)))
    # -10 dBFS peak sine = -13 dB RMS; 17 dB over -> 4.25 dB over -> -25.75 dB RMS
    assert abs(rms_db - (-25.75)) < 1.0, rms_db
    print(f"compressor 4:1 @ -30: output RMS {rms_db:.2f} dB (expect ~-25.75)  OK")


def test_normaliser_target():
    s = copy.deepcopy(DEFAULT_SETTINGS)
    for k in ("comp", "eq", "limiter"):
        s[k]["enabled"] = False
    s["normalise"].update(target_lufs=-18, speed_s=1.0)
    e = Engine(s)
    run(e, tone(1000, -26, 10))
    m = e.meters["out_lufs_m"]
    assert abs(m - (-18)) < 0.5, m
    print(f"normaliser: quiet -26 dBFS tone brought to {m} LUFS (target -18)  OK")


def test_speed():
    e = Engine()
    sig = tone(440, -12, 10) + 0.05 * np.random.randn(10 * SR, 2)
    t = time.perf_counter(); run(e, sig); dt = time.perf_counter() - t
    print(f"speed: 10 s of stereo audio processed in {dt*1000:.0f} ms ({10/dt:.0f}x real time)")
    assert dt < 2.0


if __name__ == "__main__":
    test_eq_response(); test_limiter_ceiling(); test_compressor_ratio(); test_normaliser_target(); test_speed()
    print("all DSP tests passed")
