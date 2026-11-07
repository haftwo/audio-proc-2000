# Audio Proc - 2000

A small live audio processor for streams. Pick a source (HLS stream, local sound card or a test signal), shape it with a broadcast-style chain, and pull the result as an HLS stream.

```
Source ──► 1 Gain ──► 2 Normaliser ──► 3 5-band EQ ──► 4 Compressor ──► 5 Limiter ──► HLS out
(HLS / ALSA / test)                                                                  /hls/stream.m3u8
```

## Run it

**Docker (Unraid or anywhere):**

```bash
docker compose up -d --build
# open http://<host>:8020
```

**Bare metal (Linux):** you need `ffmpeg`, Python 3.10+ and `pip install numpy scipy`, then:

```bash
python3 -m app.server          # http://localhost:8020
python3 -m tests.test_dsp      # DSP sanity tests
```

## Inputs

| Source | How it works |
|---|---|
| **HLS stream** | Any `http(s)` URL ffmpeg can read: HLS `.m3u8`, plus Icecast/MP3/AAC streams. Reconnects automatically with back-off if it drops. |
| **Sound card** | Capture devices are detected from `/proc/asound` (ALSA) and opened as `plughw:CARD=…`, so mono or other-rate cards get converted to 48 kHz stereo. In Docker, pass `--device /dev/snd`. |
| **Test signal** | A varying-level test programme (to watch the dynamics work), a 1 kHz tone, or pink noise. |

## Processing

| Module | Controls | Notes |
|---|---|---|
| Gain | ±24 dB | Input trim |
| Normaliser | Target LUFS, max boost/cut, speed, gate | Slow loudness AGC on K-weighted (BS.1770) momentary loudness. Holds its gain below the gate so it doesn't pump up silence. |
| 5-band EQ | Gain / frequency per band, Q on the three mids | Low shelf 80 Hz · Low-Mid 250 Hz · Mid 1 kHz · High-Mid 4 kHz · High shelf 10 kHz (defaults) |
| Compressor | Threshold, ratio, attack, release, knee, make-up | Feed-forward, stereo-linked, soft knee |
| Limiter | Ceiling, release | Look-ahead brickwall (0.67 ms); output never exceeds the ceiling |

Each module has an IN/OUT bypass, and there's a master **Bypass all**. Double-click any slider to reset it. Settings and the last-used source are saved to `data/state.json` and restored on restart.

## Output

- `http://<host>:8020/hls/stream.m3u8`: AAC (64–320 kbps selectable), 48 kHz stereo, 2 s segments, 6-segment live window, with `PROGRAM-DATE-TIME` tags.
- The encoder runs all the time, so the output URL stays valid while you switch sources (silence fills the gaps).
- **Monitor** in the panel plays the HLS output itself (via hls.js), so it's 6–10 s behind the meters.

## Publishing through the Cloudflare tunnel

Point a subdomain (e.g. `audioproc.aka-sound.com`) at `http://<unraid-ip>:8020` in the tunnel config. Set `AUTH_USER` / `AUTH_PASS` so the control panel isn't open to the internet; `/hls/` stays unauthenticated so players can pull the stream.

## API

| Method | Path | Body |
|---|---|---|
| GET | `/api/state` | settings, meters, source status, log tail |
| GET | `/api/devices` | detected sound cards and test signals |
| POST | `/api/settings` | partial settings, e.g. `{"comp":{"threshold_db":-24}}` |
| POST | `/api/source` | `{"type":"hls","url":"…"}` · `{"type":"soundcard","device":"plughw:CARD=…,DEV=0"}` · `{"type":"test","signal":"music"}` · `{"type":null}` to stop |
| POST | `/api/reset` | restore default processing |

## Design notes and limits

- Everything runs at 48 kHz float. Each ffmpeg input decodes to raw PCM, a real-time clock thread processes 1024-sample blocks (21 ms) and feeds one long-running ffmpeg HLS encoder. Processing uses about 2% of one CPU core.
- An HLS input sends several segments at once when it starts. The app buffers up to 3 s of that burst instead of skipping ahead, which adds up to 3 s of latency. That's small next to normal HLS delay.
- Sound-card input runs on the card's clock and output runs on the system clock. Over long runs a tiny drift can force an occasional dropped or silent block. If that becomes audible, adding drift-correcting resampling is the next step.
- Sound-card detection is Linux/ALSA only. On a Mac, use HLS or test sources, or run it in Docker on the Unraid box.
- The logo loads from `transcriber2000.aka-sound.com`, and the monitor player loads hls.js from jsDelivr, so the browser needs internet access for both. The processing doesn't.
