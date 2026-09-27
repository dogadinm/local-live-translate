# Local Live Translate

Live subtitles for anything playing on your PC — a video in the browser, a
player, Discord, a game. It recognises the speech, translates it, and shows the
result in a strip on top of every window. The target language switches on the
fly. Everything runs locally: no cloud, no API keys.

```
everything playing on the PC
        │  WASAPI loopback (no driver needed)
        ▼
  speech segmentation      ← hands out the phrase every ~0.8 s while it is spoken
        ▼
  faster-whisper turbo     ← recognises it, detects the language per phrase
        ▼
  NLLB-200                 ← translates into the chosen language
        ▼
  subtitle strip           ← on top of everything, rewritten as speech goes on
```

## Requirements

- Windows, Python 3.10+
- An NVIDIA GPU is not required but the streaming mode needs one to keep up

## Install

```
pip install -r requirements.txt
python convert_model.py     # once: downloads NLLB-200 and squeezes it to int8 (~660 MB)
```

`convert_model.py` needs `torch`, but only while converting. Afterwards you can
remove it — nothing at runtime uses it. Both models are run by CTranslate2, a
C++ engine, with no Python framework on top.

## Run

```
python main.py                  # translate into Russian, source detected automatically
python main.py --tgt en         # into English
python main.py --src cs         # pin the source language
python main.py --list-devices   # show output devices
python main.py --audio-file sample.wav  # read a PCM16 mono 16 kHz WAV
```

Whisper models are downloaded on first run (~1.6 GB for turbo).

| Flag | What it does |
|---|---|
| `--tgt` | Target language, `ru` by default. Also switchable in the window |
| `--src` | Source language. Defaults to `Auto` — detected again on every phrase |
| `--whisper` | Whisper model, `large-v3-turbo` by default. Smaller and faster: `small`, `medium` |
| `--out-device` | Listen to a specific output device instead of the current one |
| `--audio-file` | Read a PCM16 mono 16 kHz WAV instead of system audio; conflicts with `--out-device` |
| `--metrics DIRECTORY` | Save timing events, resource samples and a comparison-ready summary |

File input is paced at the recording's original speed, but does not play sound
through your speakers. After EOF the last phrase is finalized and queued work
is processed; the subtitle window stays open until you close it. Unsupported
WAV formats are rejected before loading the models. For unpaced tests,
instantiate `FileAudioSource(path, realtime=False)` directly.

The window has two dropdowns: **Source** — the spoken language (`Auto`, or a
specific one, which then overrides detection) — and **Translate to**. Both take
effect for the next emitted draft or final. Results using previous settings
are discarded. Drag the strip with the mouse, quit with `Esc` or `✕`.

## Benchmark comparison

Use the same WAV, languages and model for both runs:

```powershell
python main.py --audio-file benchmark.wav --src en --tgt ru --metrics runs/before
# After changing the implementation:
python main.py --audio-file benchmark.wav --src en --tgt ru --metrics runs/after
python compare_metrics.py runs/before runs/after
```

Wait for `Finished — all phrases processed` and the `[metrics] saved` console
message before closing the window. `summary.json` is saved after the UI has
handled all queued subtitle updates. `events.jsonl` contains detailed timings
and final original/translated text. Early close or failure is marked as an
incomplete run. Existing `events.jsonl` files are never overwritten; use names
such as `runs/before-02` for repetitions.

The report separates model loading from processing, records complete Whisper
calls (including generator consumption), translation requests and actual NLLB
inference calls, job queue waiting times, draft/final display lag and first
subtitle delay. Process CPU, Windows RAM and queue sizes are sampled every
0.5 seconds. CPU uses one core as 100%. GPU utilization, VRAM and power are not
collected. Display timing means Tk label update, not physical screen refresh.

There is no automatic warmup: first inference is included and recorded as
such. Repeat each configuration three times under similar background load.
Comparison warns about differing file hashes, model/settings/runtime metadata
or incomplete runs. Review saved final texts as well: `finals_not_displayed`
can include empty recognition and invalidated settings, not just lost work.

## Tests

```
python test_segmenter.py
python -m unittest test_audio_sources test_app test_metrics
```

Uses the runtime dependencies, without loading model weights or opening audio
devices or windows. If you install pytest, `pytest test_segmenter.py test_audio_sources.py test_app.py` works
too — the tests are written to suit both.

They cover speech segmentation, event identities and timestamps, draft
coalescing, metadata propagation through workers and the overlay, and target
language changes during translation. Audio is synthetic and model inference
and UI widgets are replaced with test doubles.
Source tests also cover WAV samples, EOF tails, device boundaries, bounded
queue shutdown, capture reopening and propagation of completion through all
three workers.
Controller tests cover settings changes during inference, stale queued UI
updates, source and model failures, EOF completion and shutdown during loading.

## Application control

`app.ApplicationController` owns startup, shutdown, language settings, the
selected audio source and worker error handling. `main.py` only parses CLI
options and assembles the application. Models load in the background while the
window displays progress. Fatal errors stop processing and stay visible in the
window; recoverable capture errors show a reconnecting status.

`config.ProcessingSettings` is immutable. The segmentation worker attaches a
snapshot to each emitted draft or final. Source/target language changes bump
its version; every downstream stage uses that same snapshot. Obsolete work is
skipped before inference, checked again after inference, and checked once more
by the UI immediately before display. A new draft of an ongoing phrase can use
the new settings; a job already running never changes language midway through.

`engines.py` defines model interfaces. Whisper returns both the language of its
text and the detected audio language (different when translating directly into
English). The controller forwards detection results to the UI. Engines neither
read picker state nor call window methods.

## Pipeline messages

`events.py` defines the stage boundaries: `AudioChunk` → `SpeechSegment` →
`Transcript` → `Translation`. Drafts and finals share a `PhraseMeta` containing
the capture stream ID, phrase ID, revision, audio timestamps, final flag and
settings version. Recognition and translation preserve this metadata through
the overlay's queue. Translation also records the target actually used.

`audio_sources.py` owns the `AudioSource` protocol, `SystemLoopbackSource` and
`FileAudioSource`. Sources only produce audio blocks and ordered
`AudioStreamEnded` boundaries. `audio.py` owns speech segmentation, and
`pipeline.worker_segment` connects either source to the recognition queue.

Each recorder opening creates a new stream ID; the segmentation worker creates
a fresh segmenter for it. Timestamps use a
monotonic capture origin plus sample offsets; they are audio boundaries, not
model completion times. Published NumPy buffers must not be modified.
`settings_version` identifies the controller's snapshot attached to each job.
Queue coalescing replaces only consecutive drafts of the same stream, phrase
and settings version with a strictly newer revision. Finals are retained.

`SpeechSegmenter.finish()` finalizes remaining speech including an incomplete
30 ms frame, without padding its samples or timestamps. EOF travels through
recognition and translation after the final phrase; closing the application
instead stops workers without draining pending work. In-flight model calls
cannot be cancelled, so shutdown only waits briefly for those workers.

The source queue holds at most 20 events (about two seconds of audio blocks).
File reading waits for capacity. Live capture treats a queue timeout as a
discontinuity: it closes the recorder, publishes an error boundary, and retries
with a new stream ID. Dropped audio is never silently joined to later audio.
Recognition and translation queues are still unbounded; their overload policy
is a separate future step.

## How the audio is captured

WASAPI loopback taps the mix Windows sends to the output device. It is the same
signal that reaches your headphones, taken just before them — no microphone is
involved and nothing is lost. Capture follows whatever output device is current,
and moves by itself if you switch from headphones to speakers.

System and headphone volume do not affect the capture. Muting **an individual
app** in the Windows volume mixer does: by the tap point that audio is already
silent.

## Two things worth knowing

**Drafts get rewritten.** While a phrase is still being spoken it is handed to
Whisper every 0.8 s, so the line updates as you listen rather than waiting for a
pause. Drafts are shown dimmed and the final version replaces them in white.
Mid-phrase they can be wrong — the meaning only settles at the end. That is a
deliberate trade for showing text immediately.

**An English target can skip translation inference.** Compatible Whisper
models translate into English directly, so `--tgt en` bypasses NLLB inference.
NLLB is still loaded at startup; loading it only when needed is a later
resource optimization.

This only works on models trained for the translation task, and the default
`large-v3-turbo` is not one of them: turbo is a pruned large-v3 retrained on
transcription alone, and the distil models are English-only. Asked to translate,
they do not fail — they silently transcribe, which looks like working code
returning untranslated text. So the shortcut is taken only for a model that
supports it (`small`, `medium`, `large-v3`), and everything else goes through
NLLB as usual.

## Limits

- **Windows only.** Capture rests on WASAPI loopback. macOS has no system audio
  tap at all; it needs BlackHole or ScreenCaptureKit.
- **NVIDIA only for the streaming mode.** CTranslate2 supports CUDA and CPU but
  not Metal, so on Apple Silicon both models fall back to the CPU and cannot
  keep up with the 0.8 s cycle.
- **All system audio at once**, not a chosen application. Windows 11 can capture
  per process, but no Python library wraps that API — it needs hand-written COM.
- **The GPU works almost continuously**: recognition and translation run about
  once a second rather than once per phrase.
