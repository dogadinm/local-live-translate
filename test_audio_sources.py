"""Source and stream-boundary tests; no audio hardware, models or GUI needed."""
import contextlib
import io
import queue
import tempfile
import threading
import unittest
import wave
from itertools import count
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from audio import SpeechSegmenter
from audio_sources import BLOCK, SAMPLE_RATE, FileAudioSource, SystemLoopbackSource
from events import AudioChunk, AudioStreamEnded
from main import parse_args
from pipeline import worker_segment, worker_transcribe, worker_translate


def tone(size):
    return (8000 * np.sin(2 * np.pi * 180 * np.arange(size) / SAMPLE_RATE)).astype("<i2")


class AudioSourcesTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "speech.wav"

    def write_wav(self, pcm, channels=1, rate=SAMPLE_RATE, width=2):
        with wave.open(str(self.path), "wb") as writer:
            writer.setparams((channels, width, rate, 0, "NONE", "not compressed"))
            writer.writeframes(pcm.tobytes())

    def read_source(self, pcm):
        self.write_wav(pcm)
        source = FileAudioSource(self.path, realtime=False, queue_size=2)
        self.addCleanup(source.stop)
        source.start()
        events = []
        while True:
            event = source.queue.get(timeout=2)
            events.append(event)
            if isinstance(event, AudioStreamEnded):
                break
        self.assertTrue(source.done.wait(2))
        self.assertIsNone(source.error)
        return events

    def test_file_samples_timestamps_and_short_last_block(self):
        pcm = tone(BLOCK * 3 + 137)
        events = self.read_source(pcm)
        chunks = events[:-1]
        self.assertEqual([len(chunk.samples) for chunk in chunks], [BLOCK] * 3 + [137])
        np.testing.assert_array_equal(np.concatenate([c.samples for c in chunks]), pcm.astype(np.float32) / 32768)
        for index, chunk in enumerate(chunks):
            self.assertEqual(chunk.samples.dtype, np.float32)
            self.assertEqual(chunk.sample_rate, SAMPLE_RATE)
            self.assertEqual(chunk.stream_id, chunks[0].stream_id)
            self.assertAlmostEqual(chunk.started_at - chunks[0].started_at, index * BLOCK / SAMPLE_RATE)
        self.assertTrue(events[-1].source_finished)
        self.assertEqual(events[-1].stream_id, chunks[0].stream_id)

    def test_empty_file_still_emits_terminal_boundary(self):
        events = self.read_source(np.zeros(0, dtype="<i2"))
        self.assertEqual(len(events), 1)
        self.assertTrue(events[0].source_finished)

    def test_reject_unsupported_wav_and_missing_file(self):
        for kwargs in ({"channels": 2}, {"rate": 48000}, {"width": 1}):
            self.write_wav(tone(BLOCK), **kwargs)
            with self.assertRaisesRegex(ValueError, "PCM 16-bit mono"):
                FileAudioSource(self.path)
        with self.assertRaises(OSError):
            FileAudioSource(self.path.parent / "missing.wav")

    def test_finish_preserves_unpadded_tail_and_is_idempotent(self):
        samples = tone(SAMPLE_RATE * 2 + 137).astype(np.float32) / 32768
        segmenter = SpeechSegmenter()
        before = segmenter.push(AudioChunk("one", 10.0, SAMPLE_RATE, samples))
        finals = segmenter.finish()
        self.assertEqual(len(finals), 1)
        final = finals[0]
        self.assertTrue(final.meta.is_final)
        self.assertEqual(final.meta.phrase_id, before[-1].meta.phrase_id)
        self.assertGreater(final.meta.revision, before[-1].meta.revision)
        self.assertAlmostEqual(final.meta.ended_at, 10.0 + len(samples) / SAMPLE_RATE)
        np.testing.assert_array_equal(final.samples, samples)
        self.assertEqual(segmenter.finish(), [])
        with self.assertRaises(ValueError):
            segmenter.push(AudioChunk("one", 12.0, SAMPLE_RATE, samples))

    def test_finish_measures_short_tail_without_rounding_speech_up(self):
        for length, expected in ((int(0.349 * SAMPLE_RATE), 0), (int(0.351 * SAMPLE_RATE), 1)):
            samples = tone(length).astype(np.float32) / 32768
            segmenter = SpeechSegmenter()
            segmenter.push(AudioChunk("one", 0.0, SAMPLE_RATE, samples))
            self.assertEqual(len(segmenter.finish()), expected)

    def test_file_and_direct_segmentation_match(self):
        events = self.read_source(tone(SAMPLE_RATE * 2 + 83))
        direct = SpeechSegmenter()
        expected = []
        incoming, outgoing = queue.Queue(), queue.Queue()
        for event in events:
            incoming.put(event)
            if isinstance(event, AudioChunk):
                expected.extend(direct.push(event))
        expected.extend(direct.finish())
        worker_segment(incoming, outgoing, threading.Event())
        actual = []
        while not outgoing.empty():
            event = outgoing.get_nowait()
            if not isinstance(event, AudioStreamEnded):
                actual.append(event)
        self.assertEqual(len(actual), len(expected))
        for left, right in zip(actual, expected):
            self.assertEqual(left.meta, right.meta)
            np.testing.assert_array_equal(left.samples, right.samples)

    def test_stream_change_finishes_old_phrase_without_mixing(self):
        for explicit_boundary in (False, True):
            incoming, outgoing = queue.Queue(), queue.Queue()
            first = tone(8000).astype(np.float32) / 32768
            second = -first
            incoming.put(AudioChunk("a", 0.0, SAMPLE_RATE, first))
            if explicit_boundary:
                incoming.put(AudioStreamEnded("a", "device_changed"))
            incoming.put(AudioChunk("b", 5.0, SAMPLE_RATE, second))
            incoming.put(AudioStreamEnded("b", source_finished=True))
            worker_segment(incoming, outgoing, threading.Event())
            messages = list(outgoing.queue)
            finals = [m for m in messages if not isinstance(m, AudioStreamEnded)]
            self.assertEqual([m.meta.stream_id for m in finals], ["a", "b"])
            self.assertTrue(all(m.meta.is_final for m in finals))
            np.testing.assert_array_equal(finals[0].samples, first)
            np.testing.assert_array_equal(finals[1].samples, second)
            self.assertIsInstance(messages[1], AudioStreamEnded)

    def test_eof_drains_all_stages_without_stopping_application(self):
        events = self.read_source(tone(SAMPLE_RATE + 137))
        incoming, segments, texts = queue.Queue(), queue.Queue(), queue.Queue()
        for event in events:
            incoming.put(event)
        stop = threading.Event()
        worker_segment(incoming, segments, stop)
        recognizer = SimpleNamespace(transcribe=lambda samples, partial: ("hello", "en"))
        translator = SimpleNamespace(translate_with_target=lambda text, language: ("ahoj", "cs"))
        output = []
        worker_transcribe(segments, recognizer, texts, stop)
        worker_translate(texts, translator, SimpleNamespace(update=output.append), stop)
        self.assertTrue(output[-1].meta.is_final)
        self.assertEqual(output[-1].text, "ahoj")
        self.assertFalse(stop.is_set())
        self.assertTrue(segments.empty() and texts.empty())

    def test_stop_interrupts_full_file_queue(self):
        self.write_wav(tone(SAMPLE_RATE * 3))
        source = FileAudioSource(self.path, realtime=False, queue_size=1)
        self.addCleanup(source.stop)
        source.queue.put(AudioStreamEnded("occupied"))
        entered = threading.Event()
        publish = source._publish

        def blocked(event):
            entered.set()
            return publish(event)

        with patch.object(source, "_publish", side_effect=blocked):
            source.start()
            self.assertTrue(entered.wait(2))
            source.stop()
        self.assertTrue(source.done.is_set())
        self.assertFalse(source._thread.is_alive())

    def test_stop_interrupts_realtime_pacing(self):
        self.write_wav(tone(SAMPLE_RATE * 3))
        source = FileAudioSource(self.path, realtime=True)
        self.addCleanup(source.stop)
        source.start()
        source.stop()
        self.assertTrue(source.done.is_set())

    def test_live_overflow_is_an_explicit_error(self):
        source = SystemLoopbackSource(queue_size=1)
        source.queue.put(AudioStreamEnded("occupied"))
        with self.assertRaisesRegex(RuntimeError, "overflow"):
            source._publish(AudioChunk("one", 0.0, SAMPLE_RATE, np.zeros(BLOCK)), live=True)

    def test_capture_device_change_reopens_and_closes_recorders(self):
        source = SystemLoopbackSource(queue_size=20)
        closed = []

        class Recorder:
            def __init__(self, name):
                self.name, self.reads = name, 0

            def __enter__(self):
                return self

            def __exit__(self, *args):
                closed.append(self.name)

            def record(self, numframes):
                self.reads += 1
                if self.name == "b" and self.reads == 2:
                    source._stop.set()
                return np.ones((numframes, 2), dtype=np.float32) * 0.1

        names = iter(["a", "b", "b", "b"])
        soundcard = SimpleNamespace(
            default_speaker=lambda: SimpleNamespace(name=next(names)),
            get_microphone=lambda name, **kwargs: SimpleNamespace(recorder=lambda **opts: Recorder(name)),
        )
        clock = count(step=3)
        with patch("audio_sources._import_soundcard", return_value=soundcard), patch("audio_sources.time.monotonic", side_effect=lambda: next(clock)):
            source._run()
        messages = list(source.queue.queue)
        self.assertEqual(closed, ["a", "b"])
        self.assertIsInstance(messages[0], AudioChunk)
        self.assertIsInstance(messages[1], AudioStreamEnded)
        self.assertEqual(messages[1].reason, "device_changed")
        self.assertIsInstance(messages[2], AudioChunk)
        self.assertNotEqual(messages[0].stream_id, messages[2].stream_id)

    def test_read_failure_publishes_terminal_error(self):
        self.write_wav(tone(BLOCK))
        source = FileAudioSource(self.path, realtime=False)
        self.addCleanup(source.stop)
        self.path.unlink()
        source.start()
        event = source.queue.get(timeout=2)
        self.assertTrue(source.done.wait(2))
        self.assertTrue(event.source_finished)
        self.assertEqual(event.reason, "error")
        self.assertIsNotNone(event.error)

    def test_cli_rejects_conflicting_sources(self):
        self.assertEqual(parse_args(["--audio-file", "speech.wav"]).audio_file, "speech.wav")
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
            parse_args(["--audio-file", "speech.wav", "--out-device", "headphones"])
        self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
