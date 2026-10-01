import json
import queue
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from app import ApplicationController
from compare_metrics import compare
from config import ProcessingSettings
from events import AudioChunk, AudioStreamEnded, PhraseMeta, RecognitionResult, Translation
from main import parse_args
from metrics import Metrics, distribution, process_rss_bytes
from ui.overlay import AUTO, SubtitleOverlay


class MetricsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)

    def recorder(self, name="run", metadata=None):
        recorder = Metrics(self.directory / name, metadata)
        self.addCleanup(recorder.close)
        return recorder

    def test_statistics_use_nearest_rank_p95_and_empty_is_missing(self):
        self.assertEqual(distribution([1, 2, 3, 4])["median"], 2.5)
        self.assertEqual(distribution(list(range(1, 101)))["p95"], 95)
        self.assertIsNone(distribution([])["p95"])

    def test_timing_waiting_display_and_final_counts(self):
        clock = [10.0]
        with patch("metrics.time.monotonic", side_effect=lambda: clock[0]):
            recorder = self.recorder()
            event = Translation(PhraseMeta("s", 1, 1, 9, 10, True), "hello", "text", "en", "ru")
            recorder.enqueue("recognition", event)
            clock[0] = 11
            with recorder.measure("recognition", event):
                clock[0] = 13
            recorder.displayed(event)
            recorder.finish()
            report = json.loads((recorder.directory / "summary.json").read_text())
            self.assertEqual(report["counts"]["recognition_calls"], 1)
            self.assertEqual(report["stats"]["recognition_wait_s"]["median"], 1)
            self.assertEqual(report["stats"]["recognition_s"]["median"], 2)
            self.assertEqual(report["stats"]["display_final_lag_s"]["p95"], 3)
            self.assertEqual(report["finals_not_displayed"], 0)
            self.assertIsNone(report["processing_s"])

    def test_existing_run_cannot_be_overwritten(self):
        recorder = self.recorder()
        with self.assertRaises(FileExistsError):
            Metrics(recorder.directory)

    def test_failure_and_interrupted_reports_are_not_complete(self):
        recorder = self.recorder()
        with self.assertRaises(RuntimeError):
            with recorder.measure("recognition"):
                raise RuntimeError("inference failed")
        recorder.close()
        report = json.loads((recorder.directory / "summary.json").read_text())
        self.assertEqual(report["outcome"], "interrupted")
        self.assertEqual(report["counts"]["recognition_errors"], 1)
        recorder.record("late")  # callbacks after shutdown are harmless

    def test_comparison_flags_different_inputs_and_handles_zero(self):
        before = self.recorder("before", {"audio_sha256": "a"})
        after = self.recorder("after", {"audio_sha256": "b"})
        before.finish()
        after.count("recognition_calls")
        after.finish()
        report = compare(before.directory, after.directory)
        self.assertIn("different audio_sha256", report)
        self.assertIn("Whisper calls", report)
        self.assertIn("n/a", report)

    def test_cli_accepts_requested_metrics_command(self):
        args = parse_args(["--audio-file", "benchmark.wav", "--src", "en", "--tgt", "ru",
                           "--metrics", "runs/before"])
        self.assertEqual(args.metrics, "runs/before")

    def test_controller_saves_completion_after_ui_display(self):
        recorder = self.recorder()
        overlay = SubtitleOverlay.__new__(SubtitleOverlay)
        overlay._queue = queue.Queue()
        overlay.text_label = SimpleNamespace(config=lambda **kw: None)
        overlay.status = SimpleNamespace(config=lambda **kw: None)
        overlay.application_status = SimpleNamespace(config=lambda **kw: None)
        overlay.source_picker = SimpleNamespace(get=lambda: AUTO)
        overlay.root = SimpleNamespace(after=lambda *args: None)
        finished = threading.Event()
        original_finish = overlay.finish_metrics

        def finish():
            original_finish()
            finished.set()

        overlay.finish_metrics = finish
        source_queue = queue.Queue()
        samples = (0.2 * np.sin(2 * np.pi * 180 * np.arange(16000) / 16000)).astype(np.float32)
        origin = time.monotonic() - 1
        source_queue.put(AudioChunk("stream", origin, 16000, samples))
        source_queue.put(AudioStreamEnded("stream", source_finished=True))
        source = SimpleNamespace(queue=source_queue, start=lambda: None, stop=lambda: None)
        app = ApplicationController(source, overlay,
            lambda: SimpleNamespace(transcribe=lambda *args, **kwargs: RecognitionResult("hello", "en", "en")),
            lambda: SimpleNamespace(translate=lambda *args: "translation"),
            ProcessingSettings("en", "ru"), metrics=recorder)
        self.addCleanup(app.stop)
        app.start()
        self.assertTrue(finished.wait(2))
        self.assertFalse((recorder.directory / "summary.json").exists())
        overlay._poll()
        app.stop()
        report = json.loads((recorder.directory / "summary.json").read_text())
        self.assertEqual(report["outcome"], "completed")
        self.assertEqual(report["counts"]["displayed_final"], 1)
        self.assertEqual(report["finals_not_displayed"], 0)
        events = [json.loads(line) for line in (recorder.directory / "events.jsonl").read_text().splitlines()]
        self.assertTrue(any(event.get("text") == "translation" for event in events))

    def test_process_memory_sample(self):
        import os
        if os.name == "nt":
            self.assertGreater(process_rss_bytes(), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
