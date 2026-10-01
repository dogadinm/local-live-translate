"""Controller, settings races and failures without model weights or a screen."""
import queue
import threading
import time
import unittest
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import numpy as np

from app import ApplicationController
from config import ProcessingSettings
from events import (ApplicationStatus, AudioChunk, AudioStreamEnded, PhraseMeta,
                    RecognitionResult, SpeechSegment, Transcript, Translation)
from ui.overlay import AUTO, SubtitleOverlay
from pipeline import worker_segment, worker_transcribe, worker_translate


class FakeSource:
    def __init__(self):
        self.queue = queue.Queue()
        self.started = threading.Event()
        self.stopped = threading.Event()

    def start(self):
        self.started.set()

    def stop(self):
        self.stopped.set()


class FakeOverlay:
    def __init__(self):
        self.messages = []
        self.statuses = []
        self.failed = threading.Event()
        self.finished = threading.Event()
        self.cleared = 0
        self.closed = False

    def clear_results(self):
        self.cleared += 1

    def set_status(self, status):
        self.statuses.append(status)
        if status.is_error:
            self.failed.set()
        if status.message.startswith("Finished"):
            self.finished.set()

    def show_transcript(self, event):
        self.messages.append(event)

    def update(self, event):
        self.messages.append(event)

    def close(self):
        self.closed = True


class ControllerTests(unittest.TestCase):
    def make_app(self, recognizer=None, translator=None, overlay=None):
        source = FakeSource()
        overlay = overlay or FakeOverlay()
        recognizer = recognizer or SimpleNamespace(transcribe=lambda *args, **kwargs: RecognitionResult("hello", "en", "en"))
        translator = translator or SimpleNamespace(translate=lambda text, src, tgt: "translated")
        app = ApplicationController(source, overlay, lambda: recognizer, lambda: translator,
                                    ProcessingSettings(source_language="en", target_language="cs"))
        self.addCleanup(app.stop)
        return app, source, overlay

    def segment(self, app):
        settings = app.get_settings()
        now = time.monotonic()
        return SpeechSegment(PhraseMeta("stream", 1, 1, now - 1, now, True, settings.version),
                             16000, np.zeros(16000, dtype=np.float32), settings)

    def enqueue(self, event):
        work = queue.Queue()
        work.put(event)
        work.put(AudioStreamEnded("stream", source_finished=True))
        return work

    def test_settings_are_immutable_and_changes_increment_only_once(self):
        app, _, overlay = self.make_app()
        old = app.get_settings()
        app.set_target_language("cs")
        self.assertIs(app.get_settings(), old)
        app.set_target_language("ru")
        app.set_source_language(None)
        self.assertEqual(app.get_settings(), ProcessingSettings(None, "ru", 2))
        self.assertEqual(old, ProcessingSettings("en", "cs", 0))
        self.assertEqual(overlay.cleared, 2)
        with self.assertRaises(FrozenInstanceError):
            old.target_language = "ru"
        with self.assertRaises(ValueError):
            app.set_target_language("invalid")
        self.assertEqual(app.get_settings().version, 2)

    def test_recognition_mid_call_change_keeps_snapshot_and_discards_result(self):
        app, _, overlay = self.make_app()
        segment = self.segment(app)

        def recognize(audio, partial, *, settings):
            app.set_source_language("cs")
            app.set_target_language("ru")
            self.assertIs(settings, segment.settings)
            self.assertEqual((settings.source_language, settings.target_language), ("en", "cs"))
            return RecognitionResult("hello", "en", "en")

        out = queue.Queue()
        worker_transcribe(self.enqueue(segment), SimpleNamespace(transcribe=recognize), out,
                          app._stop, is_current=app.is_current, on_transcript=app.on_transcript)
        self.assertIsInstance(out.get_nowait(), AudioStreamEnded)
        self.assertTrue(out.empty())
        self.assertEqual(overlay.messages, [])

    def test_translation_mid_call_change_discards_old_result(self):
        app, _, overlay = self.make_app()
        segment = self.segment(app)
        transcript = Transcript(segment.meta, "hello", "en", segment.settings, "en")

        def translate(text, src, target):
            app.set_target_language("ru")
            self.assertEqual(target, "cs")
            return "ahoj"

        worker_translate(self.enqueue(transcript), SimpleNamespace(translate=translate), app,
                         app._stop, is_current=app.is_current)
        self.assertEqual(overlay.messages, [])

    def test_obsolete_queued_jobs_do_not_call_models(self):
        app, _, _ = self.make_app()
        segment = self.segment(app)
        app.set_target_language("ru")

        def forbidden(*args, **kwargs):
            self.fail("obsolete job reached model")

        worker_transcribe(self.enqueue(segment), SimpleNamespace(transcribe=forbidden), queue.Queue(),
                          app._stop, is_current=app.is_current)
        transcript = Transcript(segment.meta, "hello", "en", segment.settings)
        worker_translate(self.enqueue(transcript), SimpleNamespace(translate=forbidden), app,
                         app._stop, is_current=app.is_current)

    def test_segmentation_stamps_snapshot_on_job(self):
        app, _, _ = self.make_app()
        app.set_target_language("ru")
        samples = (0.2 * np.sin(2 * np.pi * 180 * np.arange(16000) / 16000)).astype(np.float32)
        out = queue.Queue()
        worker_segment(self.enqueue(AudioChunk("stream", 10, 16000, samples)), out,
                       app._stop, get_settings=app.get_settings)
        job = out.get_nowait()
        self.assertIs(job.settings, app.get_settings())
        self.assertEqual(job.meta.settings_version, 1)

    def test_ui_rechecks_version_for_already_queued_results(self):
        overlay = SubtitleOverlay.__new__(SubtitleOverlay)
        overlay._queue = queue.Queue()
        rendered, detected = [], []
        overlay.text_label = SimpleNamespace(config=lambda **kw: rendered.append(kw))
        overlay.status = SimpleNamespace(config=lambda **kw: detected.append(kw))
        overlay.application_status = SimpleNamespace(config=lambda **kw: None)
        overlay.source_picker = SimpleNamespace(get=lambda: AUTO)
        overlay.root = SimpleNamespace(after=lambda *args: None)
        app, _, _ = self.make_app(overlay=overlay)
        old = self.segment(app)
        app.update(Translation(old.meta, "hello", "old", "en", "cs"))
        app.on_transcript(Transcript(old.meta, "hello", "en", old.settings, "en"))
        app.set_target_language("ru")
        new = self.segment(app)
        app.update(Translation(new.meta, "hello", "new", "en", "ru"))
        overlay._poll()
        self.assertEqual([item["text"] for item in rendered], ["", "new"])
        self.assertEqual(detected, [{"text": ""}])

    def test_loading_failure_is_visible_and_stops_source(self):
        app, source, overlay = self.make_app()

        def fail():
            raise RuntimeError("missing weights")

        app._recognizer_factory = fail
        app.start()
        self.assertTrue(overlay.failed.wait(2))
        self.assertTrue(source.stopped.wait(2))
        self.assertFalse(source.started.is_set())
        self.assertIn("missing weights", overlay.statuses[-1].message)

    def test_all_worker_exceptions_are_reported(self):
        for stage in ("Segmentation", "Recognition", "Translation"):
            app, source, overlay = self.make_app()

            def fail():
                raise RuntimeError("engine failed")

            app._guard(stage, fail, (), {})
            self.assertTrue(app._stop.is_set())
            self.assertTrue(source.stopped.is_set())
            self.assertEqual(overlay.statuses[-1], ApplicationStatus(f"{stage}: engine failed", True))

    def test_source_errors_distinguish_retry_from_fatal(self):
        app, source, overlay = self.make_app()
        app._on_boundary(AudioStreamEnded("stream", "error", error="device lost"))
        self.assertFalse(app._stop.is_set())
        self.assertIn("Reconnecting", overlay.statuses[-1].message)
        app._on_stream("next")
        self.assertFalse(overlay.statuses[-1].is_error)
        app._on_boundary(AudioStreamEnded("next", "error", True, "file unreadable"))
        self.assertTrue(app._stop.is_set())
        self.assertTrue(source.stopped.is_set())
        app._on_finished()
        self.assertTrue(overlay.statuses[-1].is_error)

    def test_stop_during_loading_prevents_source_start(self):
        app, source, _ = self.make_app()
        entered, release = threading.Event(), threading.Event()
        self.addCleanup(release.set)

        def load():
            entered.set()
            release.wait(3)
            return SimpleNamespace()

        app._recognizer_factory = load
        app.start()
        self.assertTrue(entered.wait(2))
        app.stop()
        release.set()
        for thread in app._threads:
            thread.join(timeout=2)
        self.assertFalse(source.started.is_set())
        self.assertTrue(all(not thread.is_alive() for thread in app._threads))
        app.stop()  # idempotent

    def test_controller_runs_eof_pipeline_and_shuts_down(self):
        app, source, overlay = self.make_app()
        samples = (0.2 * np.sin(2 * np.pi * 180 * np.arange(16000) / 16000)).astype(np.float32)
        source.queue.put(AudioChunk("stream", time.monotonic() - 1, 16000, samples))
        source.queue.put(AudioStreamEnded("stream", source_finished=True))
        app.start()
        self.assertTrue(overlay.finished.wait(2))
        translations = [message for message in overlay.messages if isinstance(message, Translation)]
        self.assertTrue(translations[-1].meta.is_final)
        self.assertEqual(translations[-1].target_language, "cs")
        app.stop()
        self.assertTrue(all(not thread.is_alive() for thread in app._threads))
        self.assertFalse(app.is_current(translations[-1].meta))

    def test_whisper_reports_audio_language_separately_from_english_text(self):
        from transcriber import Transcriber
        engine = Transcriber.__new__(Transcriber)
        engine._can_translate = True
        calls = []

        def infer(audio, **kwargs):
            calls.append(kwargs)
            return iter([SimpleNamespace(text="hello")]), SimpleNamespace(language="cs")

        engine.model = SimpleNamespace(transcribe=infer)
        result = engine.transcribe(np.zeros(10), settings=ProcessingSettings(target_language="en"))
        self.assertEqual(result, RecognitionResult("hello", "en", "cs"))
        self.assertEqual(calls[-1]["task"], "translate")
        engine._can_translate = False
        engine.transcribe(np.zeros(10), settings=ProcessingSettings("cs", "en"))
        self.assertEqual(calls[-1]["task"], "transcribe")
        self.assertEqual(calls[-1]["language"], "cs")

    def test_actual_worker_failure_stops_pipeline_and_reports_stage(self):
        for stage in ("Recognition", "Translation"):
            with self.subTest(stage=stage):
                def fail(*args, **kwargs):
                    raise RuntimeError("inference failed")

                kwargs = ({"recognizer": SimpleNamespace(transcribe=fail)} if stage == "Recognition"
                          else {"translator": SimpleNamespace(translate=fail)})
                app, source, overlay = self.make_app(**kwargs)
                app.start()
                self.assertTrue(source.started.wait(2))
                samples = (0.2 * np.sin(2 * np.pi * 180 * np.arange(16000) / 16000)).astype(np.float32)
                source.queue.put(AudioChunk("stream", time.monotonic() - 1, 16000, samples))
                source.queue.put(AudioStreamEnded("stream", source_finished=True))
                self.assertTrue(overlay.failed.wait(2))
                app.stop()
                self.assertIn(stage, overlay.statuses[-1].message)
                self.assertTrue(all(not thread.is_alive() for thread in app._threads))

    def test_shutdown_discards_result_of_inflight_recognition(self):
        app, _, overlay = self.make_app()
        segment = self.segment(app)

        def recognize(*args, **kwargs):
            app.stop()
            return RecognitionResult("late", "en", "en")

        output = queue.Queue()
        worker_transcribe(self.enqueue(segment), SimpleNamespace(transcribe=recognize), output,
                          app._stop, is_current=app.is_current, on_transcript=app.on_transcript)
        self.assertTrue(output.empty())
        self.assertEqual(overlay.messages, [])

    def test_old_stream_results_are_rejected_after_reconnect(self):
        app, _, overlay = self.make_app()
        old = self.segment(app)
        app._on_stream("new-stream")
        app.update(Translation(old.meta, "hello", "old", "en", "cs"))
        self.assertEqual(overlay.messages, [])

    def test_run_cleans_up_when_window_loop_fails(self):
        app, source, overlay = self.make_app()

        def fail():
            raise RuntimeError("window failed")

        overlay.run = fail
        with self.assertRaisesRegex(RuntimeError, "window failed"):
            app.run()
        self.assertTrue(source.stopped.is_set())
        self.assertTrue(overlay.closed)


if __name__ == "__main__":
    unittest.main(verbosity=2)
