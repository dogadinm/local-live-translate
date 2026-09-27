"""Overload and ordering tests with controlled clocks and blocked fake models."""
import queue
import tempfile
import threading
import time
import unittest
from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from config import ProcessingSettings
from events import AudioStreamEnded, PhraseMeta, RecognitionResult, SpeechSegment, Transcript, Translation
from metrics import Metrics
from overlay import SubtitleOverlay
from pipeline import drain, worker_transcribe, worker_translate
from scheduling import PendingQueue, presentation_is_newer


def message(phrase=1, revision=1, final=False, text="hello", language="en", target="ru", ended_at=100):
    return Transcript(PhraseMeta("s", phrase, revision, ended_at - 1, ended_at, final),
                      text, language, ProcessingSettings(target_language=target))


class SchedulingTests(unittest.TestCase):
    def make_queue(self, capacity=8, clock=lambda: 100):
        dropped = []
        work = PendingQueue(capacity, 8, on_drop=lambda event, reason: dropped.append((event, reason)), clock=clock)
        return work, dropped

    def test_nonadjacent_drafts_are_replaced_and_final_takes_their_slot(self):
        work, dropped = self.make_queue()
        work.put(message())
        work.put(message(phrase=2))
        work.put(message(revision=2))
        work.put(message(revision=3, final=True))
        self.assertEqual(work.qsize(), 2)
        self.assertEqual(work.get_nowait(), message(revision=3, final=True))
        self.assertEqual(work.get_nowait().meta.phrase_id, 2)
        self.assertEqual([reason for _, reason in dropped], ["coalesced", "coalesced"])

    def test_final_cannot_be_replaced_by_a_late_draft(self):
        work, dropped = self.make_queue()
        work.put(message(revision=3, final=True))
        work.put(message(revision=99))
        self.assertTrue(work.get_nowait().meta.is_final)
        self.assertEqual(dropped[-1][1], "obsolete_revision")

    def test_overflow_prefers_discarding_drafts(self):
        work, dropped = self.make_queue(2)
        work.put(message(phrase=1, final=True))
        work.put(message(phrase=2, final=True))
        work.put(message(phrase=3))
        self.assertEqual(dropped[-1][0].meta.phrase_id, 3)
        self.assertEqual(work.qsize(), 2)
        work.put(message(phrase=4, final=True))
        self.assertEqual(dropped[-1][0].meta.phrase_id, 1)
        self.assertEqual(dropped[-1][1], "overflow")

    def test_age_pruned_on_get_without_new_audio(self):
        clock = [100]
        work, dropped = self.make_queue(clock=lambda: clock[0])
        work.put(message(final=True))
        work.put(AudioStreamEnded("s", source_finished=True))
        clock[0] = 109
        self.assertIsInstance(work.get_nowait(), AudioStreamEnded)
        self.assertEqual(dropped[-1][1], "expired")
        self.assertTrue(work.empty())

    def test_eof_survives_overload_and_seals_queue(self):
        work, dropped = self.make_queue(2)
        for phrase in range(100):
            work.put(message(phrase=phrase, final=True))
            self.assertLessEqual(work.qsize(), 2)
        work.put(AudioStreamEnded("s", source_finished=True))
        work.put(message(phrase=101))
        self.assertEqual(dropped[-1][1], "after_eof")
        self.assertTrue(work.get_nowait().meta.is_final)
        self.assertTrue(work.get_nowait().source_finished)

    def test_boundary_storm_is_bounded(self):
        work, _ = self.make_queue(2)
        for i in range(100):
            work.put(AudioStreamEnded(str(i), "device_changed"))
            self.assertLessEqual(work.qsize(), 2)
        work.put(AudioStreamEnded("end", source_finished=True))
        work.get_nowait()
        self.assertTrue(work.get_nowait().source_finished)

    def test_draining_pending_queue_does_not_hide_replaceable_backlog(self):
        work, _ = self.make_queue()
        work.put(message())
        work.put(message(phrase=2))
        first = work.get_nowait()
        self.assertEqual(drain(work, first), [first])
        work.put(message(phrase=2, revision=2, final=True))
        self.assertTrue(work.get_nowait().meta.is_final)

    def test_slow_recognition_keeps_only_latest_waiting_final(self):
        work, dropped = self.make_queue(capacity=3)
        output = queue.Queue()
        entered, release = threading.Event(), threading.Event()
        stop = threading.Event()
        calls = []

        def segment(phrase, revision=1, final=False):
            event = message(phrase, revision, final)
            return SpeechSegment(event.meta, 16000, np.array([phrase], dtype=np.float32), event.settings)

        def recognize(samples, **kwargs):
            calls.append(int(samples[0]))
            if len(calls) == 1:
                entered.set()
                release.wait(2)
            return RecognitionResult("hello", "en", "en")

        thread = threading.Thread(target=worker_transcribe,
                                  args=(work, SimpleNamespace(transcribe=recognize), output, stop))
        work.put(segment(1))
        thread.start()
        try:
            self.assertTrue(entered.wait(2))
            for revision in range(1, 50):
                work.put(segment(2, revision))
            work.put(segment(2, 50, True))
            work.put(AudioStreamEnded("s", source_finished=True))
            self.assertEqual(work.qsize(), 2)
        finally:
            release.set()
            thread.join(2)
            stop.set()
        self.assertFalse(thread.is_alive())
        self.assertEqual(calls, [1, 2])
        self.assertEqual(len(dropped), 49)

    def test_cached_draft_is_promoted_to_final_without_second_call(self):
        work, _ = self.make_queue()
        output, calls = [], []
        first = message()
        final = message(revision=2, final=True)
        with tempfile.TemporaryDirectory() as directory:
            metrics = Metrics(directory)
            try:
                metrics.enqueue("translation", first)
                work.put(first)

                def translate(text, source, target):
                    calls.append((text, source, target))
                    metrics.enqueue("translation", final)
                    work.put(final)
                    work.put(AudioStreamEnded("s", source_finished=True))
                    return "translation"

                worker_translate(work, SimpleNamespace(translate=translate), SimpleNamespace(update=output.append),
                                 threading.Event(), metrics=metrics)
                self.assertEqual(len(calls), 1)
                self.assertEqual([event.meta.is_final for event in output], [False, True])
                self.assertEqual(output[-1].meta.revision, 2)
                self.assertEqual(metrics._counts["translation_cache_hits"], 1)
                self.assertEqual(metrics._pending, {})
            finally:
                metrics.close()

    def test_cache_keys_include_both_languages_and_is_bounded(self):
        for capacity, expected in ((128, 4), (1, 5)):
            work = queue.Queue()
            for i, (text, source, target) in enumerate([
                ("a", "en", "ru"), ("a", "en", "cs"), ("a", "cs", "ru"),
                ("b", "en", "ru"), ("a", "en", "ru")]):
                work.put(message(i, final=True, text=text, language=source, target=target))
            work.put(AudioStreamEnded("s", source_finished=True))
            calls = []

            def translate(*args):
                calls.append(args)
                return "translated"

            worker_translate(work, SimpleNamespace(translate=translate), SimpleNamespace(update=lambda event: None),
                             threading.Event(), cache_size=capacity)
            self.assertEqual(len(calls), expected)

    def test_expensive_result_past_deadline_is_not_shown(self):
        clock = [100]
        work, dropped = self.make_queue(clock=lambda: clock[0])
        work.put(message(final=True))
        work.put(AudioStreamEnded("s", source_finished=True))
        output = []

        def translate(*args):
            clock[0] = 110
            return "too late"

        worker_translate(work, SimpleNamespace(translate=translate), SimpleNamespace(update=output.append), threading.Event())
        self.assertEqual(output, [])
        self.assertEqual(dropped[-1][1], "expired_after_inference")

    def test_ui_rejects_late_revisions_and_previous_phrases(self):
        overlay = SubtitleOverlay.__new__(SubtitleOverlay)
        overlay._queue = queue.Queue()
        overlay.is_current = lambda meta: True
        shown = []
        overlay.text_label = SimpleNamespace(config=lambda **kw: shown.append(kw["text"]))
        overlay.root = SimpleNamespace(after=lambda *args: None)
        for event, text in [(message(revision=2), "draft"), (message(revision=1), "old"),
                            (message(revision=3, final=True), "final"), (message(revision=99), "late draft"),
                            (message(phrase=2), "next"), (message(revision=4, final=True), "old phrase")]:
            overlay.update(Translation(event.meta, "source", text, "en", "ru"))
        overlay._poll()
        self.assertEqual(shown, ["draft", "final", "next"])

    def test_wait_timeout_and_wakeup(self):
        work, _ = self.make_queue()
        with self.assertRaises(queue.Empty):
            work.get(timeout=0.01)
        received = []
        thread = threading.Thread(target=lambda: received.append(work.get(timeout=1)))
        thread.start()
        work.put(message())
        thread.join(2)
        self.assertEqual(received, [message()])


if __name__ == "__main__":
    unittest.main(verbosity=2)
