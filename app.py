"""Application lifecycle, settings and presentation policy."""
import threading
import time
from contextlib import nullcontext
from dataclasses import asdict, replace
from typing import Callable

from audio_sources import AudioSource
from config import ProcessingSettings, QueuePolicy
from engines import Recognizer, TextTranslator
from events import ApplicationStatus, AudioStreamEnded, PhraseMeta, Transcript, Translation
from langs import flores
from pipeline import worker_segment, worker_transcribe, worker_translate
from scheduling import PendingQueue


class ApplicationController:
    def __init__(self, source: AudioSource, overlay,
                 recognizer_factory: Callable[[], Recognizer],
                 translator_factory: Callable[[], TextTranslator],
                 settings: ProcessingSettings = ProcessingSettings(), metrics=None,
                 queue_policy: QueuePolicy = QueuePolicy()):
        self.queue_policy = queue_policy
        self.metrics = metrics
        self.source = source
        self.overlay = overlay
        self._recognizer_factory = recognizer_factory
        self._translator_factory = translator_factory
        self._settings = settings
        self._settings_lock = threading.Lock()
        self._lifecycle_lock = threading.Lock()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._started = False
        self._active_stream: str | None = None
        overlay.on_source_change = self.set_source_language
        overlay.on_target_change = self.set_target_language
        overlay.on_close = self.stop
        overlay.is_current = self.is_current
        overlay.metrics = metrics
        overlay.max_display_lag = queue_policy.max_lag_s
        if metrics:
            metrics.add_metadata(queue_policy=asdict(queue_policy), scheduler="bounded-v1")

    def get_settings(self) -> ProcessingSettings:
        with self._settings_lock:
            return self._settings

    def is_current(self, meta: PhraseMeta) -> bool:
        with self._settings_lock:
            return (not self._stop.is_set()
                    and meta.settings_version == self._settings.version
                    and (self._active_stream is None or meta.stream_id == self._active_stream))

    def _change_settings(self, **changes) -> None:
        with self._settings_lock:
            if self._stop.is_set() or all(getattr(self._settings, key) == value for key, value in changes.items()):
                return
            self._settings = replace(self._settings, **changes, version=self._settings.version + 1)
            if self.metrics:
                self.metrics.count("settings_changes")
            self.overlay.clear_results()

    def set_source_language(self, iso: str | None) -> None:
        if iso is not None and flores(iso) is None:
            raise ValueError(f"Unsupported source language: {iso}")
        self._change_settings(source_language=iso)

    def set_target_language(self, iso: str) -> None:
        if flores(iso) is None:
            raise ValueError(f"Unsupported target language: {iso}")
        self._change_settings(target_language=iso)

    def _status(self, message: str, error: bool = False) -> None:
        if error or not self._stop.is_set():
            self.overlay.set_status(ApplicationStatus(message, error))

    def _fail(self, stage: str, exc: Exception) -> None:
        if not self._stop.is_set():
            self._stop.set()
            message = f"{stage}: {exc}"
            print(f"[error] {message}")
            self._status(message, True)
            if self.metrics:
                self.metrics.record("error", stage=stage, message=str(exc))
                self.metrics.finish("error")
            self.overlay.clear_results()
            self.source.stop()

    def _guard(self, stage, target, args, kwargs):
        try:
            target(*args, **kwargs)
        except Exception as exc:
            self._fail(stage, exc)

    def _spawn(self, stage, target, *args, **kwargs):
        # Called with the lifecycle lock held; stop cannot miss a new worker.
        thread = threading.Thread(target=self._guard, args=(stage, target, args, kwargs),
                                  name=stage, daemon=True)
        self._threads.append(thread)
        thread.start()

    def start(self) -> None:
        with self._lifecycle_lock:
            if self._started or self._stop.is_set():
                raise RuntimeError("Application controller can only be started once")
            self._started = True
            self._spawn("Loading models", self._load_and_start)

    def _load_and_start(self):
        self._status("Loading speech recognition…")
        with self.metrics.measure("load_recognition") if self.metrics else nullcontext():
            recognizer = self._recognizer_factory()
        if self._stop.is_set():
            return
        self._status("Loading translation…")
        with self.metrics.measure("load_translation") if self.metrics else nullcontext():
            translator = self._translator_factory()
        if self._stop.is_set():
            return
        def pending(stage):
            return PendingQueue(self.queue_policy.max_pending, self.queue_policy.max_lag_s,
                                on_drop=lambda event, reason: self._dropped(stage, event, reason))

        segments, transcripts = pending("recognition"), pending("translation")
        with self._lifecycle_lock:
            if self._stop.is_set():
                return
            self._status("Waiting for audio…")
            if self.metrics:
                self.metrics.processing_start({"audio": self.source.queue, "recognition": segments,
                                               "translation": transcripts})
            self._spawn("Segmentation", worker_segment, self.source.queue, segments, self._stop,
                        get_settings=self.get_settings, on_boundary=self._on_boundary,
                        on_stream=self._on_stream, metrics=self.metrics)
            self._spawn("Recognition", worker_transcribe, segments, recognizer, transcripts, self._stop,
                        is_current=self.is_current, on_transcript=self.on_transcript, metrics=self.metrics)
            self._spawn("Translation", worker_translate, transcripts, translator, self, self._stop,
                        is_current=self.is_current, on_finished=self._on_finished, metrics=self.metrics,
                        cache_size=self.queue_policy.translation_cache_size)
            if not self._stop.is_set():
                try:
                    self.source.start()
                except Exception as exc:
                    self._fail("Audio source", exc)

    def _on_stream(self, stream_id: str):
        with self._settings_lock:
            self._active_stream = stream_id
            self.overlay.clear_results()
        self._status("Listening…")

    def _dropped(self, stage, event, reason):
        if isinstance(event, AudioStreamEnded):
            if self.metrics:
                self.metrics.count(stage + "_boundaries_coalesced")
            return
        if self.metrics:
            self.metrics.discard(stage, event, reason)
        if event.meta.is_final and reason in ("overflow", "expired", "expired_after_inference"):
            print(f"[overload] {stage}: skipped phrase {event.meta.phrase_id} ({reason})")
            self._status("Overloaded — skipped an old phrase to catch up")

    def _on_boundary(self, event: AudioStreamEnded):
        if self._stop.is_set():
            return
        if event.error:
            if event.source_finished:
                self._fail("Audio source", RuntimeError(event.error))
            else:
                self._status(f"Audio source: {event.error}. Reconnecting…", True)

    def _on_finished(self):
        if not self._stop.is_set():
            self._status("Finished — all phrases processed")
            if self.metrics:
                self.overlay.finish_metrics()

    def on_transcript(self, transcript: Transcript):
        if self.is_current(transcript.meta):
            self.overlay.show_transcript(transcript)

    def update(self, translation: Translation):
        if self.is_current(translation.meta):
            self.overlay.update(translation)

    def stop(self) -> None:
        self._stop.set()
        with self._lifecycle_lock:
            self.source.stop()
            threads = list(self._threads)
        deadline = time.monotonic() + 0.6
        for thread in threads:
            if thread is not threading.current_thread():
                thread.join(timeout=max(0.0, deadline - time.monotonic()))
        if self.metrics:
            self.metrics.close()

    def run(self) -> None:
        try:
            self.start()
            self.overlay.run()
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()
            self.overlay.close()
