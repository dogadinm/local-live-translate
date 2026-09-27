"""Opt-in benchmark recording. JSONL contains timings and final subtitle text."""
import ctypes
import hashlib
import json
import math
import os
import platform
import threading
import time
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def distribution(values):
    values = sorted(values)
    if not values:
        return {"count": 0, "median": None, "p95": None, "max": None}
    middle = len(values) // 2
    median = values[middle] if len(values) % 2 else (values[middle - 1] + values[middle]) / 2
    return {"count": len(values), "median": median,
            "p95": values[max(0, math.ceil(len(values) * .95) - 1)], "max": values[-1]}


def process_rss_bytes():
    if os.name == "nt":
        from ctypes import wintypes

        class MemoryCounters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in (
                    "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                    "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage",
                    "PagefileUsage", "PeakPagefileUsage")]

        handle = ctypes.windll.kernel32.GetCurrentProcess
        handle.restype = wintypes.HANDLE
        read = ctypes.windll.psapi.GetProcessMemoryInfo
        read.argtypes = [wintypes.HANDLE, ctypes.POINTER(MemoryCounters), wintypes.DWORD]
        read.restype = wintypes.BOOL
        counters = MemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        if read(handle(), ctypes.byref(counters), counters.cb):
            return counters.WorkingSetSize
    return None


def run_metadata(args):
    packages = {}
    for name in ("faster-whisper", "ctranslate2", "numpy"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    fingerprint = None
    if args.audio_file:
        digest = hashlib.sha256()
        with open(args.audio_file, "rb") as audio:
            for block in iter(lambda: audio.read(1024 * 1024), b""):
                digest.update(block)
        fingerprint = digest.hexdigest()
    return {"whisper": args.whisper, "source_language": args.src, "target_language": args.tgt,
            "audio_file": args.audio_file, "audio_sha256": fingerprint,
            "out_device": args.out_device, "python": platform.python_version(),
            "platform": platform.platform(), "packages": packages,
            "warmup": "none; first inference included", "realtime": True}


class Metrics:
    def __init__(self, directory, metadata=None):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        # Never overwrite an earlier measurement.
        self._file = (self.directory / "events.jsonl").open("x", encoding="utf-8", buffering=1)
        self._lock = threading.RLock()
        self._closed = False
        self._started = time.monotonic()
        self._processing_started = None
        self._processing_ended = None
        self._outcome = "running"
        self._metadata = metadata or {}
        self._counts = Counter()
        self._samples = defaultdict(list)
        self._pending = {}
        self._queues = {}
        self._first_shown = set()
        self._finals = set()
        self._shown_finals = set()
        self._sampler_stop = threading.Event()
        self._sampler = None
        self.record("run", metadata=self._metadata)

    def record(self, kind, **fields):
        with self._lock:
            if not self._closed:
                self._file.write(json.dumps({"event": kind, "elapsed_s": time.monotonic() - self._started,
                                             **fields}, ensure_ascii=False) + "\n")

    def add_metadata(self, **fields):
        with self._lock:
            self._metadata.update(fields)
            self.record("metadata", **fields)

    def count(self, name):
        with self._lock:
            if not self._closed:
                self._counts[name] += 1

    @staticmethod
    def key(meta):
        return (meta.stream_id, meta.phrase_id, meta.revision, meta.settings_version)

    def enqueue(self, stage, event):
        with self._lock:
            if self._closed:
                return
            self._pending[(stage, self.key(event.meta))] = time.monotonic()
            if stage == "recognition":
                self._counts["segments_final" if event.meta.is_final else "segments_draft"] += 1
                if event.meta.is_final:
                    self._finals.add(self.key(event.meta))

    def discard(self, stage, event, reason):
        with self._lock:
            self._pending.pop((stage, self.key(event.meta)), None)
            self.count(f"{stage}_{reason}")

    @contextmanager
    def measure(self, stage, event=None):
        started = time.monotonic()
        self.count(stage + "_calls")
        with self._lock:
            queued = self._pending.pop((stage, self.key(event.meta)), None) if event else None
            if queued is not None:
                self._samples[stage + "_wait_s"].append(started - queued)
        outcome = "ok"
        try:
            yield
        except Exception:
            outcome = "error"
            self.count(stage + "_errors")
            raise
        finally:
            duration = time.monotonic() - started
            with self._lock:
                if not self._closed:
                    self._samples[stage + "_s"].append(duration)
                    self.record("call", stage=stage, duration_s=duration, outcome=outcome,
                                meta=asdict(event.meta) if event else None)

    def processing_start(self, queues):
        with self._lock:
            self._processing_started = time.monotonic()
            self._queues = queues
        self._sampler = threading.Thread(target=self._sample_resources, daemon=True)
        self._sampler.start()

    def _sample_resources(self):
        previous_time, previous_cpu = time.monotonic(), time.process_time()
        while not self._sampler_stop.is_set():
            now, cpu = time.monotonic(), time.process_time()
            rss = process_rss_bytes()
            cpu_percent = 100 * (cpu - previous_cpu) / max(now - previous_time, 1e-9)
            previous_time, previous_cpu = now, cpu
            with self._lock:
                if self._closed or self._processing_ended is not None:
                    return
                if now - self._processing_started >= 0.1:
                    self._samples["cpu_percent_one_core"].append(cpu_percent)
                if rss is not None:
                    self._samples["rss_mb"].append(rss / 1024 ** 2)
                sizes = {name: work.qsize() for name, work in self._queues.items()}
                for name, size in sizes.items():
                    self._samples[name + "_queue_size"].append(size)
                self.record("resources", cpu_percent_one_core=cpu_percent, rss_bytes=rss, queues=sizes)
            self._sampler_stop.wait(0.5)

    def displayed(self, translation):
        now = time.monotonic()
        meta = translation.meta
        with self._lock:
            if self._closed:
                return
            kind = "final" if meta.is_final else "draft"
            self._counts["displayed_" + kind] += 1
            self._samples["display_" + kind + "_lag_s"].append(now - meta.ended_at)
            phrase = (meta.stream_id, meta.phrase_id, meta.settings_version)
            if phrase not in self._first_shown:
                self._first_shown.add(phrase)
                self._samples["first_display_since_phrase_start_s"].append(now - meta.started_at)
            if meta.is_final:
                self._shown_finals.add(self.key(meta))
            self.record("display", meta=asdict(meta), lag_s=now - meta.ended_at,
                        source_text=translation.source_text if meta.is_final else None,
                        text=translation.text if meta.is_final else None,
                        source_language=translation.source_language, target_language=translation.target_language)

    def finish(self, outcome="completed"):
        with self._lock:
            if self._closed:
                return
            if self._processing_ended is None:
                self._processing_ended = time.monotonic()
                self._outcome = outcome
            self._sampler_stop.set()
            self._write_summary()

    def _write_summary(self):
        summary = {"schema_version": 1, "outcome": self._outcome, "metadata": self._metadata,
                   "processing_s": (self._processing_ended - self._processing_started
                                    if self._processing_started is not None else None),
                   "counts": dict(self._counts),
                   "finals_not_displayed": len(self._finals - self._shown_finals),
                   "stats": {name: distribution(values) for name, values in self._samples.items()},
                   "notes": ["No automatic warmup; first inference included.",
                             "Display time is Tk label update, not physical screen presentation.",
                             "CPU percent uses one core as 100%; may exceed 100%.",
                             "RAM and queue sizes sampled every 0.5s; brief peaks may be missed.",
                             "GPU utilization, power and VRAM are not collected.",
                             "Finals not displayed includes empty recognition, obsolete settings and interruption; it is not a quality score."]}
        temporary = self.directory / "summary.tmp"
        temporary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self.directory / "summary.json")

    def close(self, outcome="interrupted"):
        self._sampler_stop.set()
        if self._sampler is not None:
            self._sampler.join(timeout=1)
        with self._lock:
            if self._closed:
                return
            self.finish(outcome)
            self._closed = True
            self._file.close()
