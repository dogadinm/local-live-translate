import queue
import time
import tkinter as tk
from tkinter import ttk
from typing import Callable

from langs import display, picker_items
from events import ApplicationStatus, ProcessingFinished, Transcript, Translation
from scheduling import presentation_is_newer

BG = "#000000"
FG_SRC = "#9aa0a6"
FG_DST = "#ffffff"
FG_DIM = "#7a7f85"
FONT = "Segoe UI"
AUTO = "Auto"


class SubtitleOverlay:
    """Borderless always-on-top subtitle bar with source and target pickers."""
    metrics = None
    max_display_lag = None
    _last_translation = None
    _last_transcript = None

    def __init__(
        self,
        source: str | None = None,
        target: str = "ru",
        on_source_change: Callable[[str | None], None] | None = None,
        on_target_change: Callable[[str], None] | None = None,
    ):
        self._queue: queue.Queue[Translation | Transcript | ApplicationStatus | ProcessingFinished | None] = queue.Queue()
        self.is_current = lambda meta: True
        self.on_close: Callable[[], None] | None = None
        self._closed = False
        self.on_source_change = on_source_change
        self.on_target_change = on_target_change
        self._items = picker_items()
        self._by_label = {label: iso for iso, label in self._items}
        self._drag = (0, 0)
        self.root = tk.Tk()
        self._build(source, target)

    def _build(self, source: str | None, target: str):
        root = self.root
        root.title("live-translate")
        root.overrideredirect(True)
        root.attributes("-topmost", True)
        root.attributes("-alpha", 0.9)
        root.configure(bg=BG)

        sw = root.winfo_screenwidth()
        sh = root.winfo_screenheight()
        height = 140
        root.geometry(f"{sw}x{height}+0+{sh - height - 48}")

        bar = tk.Frame(root, bg=BG)
        bar.pack(fill="x", padx=16, pady=(6, 0))
        labels = [label for _, label in self._items]

        tk.Label(bar, text="Source:", font=(FONT, 10), fg=FG_DIM, bg=BG).pack(side="left")
        self.source_picker = ttk.Combobox(
            bar, state="readonly", width=13, font=(FONT, 10), values=[AUTO] + labels
        )
        self.source_picker.set(display(source) if source else AUTO)
        self.source_picker.pack(side="left", padx=(6, 18))
        self.source_picker.bind("<<ComboboxSelected>>", self._source_changed)

        tk.Label(bar, text="Translate to:", font=(FONT, 10), fg=FG_DIM, bg=BG).pack(side="left")
        self.target_picker = ttk.Combobox(
            bar, state="readonly", width=13, font=(FONT, 10), values=labels
        )
        self.target_picker.set(display(target))
        self.target_picker.pack(side="left", padx=(6, 0))
        self.target_picker.bind("<<ComboboxSelected>>", self._target_changed)

        tk.Button(
            bar, text="✕", font=(FONT, 10), fg=FG_DIM, bg=BG,
            relief="flat", bd=0, activebackground=BG, activeforeground="#ff6b6b",
            command=self.close,
        ).pack(side="right")

        self.status = tk.Label(bar, text="", font=(FONT, 10), fg=FG_DIM, bg=BG)
        self.status.pack(side="right", padx=(0, 16))
        self.application_status = tk.Label(root, text="", font=(FONT, 9), fg=FG_DIM, bg=BG)
        self.application_status.pack(fill="x", padx=16)

        self.text_label = tk.Label(
            root, text="Waiting for speech…", font=(FONT, 21, "bold"), fg=FG_DST, bg=BG,
            wraplength=sw - 60, justify="center",
        )
        self.text_label.pack(fill="both", expand=True, padx=20, pady=(8, 10))

        for widget in (root, bar, self.text_label):
            widget.bind("<Button-1>", self._drag_start)
            widget.bind("<B1-Motion>", self._drag_move)
        root.bind("<Escape>", lambda _: self.close())
        root.protocol("WM_DELETE_WINDOW", self.close)

    def _source_changed(self, _event):
        label = self.source_picker.get()
        iso = None if label == AUTO else self._by_label.get(label)
        self.status.config(text="" if iso else "detecting…")
        if self.on_source_change:
            self.on_source_change(iso)
        self.root.focus_set()

    def _target_changed(self, _event):
        iso = self._by_label.get(self.target_picker.get())
        if iso and self.on_target_change:
            self.on_target_change(iso)
        self.root.focus_set()

    def _drag_start(self, event):
        self._drag = (event.x_root - self.root.winfo_x(), event.y_root - self.root.winfo_y())

    def _drag_move(self, event):
        dx, dy = self._drag
        self.root.geometry(f"+{event.x_root - dx}+{event.y_root - dy}")

    # --- thread-safe API ---------------------------------------------------

    def update(self, translation: Translation):
        """Drafts are shown dimmed so it is obvious they may still be rewritten."""
        self._queue.put(translation)

    def show_transcript(self, transcript: Transcript):
        self._queue.put(transcript)

    def clear_results(self):
        self._queue.put(None)

    def set_status(self, status: ApplicationStatus):
        self._queue.put(status)

    def finish_metrics(self):
        self._queue.put(ProcessingFinished())

    def close(self):
        """Main-thread shutdown; model calls may still be finishing in the background."""
        if self._closed:
            return
        self._closed = True
        try:
            if self.on_close is not None:
                self.on_close()
        finally:
            self.root.destroy()

    # --- main loop ---------------------------------------------------------

    def run(self):
        self._poll()
        self.root.mainloop()

    def _poll(self):
        try:
            while True:
                event = self._queue.get_nowait()
                if isinstance(event, (Translation, Transcript)) and not self.is_current(event.meta):
                    continue
                if isinstance(event, ProcessingFinished):
                    if self.metrics:
                        self.metrics.finish()
                        print(f"[metrics] saved to {self.metrics.directory / 'summary.json'}")
                elif event is None:
                    self._last_translation = None
                    self._last_transcript = None
                    self.text_label.config(text="")
                    self.status.config(text="")
                elif isinstance(event, ApplicationStatus):
                    self.application_status.config(text=event.message, fg="#ff6b6b" if event.is_error else FG_DIM)
                elif isinstance(event, Translation):
                    if not presentation_is_newer(event.meta, self._last_translation):
                        if self.metrics:
                            self.metrics.count("display_obsolete_revision")
                        continue
                    if self.max_display_lag is not None and time.monotonic() - event.meta.ended_at > self.max_display_lag:
                        if self.metrics:
                            self.metrics.discard("display", event, "expired")
                        continue
                    self._last_translation = event.meta
                    self.text_label.config(
                        text=event.text, fg=FG_DST if event.meta.is_final else FG_SRC
                    )
                    if self.metrics:
                        self.metrics.displayed(event)
                elif (event.detected_language and self.source_picker.get() == AUTO
                      and presentation_is_newer(event.meta, self._last_transcript)):
                    self._last_transcript = event.meta
                    self.status.config(text=f"detected: {display(event.detected_language)}")
        except queue.Empty:
            pass
        self.root.after(80, self._poll)
