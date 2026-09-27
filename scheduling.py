"""Bounded pending work for live subtitles, with explicit loss accounting."""
import queue
import threading
import time
from collections import deque

from events import AudioStreamEnded


def phrase_key(meta):
    return meta.stream_id, meta.phrase_id, meta.settings_version


def supersedes(new, old):
    if old.is_final and not new.is_final:
        return False
    if new.is_final and not old.is_final:
        return True
    return new.revision > old.revision


def presentation_is_newer(new, old):
    if old is None or (new.stream_id, new.settings_version) != (old.stream_id, old.settings_version):
        return True
    if new.phrase_id != old.phrase_id:
        return new.phrase_id > old.phrase_id
    return supersedes(new, old)


class PendingQueue:
    """Nonblocking producer, bounded FIFO with replaceable pending phrases.

    Capacity includes stream boundaries. Terminal EOF is never evicted and
    seals the queue. A timeout/overflow drops drafts first, then old finals.
    Only waiting work can be replaced; consumers take exactly one job at a time.
    """
    def __init__(self, maxsize=8, max_lag_s=8.0, on_drop=None, clock=time.monotonic):
        if maxsize < 1 or max_lag_s <= 0:
            raise ValueError("Queue size and maximum lag must be positive")
        self.maxsize = maxsize
        self.max_lag_s = max_lag_s
        self._items = deque()
        self._condition = threading.Condition()
        self._sealed = False
        self._on_drop = on_drop or (lambda event, reason: None)
        self._clock = clock

    def qsize(self):
        with self._condition:
            return len(self._items)

    def empty(self):
        return self.qsize() == 0

    def expired(self, event):
        return not isinstance(event, AudioStreamEnded) and self._clock() - event.meta.ended_at > self.max_lag_s

    def _prune(self, dropped):
        kept = deque()
        for event in self._items:
            if self.expired(event):
                dropped.append((event, "expired"))
            else:
                kept.append(event)
        self._items = kept

    def _report(self, dropped):
        # Callbacks run outside the queue lock (metrics also samples qsize()).
        for event, reason in dropped:
            self._on_drop(event, reason)

    def report_drop(self, event, reason):
        self._on_drop(event, reason)

    def put(self, event, block=True, timeout=None):
        dropped = []
        with self._condition:
            self._prune(dropped)
            if self._sealed:
                dropped.append((event, "after_eof"))
            elif self.expired(event):
                dropped.append((event, "expired"))
            else:
                index = next((i for i, old in enumerate(self._items)
                              if not isinstance(old, AudioStreamEnded)
                              and not isinstance(event, AudioStreamEnded)
                              and phrase_key(old.meta) == phrase_key(event.meta)), None)
                if index is not None:
                    old = self._items[index]
                    if supersedes(event.meta, old.meta):
                        self._items[index] = event
                        dropped.append((old, "coalesced"))
                    else:
                        dropped.append((event, "obsolete_revision"))
                else:
                    self._items.append(event)
                if isinstance(event, AudioStreamEnded) and event.source_finished:
                    self._sealed = True
                while len(self._items) > self.maxsize:
                    # Coalesce redundant nonterminal control markers first;
                    # segmentation already consumed them upstream.
                    victim = next((i for i, item in enumerate(self._items)
                                   if isinstance(item, AudioStreamEnded) and not item.source_finished), None)
                    if victim is None:
                        victim = next((i for i, item in enumerate(self._items)
                                       if not isinstance(item, AudioStreamEnded) and not item.meta.is_final), None)
                    if victim is None:
                        victim = next(i for i, item in enumerate(self._items)
                                      if not isinstance(item, AudioStreamEnded))
                    old = self._items[victim]
                    del self._items[victim]
                    dropped.append((old, "overflow"))
                self._condition.notify()
        self._report(dropped)

    def put_nowait(self, event):
        self.put(event, block=False)

    def get(self, block=True, timeout=None):
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            dropped = []
            result = None
            empty = False
            with self._condition:
                self._prune(dropped)
                if self._items:
                    result = self._items.popleft()
                elif not block:
                    empty = True
                elif not dropped:
                    remaining = None if deadline is None else deadline - time.monotonic()
                    if remaining is not None and remaining <= 0:
                        empty = True
                    else:
                        self._condition.wait(remaining)
            self._report(dropped)
            if result is not None:
                return result
            if empty:
                raise queue.Empty

    def get_nowait(self):
        return self.get(block=False)
