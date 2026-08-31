"""Tests for the two pieces of pure logic in the pipeline.

Everything else needs a GPU, a sound card or a screen; these two do not, and
they are where the subtle bugs live. Run with pytest, or directly:

    python test_segmenter.py
"""
import queue

import numpy as np

from audio import FRAME, SAMPLE_RATE, SpeechSegmenter
from pipeline import drain

rng = np.random.default_rng(0)


def silence(seconds: float, level: float = 0.0015) -> np.ndarray:
    """Room tone: audible to a naive threshold, not to an adaptive one."""
    return (rng.standard_normal(int(SAMPLE_RATE * seconds)) * level).astype(np.float32)


def speech(seconds: float) -> np.ndarray:
    """A tone wobbling at 4 Hz — close enough to syllables for the detector."""
    t = np.arange(int(SAMPLE_RATE * seconds)) / SAMPLE_RATE
    wave = 0.25 * np.sin(2 * np.pi * 180 * t) * (1 + 0.5 * np.sin(2 * np.pi * 4 * t))
    return wave.astype(np.float32)


def feed(stream: np.ndarray, segmenter: SpeechSegmenter | None = None):
    """Push audio through in 100 ms blocks, as the capture thread does."""
    segmenter = segmenter or SpeechSegmenter()
    out = []
    for i in range(0, len(stream), 1600):
        out.extend(segmenter.push(stream[i : i + 1600]))
    return out


def seconds(audio: np.ndarray) -> float:
    return len(audio) / SAMPLE_RATE


# --- SpeechSegmenter ------------------------------------------------------


def test_background_noise_alone_emits_nothing():
    assert feed(silence(6.0)) == []


def test_click_shorter_than_min_speech_is_dropped():
    # a notification blip, not speech: 150 ms of sound between quiet stretches
    events = feed(np.concatenate([silence(2.0), speech(0.15), silence(1.5)]))
    assert events == []


def test_phrase_emits_drafts_then_exactly_one_final():
    events = feed(np.concatenate([silence(2.0), speech(4.0), silence(1.5)]))
    drafts = [audio for audio, final in events if not final]
    finals = [audio for audio, final in events if final]

    assert len(finals) == 1, "a phrase ends once"
    assert len(drafts) >= 3, f"4 s of speech should stream several drafts, got {len(drafts)}"
    assert events[-1][1] is True, "the final must come last"


def test_drafts_only_grow_and_the_final_is_longest():
    events = feed(np.concatenate([silence(2.0), speech(4.0), silence(1.5)]))
    lengths = [len(audio) for audio, _ in events]

    assert lengths == sorted(lengths), "each draft re-sends the phrase so far, so it can only grow"
    assert lengths[-1] == max(lengths), "the final carries the whole phrase"


def test_final_includes_preroll_and_trailing_silence():
    events = feed(np.concatenate([silence(2.0), speech(3.0), silence(1.5)]))
    final = next(audio for audio, is_final in events if is_final)

    # 3 s of speech, plus up to 250 ms of pre-roll and 400 ms of trailing pause
    assert 3.0 < seconds(final) < 3.8, seconds(final)


def test_speech_without_pauses_is_force_cut_at_the_ceiling():
    segmenter = SpeechSegmenter(max_speech_ms=6000)
    events = feed(np.concatenate([silence(1.0), speech(14.0), silence(1.5)]), segmenter)
    finals = [audio for audio, final in events if final]

    assert len(finals) >= 2, "14 s of unbroken speech cannot come back as one segment"
    assert all(seconds(audio) <= 6.2 for audio in finals), [seconds(a) for a in finals]


def test_two_phrases_separated_by_a_pause_stay_separate():
    events = feed(
        np.concatenate([silence(1.5), speech(2.0), silence(1.5), speech(2.0), silence(1.5)])
    )
    assert sum(1 for _, final in events if final) == 2


def test_frame_size_divides_the_block_size():
    # push() slices 100 ms blocks into 30 ms frames and carries the remainder;
    # this guards the arithmetic those two constants depend on
    assert FRAME == SAMPLE_RATE * 30 // 1000


# --- drain ----------------------------------------------------------------


def test_drain_keeps_only_the_newest_of_consecutive_drafts():
    work = queue.Queue()
    work.put(("second", False))
    work.put(("third", False))

    assert drain(work, ("first", False)) == [("third", False)]


def test_drain_never_drops_a_final():
    work = queue.Queue()
    work.put(("b", True))
    work.put(("c", True))

    assert drain(work, ("a", True)) == [("a", True), ("b", True), ("c", True)]


def test_drain_starts_a_new_draft_run_after_a_final():
    work = queue.Queue()
    work.put(("phrase-1-final", True))
    work.put(("phrase-2-draft", False))

    kept = drain(work, ("phrase-1-draft", False))
    assert kept == [("phrase-1-draft", False), ("phrase-1-final", True), ("phrase-2-draft", False)]


def test_drain_on_an_empty_queue_returns_just_the_first_item():
    assert drain(queue.Queue(), ("only", True)) == [("only", True)]


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    failed = 0
    for test in tests:
        try:
            test()
            print(f"  ok    {test.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"  FAIL  {test.__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
