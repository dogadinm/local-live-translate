"""Tests for the pipeline logic that runs without a GPU, sound card or screen.

Cutting speech into phrases, discarding superseded drafts, and deciding which
Whisper models may take the English shortcut — the three places where a bug
produces plausible-looking output instead of an error. Run with pytest, or:

    python test_segmenter.py
"""
import queue
import threading
from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from audio import FRAME, SAMPLE_RATE, SpeechSegmenter
from events import AudioChunk, PhraseMeta, SpeechSegment, Transcript, Translation
from pipeline import drain, worker_transcribe, worker_translate

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
        out.extend(segmenter.push(AudioChunk(
            stream_id="test", started_at=100.0 + i / SAMPLE_RATE,
            sample_rate=SAMPLE_RATE, samples=stream[i : i + 1600],
        )))
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
    drafts = [event.samples for event in events if not event.meta.is_final]
    finals = [event.samples for event in events if event.meta.is_final]

    assert len(finals) == 1, "a phrase ends once"
    assert len(drafts) >= 3, f"4 s of speech should stream several drafts, got {len(drafts)}"
    assert events[-1].meta.is_final is True, "the final must come last"


def test_drafts_only_grow_and_the_final_is_longest():
    events = feed(np.concatenate([silence(2.0), speech(4.0), silence(1.5)]))
    lengths = [len(event.samples) for event in events]

    assert lengths == sorted(lengths), "each draft re-sends the phrase so far, so it can only grow"
    assert lengths[-1] == max(lengths), "the final carries the whole phrase"


def test_final_includes_preroll_and_trailing_silence():
    events = feed(np.concatenate([silence(2.0), speech(3.0), silence(1.5)]))
    final = next(event.samples for event in events if event.meta.is_final)

    # 3 s of speech, plus up to 250 ms of pre-roll and 400 ms of trailing pause
    assert 3.0 < seconds(final) < 3.8, seconds(final)


def test_speech_without_pauses_is_force_cut_at_the_ceiling():
    segmenter = SpeechSegmenter(max_speech_ms=6000)
    events = feed(np.concatenate([silence(1.0), speech(14.0), silence(1.5)]), segmenter)
    finals = [event.samples for event in events if event.meta.is_final]

    assert len(finals) >= 2, "14 s of unbroken speech cannot come back as one segment"
    assert all(seconds(audio) <= 6.2 for audio in finals), [seconds(a) for a in finals]


def test_two_phrases_separated_by_a_pause_stay_separate():
    events = feed(
        np.concatenate([silence(1.5), speech(2.0), silence(1.5), speech(2.0), silence(1.5)])
    )
    assert sum(event.meta.is_final for event in events) == 2


def test_frame_size_divides_the_block_size():
    # push() slices 100 ms blocks into 30 ms frames and carries the remainder;
    # this guards the arithmetic those two constants depend on
    assert FRAME == SAMPLE_RATE * 30 // 1000


# --- drain ----------------------------------------------------------------


def transcript(text="text", revision=1, final=False, phrase_id=1, stream_id="test", settings_version=0):
    return Transcript(
        PhraseMeta(stream_id, phrase_id, revision, 100.0, 101.0, final, settings_version),
        text, "en",
    )


def test_drain_keeps_only_the_newest_of_consecutive_drafts():
    work = queue.Queue()
    work.put(transcript("second", revision=2))
    work.put(transcript("third", revision=3))

    assert drain(work, transcript("first")) == [transcript("third", revision=3)]


def test_drain_never_drops_a_final():
    work = queue.Queue()
    work.put(transcript("b", final=True, phrase_id=2))
    work.put(transcript("c", final=True, phrase_id=3))

    assert [item.text for item in drain(work, transcript("a", final=True))] == ["a", "b", "c"]


def test_drain_starts_a_new_draft_run_after_a_final():
    work = queue.Queue()
    work.put(transcript("phrase-1-final", revision=2, final=True))
    work.put(transcript("phrase-2-draft", phrase_id=2))

    kept = drain(work, transcript("phrase-1-draft"))
    assert [item.text for item in kept] == ["phrase-1-draft", "phrase-1-final", "phrase-2-draft"]


def test_drain_on_an_empty_queue_returns_just_the_first_item():
    item = transcript("only", final=True)
    assert drain(queue.Queue(), item) == [item]


def test_drain_does_not_merge_different_phrases_streams_or_settings():
    first = transcript()
    for changes in ({"phrase_id": 2}, {"stream_id": "other"}, {"settings_version": 1}):
        other = replace(first, meta=replace(first.meta, revision=2, **changes))
        work = queue.Queue()
        work.put(other)
        assert drain(work, first) == [first, other]


def test_drain_only_replaces_with_a_strictly_newer_revision():
    first = transcript(revision=3)
    for revision in (2, 3):
        other = transcript(revision=revision)
        work = queue.Queue()
        work.put(other)
        assert drain(work, first) == [first, other]


def test_phrase_metadata_tracks_samples_and_revisions():
    stream = np.concatenate([silence(1.5), speech(2.0), silence(1.5), speech(2.0), silence(1.5)])
    events = feed(stream)
    finals = [event for event in events if event.meta.is_final]
    assert [event.meta.phrase_id for event in finals] == [1, 2]
    for final in finals:
        phrase = [event for event in events if event.meta.phrase_id == final.meta.phrase_id]
        assert [event.meta.revision for event in phrase] == list(range(1, len(phrase) + 1))
        assert all(event.meta.started_at == final.meta.started_at for event in phrase)
        assert [event.meta.ended_at for event in phrase] == sorted(event.meta.ended_at for event in phrase)
    for event in events:
        meta = event.meta
        assert meta.stream_id == "test" and meta.settings_version == 0
        start = round((meta.started_at - 100.0) * SAMPLE_RATE)
        end = round((meta.ended_at - 100.0) * SAMPLE_RATE)
        np.testing.assert_array_equal(event.samples, stream[start:end])


def test_forced_cuts_keep_contiguous_sample_timestamps():
    stream = np.concatenate([silence(1.0), speech(14.0), silence(1.5)])
    for event in feed(stream):
        start = round((event.meta.started_at - 100.0) * SAMPLE_RATE)
        end = round((event.meta.ended_at - 100.0) * SAMPLE_RATE)
        np.testing.assert_array_equal(event.samples, stream[start:end])


def test_segmenter_rejects_mixed_streams_and_wrong_audio_format():
    segmenter = SpeechSegmenter()
    chunk = AudioChunk("one", 100.0, SAMPLE_RATE, silence(0.1))
    segmenter.push(chunk)
    for bad in (replace(chunk, stream_id="two"), replace(chunk, sample_rate=48000),
                replace(chunk, samples=np.zeros((1600, 2)))):
        try:
            segmenter.push(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid chunk accepted")


def test_workers_preserve_metadata_to_overlay():
    for final in (False, True):
        meta = transcript(final=final, settings_version=7).meta
        segment = SpeechSegment(meta, SAMPLE_RATE, speech(1.0))
        audio_queue = queue.Queue()
        audio_queue.put(segment)
        text_queue = queue.Queue()
        stop = threading.Event()

        def recognize(audio, partial):
            assert audio is segment.samples
            assert partial is not final
            stop.set()
            return "hello", "en"

        worker_transcribe(SimpleNamespace(queue=audio_queue), SimpleNamespace(transcribe=recognize), text_queue, stop)
        recognized = text_queue.get_nowait()
        assert recognized.meta is meta
        text_queue.put(recognized)
        stop.clear()
        output = []

        def translate(text, language):
            assert (text, language) == ("hello", "en")
            stop.set()
            return "ahoj", "cs"

        worker_translate(text_queue, SimpleNamespace(translate_with_target=translate),
                         SimpleNamespace(update=output.append), stop)
        assert output == [Translation(meta, "hello", "ahoj", "en", "cs")]
        assert output[0].meta is meta


def test_translation_reports_target_used_even_if_picker_changes_mid_call():
    from translator import Translator
    from langs import flores

    translator = Translator.__new__(Translator)  # exercise routing without loading weights
    translator._lock = threading.Lock()
    translator.tgt_iso = "cs"
    translator.tokenizer = SimpleNamespace(
        encode=lambda text, **kwargs: [1],
        convert_ids_to_tokens=lambda ids: ["hello"],
        convert_tokens_to_ids=lambda tokens: [2],
        decode=lambda ids, **kwargs: "ahoj",
    )

    def infer(tokens, **kwargs):
        assert kwargs["target_prefix"] == [[flores("cs")]]
        translator.set_target("ru")
        return [SimpleNamespace(hypotheses=[["ahoj"]])]

    translator.translator = SimpleNamespace(translate_batch=infer)
    assert translator.translate_with_target("hello", "en") == ("ahoj", "cs")
    assert translator.tgt_iso == "ru"
    assert translator.translate_with_target("", "en") == ("", "ru")
    assert translator.translate_with_target("already translated", "ru") == ("already translated", "ru")
    assert translator.translate_with_target("hello", "unknown")[1] == "ru"
    assert translator.translate("already translated", "ru") == "already translated"


def test_overlay_keeps_event_until_display_and_styles_drafts_and_finals():
    from overlay import SubtitleOverlay, FG_DST, FG_SRC, AUTO
    from langs import display

    overlay = SubtitleOverlay.__new__(SubtitleOverlay)
    overlay._queue = queue.Queue()
    rendered = []
    statuses = []
    callbacks = []
    overlay.text_label = SimpleNamespace(config=lambda **kwargs: rendered.append(kwargs))
    overlay.status = SimpleNamespace(config=lambda **kwargs: statuses.append(kwargs))
    overlay.source_picker = SimpleNamespace(get=lambda: AUTO)
    overlay.root = SimpleNamespace(after=lambda delay, callback: callbacks.append(delay))
    events = []
    for final in (False, True):
        event = Translation(transcript(final=final).meta, "hello", "ahoj", "en", "cs")
        overlay.update(event)
        queued = overlay._queue.get_nowait()
        assert queued is event
        events.append(queued)
    for event in events:
        overlay.update(event)
    overlay.set_detected_language("en")
    overlay._poll()
    assert rendered == [{"text": "ahoj", "fg": FG_SRC}, {"text": "ahoj", "fg": FG_DST}]
    assert statuses == [{"text": f"detected: {display('en')}"}]
    assert callbacks == [80]


# --- which models may take the English shortcut ---------------------------


def test_turbo_and_distil_models_cannot_translate():
    from transcriber import _can_translate

    # asking these to translate returns untranslated text instead of an error,
    # so the check has to be right or the bug is invisible
    assert not _can_translate("large-v3-turbo")
    assert not _can_translate("turbo")
    assert not _can_translate("distil-large-v3")
    assert not _can_translate("small.en")


def test_multilingual_models_can_translate():
    from transcriber import _can_translate

    assert _can_translate("large-v3")
    assert _can_translate("large-v2")
    assert _can_translate("medium")
    assert _can_translate("small")


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
