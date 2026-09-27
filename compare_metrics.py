"""Compare two --metrics directories without loading models."""
import argparse
import json
from pathlib import Path


def compare(before, after):
    left = json.loads((Path(before) / "summary.json").read_text(encoding="utf-8"))
    right = json.loads((Path(after) / "summary.json").read_text(encoding="utf-8"))
    lines = []
    for key in ("audio_sha256", "whisper", "source_language", "target_language", "realtime", "warmup", "recognition_device", "recognition_compute_type", "translation_device", "packages"):
        if left["metadata"].get(key) != right["metadata"].get(key):
            lines.append(f"WARNING: different {key}; runs may not be comparable.")
    if not left["metadata"].get("audio_sha256") or not right["metadata"].get("audio_sha256"):
        lines.append("WARNING: no file fingerprint; input equality cannot be verified.")
    if left["outcome"] != "completed" or right["outcome"] != "completed":
        lines.append("WARNING: at least one run did not complete.")
    for report in (left, right):
        if report["counts"].get("settings_changes", 0):
            lines.append("WARNING: language settings changed during a run.")
            break
    rows = [("Whisper calls", "counts", "recognition_calls", None),
            ("NLLB inference calls", "counts", "nllb_calls", None),
            ("Translation requests", "counts", "translation_calls", None),
            ("Reused translations", "counts", "translation_cache_hits", None),
            ("ASR replaced drafts", "counts", "recognition_coalesced", None),
            ("ASR finals dropped", "counts", "recognition_finals_dropped", None),
            ("Translation finals dropped", "counts", "translation_finals_dropped", None),
            ("Display finals expired", "counts", "display_finals_dropped", None),
            ("Whisper median (s)", "stats", "recognition_s", "median"),
            ("Whisper p95 (s)", "stats", "recognition_s", "p95"),
            ("Translation p95 (s)", "stats", "translation_s", "p95"),
            ("ASR queue wait p95 (s)", "stats", "recognition_wait_s", "p95"),
            ("Draft display lag p95 (s)", "stats", "display_draft_lag_s", "p95"),
            ("Final display lag p95 (s)", "stats", "display_final_lag_s", "p95"),
            ("First subtitle p95 (s)", "stats", "first_display_since_phrase_start_s", "p95"),
            ("RAM sampled max (MiB)", "stats", "rss_mb", "max"),
            ("CPU median (% one core)", "stats", "cpu_percent_one_core", "median"),
            ("Displayed finals", "counts", "displayed_final", None)]
    lines.append(f"{'Metric':34} {'Before':>12} {'After':>12} {'Change':>12}")
    def row(label, a, b):
        delta = f"{(b / a - 1) * 100:+.1f}%" if a not in (None, 0) and b is not None else "n/a"
        fmt = lambda value: "n/a" if value is None else f"{value:.3f}"
        lines.append(f"{label:34} {fmt(a):>12} {fmt(b):>12} {delta:>12}")
    for label, section, key, stat in rows:
        values = [r[section].get(key, {} if stat else 0) for r in (left, right)]
        if stat:
            values = [value.get(stat) for value in values]
        row(label, *values)
    row("Finals not displayed", left["finals_not_displayed"], right["finals_not_displayed"])
    lines.append("Final subtitle texts: see display events with is_final=true in events.jsonl.")
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before")
    parser.add_argument("after")
    args = parser.parse_args()
    print(compare(args.before, args.after))
