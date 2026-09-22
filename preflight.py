"""Refuse to rent a GPU until the data actually fits in the model.

Run this before every `kev_modal.py::train`. It costs nothing -- CPU only, no GPU, no Modal app -- and it answers the
one question that a training run answers expensively and ~40 minutes late: how many of these records can Kev's
context actually hold?

    cd ../kev && uv run python ../kevmon/preflight.py ../kevmon/data/pokebot-v2

It has to run from the `kev` checkout because it uses kev's own tokenizer and `kev.model.fits` -- the same check
`kev_modal.py::check_partitions` runs on the GPU box, so a pass here means a pass there.

Records that do not fit are not an error anyone sees: the trainer prints a `warning:` line and silently drops them,
so an over-long state does not fail a run, it just quietly deletes a quarter of the training set and bills you the
full price for what is left. The state budget is 384 tokens (`kev.model.MAX_STATE`) and the whole packed
state-plus-options encoding is capped at 2048 (`MAX_PACKED`), which is why adding a field to `render_state` is never
free: every option branch pays for it too.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

PARTITIONS = ("train", "calibration", "development")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("data", help="directory holding train/calibration/development.jsonl")
    ap.add_argument("--init-from", default="jaredpalmer/kev-4b", help="checkpoint whose tokenizer and base to check against")
    ap.add_argument("--max-dropped", type=float, default=0.005,
                    help="fraction of records allowed to exceed the context before this fails (default 0.5%%)")
    a = ap.parse_args()

    try:
        from kev.api import render
        from kev.checkpoint import Checkpoint
        from kev.data import load_records, materialize
        from kev.model import MAX_STATE, fits, load_tokenizer
    except ImportError as e:
        print(f"cannot import kev ({e}).\nRun this from the kev checkout: cd ../kev && uv run python ../kevmon/preflight.py {a.data}")
        return 2

    checkpoint = Checkpoint(a.init_from)
    tokeniser = load_tokenizer(checkpoint.meta.base, revision=checkpoint.meta.base_revision)
    worst = 0.0
    report = {}
    for partition in PARTITIONS:
        path = Path(a.data) / f"{partition}.jsonl"
        if not path.exists():
            print(f"MISSING {path}")
            return 2
        records = load_records(path)
        over = [r for r in records if not fits(materialize(r), tokeniser)]
        tokens = sorted(len(tokeniser(render(r["state"]), add_special_tokens=False).input_ids) for r in records)
        dropped = len(over) / len(records)
        worst = max(worst, dropped)
        median, p95 = statistics.median(tokens), tokens[int(0.95 * (len(tokens) - 1))]
        report[partition] = {
            "records": len(records), "would_be_dropped": len(over), "dropped_fraction": round(dropped, 4),
            "state_tokens": {"median": median, "p95": p95, "max": tokens[-1], "limit": MAX_STATE,
                             "headroom_at_p95": MAX_STATE - p95},
        }
        print(f"{partition:<12} {len(records):>6} records  {len(over):>5} would be dropped ({dropped:6.2%})  "
              f"state tokens median {median:>4} p95 {p95:>4} max {tokens[-1]:>4} / {MAX_STATE}")

    print(json.dumps(report, indent=2, default=str))
    if worst > a.max_dropped:
        print(f"\nFAIL: {worst:.2%} of records exceed Kev's context (limit {a.max_dropped:.2%}). "
              f"Shorten render_state in kev_showdown/common.py and re-collect before spending money on a GPU.")
        return 1
    print(f"\nOK: at most {worst:.2%} of records would be dropped. Safe to train.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
