"""collect.py's pure data-shaping steps: the win-label balancing and the per-split summary that reports it.

These are the guards on the failure that actually happened -- a teacher with an ~80% win rate made "yes" the right
answer four times out of five, so the noul head scored 0.80 by ignoring the board entirely and nothing in the
pipeline said so out loud.
"""
from __future__ import annotations

import random
from typing import cast

from collect import LabelledRecord, balance_win_questions, summarise


def record(won: bool, options: tuple[str, ...] = ("use earthquake", "switch to skarmory"),
           label: str = "use earthquake", turn: int = 3) -> LabelledRecord:
    return cast(LabelledRecord, {
        "state": {"turn": turn},
        "questions": {
            "action": {"type": "choice", "instructions": "pick", "criteria": {o: o for o in options}, "label": label},
            "win": {"type": "noul", "instructions": "win?", "label": won},
        },
    })


def per_tag(n_won: int, n_lost: int) -> dict[str, list[tuple[bool, LabelledRecord]]]:
    return {
        "won-battle": [(True, record(True)) for _ in range(n_won)],
        "lost-battle": [(False, record(False)) for _ in range(n_lost)],
    }


def win_labels(rows: dict[str, list[tuple[bool, LabelledRecord]]]) -> list[bool]:
    return [bool(r["questions"]["win"]["label"]) for batch in rows.values() for _, r in batch if "win" in r["questions"]]


def test_balancing_pulls_a_lopsided_win_rate_back_towards_even() -> None:
    rows = per_tag(n_won=800, n_lost=200)
    assert sum(win_labels(rows)) / len(win_labels(rows)) == 0.8

    report = balance_win_questions(rows, random.Random(0))
    labels = win_labels(rows)
    assert report["before_true_rate"] == 0.8
    assert report["majority_outcome"] is True
    assert 0.4 < sum(labels) / len(labels) < 0.6


def test_balancing_never_drops_an_action_label() -> None:
    """The imbalance is in the outcome, not in the teacher's move, so throwing away action labels to fix it would
    trade a real problem for a worse one."""
    rows = per_tag(n_won=500, n_lost=100)
    before = sum(len(batch) for batch in rows.values())
    balance_win_questions(rows, random.Random(0))
    after = [r for batch in rows.values() for _, r in batch]
    assert len(after) == before
    assert all("action" in r["questions"] and "label" in r["questions"]["action"] for r in after)
    assert any("win" not in r["questions"] for r in after), "nothing was balanced away at all"


def test_balancing_leaves_an_already_even_split_alone() -> None:
    rows = per_tag(n_won=300, n_lost=300)
    report = balance_win_questions(rows, random.Random(0))
    assert report["win_questions_dropped"] == 0
    assert report["keep_probability"] == 1.0


def test_summary_reports_balance_and_shuffle_health() -> None:
    """The summary is the early-warning system: it has to surface the win rate and the option-position bias that
    would otherwise only show up as a suspiciously good offline score."""
    records = [record(True) for _ in range(75)] + [record(False) for _ in range(25)]
    summary = summarise(records)
    assert summary["records"] == 100
    assert summary["win_questions"] == 100
    assert summary["win_true_rate"] == 0.75
    assert summary["action_kinds"] == {"move": 100}
    assert summary["random_guess_accuracy"] == 0.5
    assert summary["label_at_first_option"] == 1.0   # every fixture puts the label first: a real run must not


def test_summary_counts_records_whose_win_question_was_balanced_away() -> None:
    records = [record(True) for _ in range(10)]
    for r in records[:6]:
        del r["questions"]["win"]
    summary = summarise(records)
    assert summary["records"] == 10 and summary["win_questions"] == 4
