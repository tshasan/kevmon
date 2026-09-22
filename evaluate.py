"""Does the fine-tune actually play better? Win rates against poke-env's baselines, with intervals, plus latency,
fallback and p(win) calibration stats.

    python evaluate.py --battles 200 --kev http://127.0.0.1:8009

    # the question that matters: is the fine-tune better than what it started from?
    python evaluate.py --battles 800 --concurrency 20 \
        --kev http://<finetuned-url> --baseline_kev http://<baseline-url> --log_file kev.log.jsonl

Read the numbers this way: Random is the floor, MaxBasePower is 'knows attacking is good', SimpleHeuristics is the
teacher. A behaviour-cloned Kev should approach ~50% vs SimpleHeuristics; beating it clearly needs more than imitation.

Every win rate carries a 95% Wilson interval, because the interval is usually wider than people expect and a run that
ignores it will happily report noise as progress: at 100 battles around a 50% win rate the interval is about +-10
points, so telling 50% from 55% needs roughly 800 battles per opponent, not 50. With --baseline_kev the difference
between the two models gets its own interval, and that interval -- not the two midpoints -- is the result.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math

from poke_env.player import MaxBasePowerPlayer, Player, RandomPlayer, SimpleHeuristicsPlayer

from kev_showdown.player import KevPlayer
from kev_showdown.telemetry import configure, emit

OPPONENTS: list[type[Player]] = [RandomPlayer, MaxBasePowerPlayer, SimpleHeuristicsPlayer]


def wilson(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a win rate. Preferred over the normal approximation because it stays inside
    [0, 1] and still behaves at the extremes -- a perfect record vs Random should not report an upper bound above 1."""
    if n == 0:
        return (0.0, 0.0)
    p = wins / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half_width = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return (max(0.0, centre - half_width), min(1.0, centre + half_width))


def newcombe(wins_a: int, n_a: int, wins_b: int, n_b: int) -> tuple[float, float]:
    """95% interval for the DIFFERENCE between two win rates (Newcombe's hybrid score method, built from the two
    Wilson intervals). Battles are not paired -- every gen9 random battle rolls a fresh team -- so this is the honest
    way to ask whether A beat B. An interval containing 0 means the two models have not been shown to differ."""
    if n_a == 0 or n_b == 0:
        return (0.0, 0.0)
    p_a, p_b = wins_a / n_a, wins_b / n_b
    low_a, high_a = wilson(wins_a, n_a)
    low_b, high_b = wilson(wins_b, n_b)
    delta = p_a - p_b
    return (delta - math.hypot(p_a - low_a, high_b - p_b), delta + math.hypot(high_a - p_a, p_b - low_b))


async def play(kev: KevPlayer, opponent_class: type[Player], battles: int, battle_format: str,
               concurrency: int) -> tuple[int, int, set[str]]:
    """One matchup. Returns (won, finished, the battle tags it produced) so every statistic can be sliced to it."""
    opponent = opponent_class(battle_format=battle_format, max_concurrent_battles=concurrency)
    before_won, before_done = kev.n_won_battles, kev.n_finished_battles
    before_tags = set(kev.battles)
    await kev.battle_against(opponent, n_battles=battles)
    return kev.n_won_battles - before_won, kev.n_finished_battles - before_done, set(kev.battles) - before_tags


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--battles", type=int, default=200)
    ap.add_argument("--format", default="gen9randombattle")
    ap.add_argument("--kev", default="http://127.0.0.1:8009")
    ap.add_argument("--baseline_kev", default="", help="a second Kev endpoint (e.g. the un-finetuned checkpoint) to A/B against")
    ap.add_argument("--concurrency", type=int, default=10,
                    help="battles in flight at once; 1 serialises every decision and makes a large --battles impractical")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--save_replays", default="", help="directory for replays (good for the demo video)")
    ap.add_argument("--log_file", default="", help="also write structured JSONL logs here")
    a = ap.parse_args()

    log = configure(a.log_file or None)
    players = {"finetuned": KevPlayer(battle_format=a.format, kev_url=a.kev, verbose=a.verbose,
                                      max_concurrent_battles=a.concurrency, save_replays=a.save_replays or False,
                                      log_file=a.log_file or None)}
    if a.baseline_kev:
        players["baseline"] = KevPlayer(battle_format=a.format, kev_url=a.baseline_kev, verbose=a.verbose,
                                        max_concurrent_battles=a.concurrency, log_file=a.log_file or None)

    results: dict[str, object] = {}
    for opponent_class in OPPONENTS:
        name = opponent_class.__name__
        records: dict[str, object] = {}
        counts: dict[str, tuple[int, int]] = {}
        for label, kev in players.items():
            won, done, tags = await play(kev, opponent_class, a.battles, a.format, a.concurrency)
            counts[label] = (won, done)
            low, high = wilson(won, done)
            records[label] = {"won": won, "battles": done, "win_rate": round(won / done, 3) if done else None,
                              "ci95": [round(low, 3), round(high, 3)], **kev.stats(tags)}
            print(f"{label:<10} vs {name:<21} {won:>4}/{done:<4} ({100 * won / max(done, 1):5.1f}%, "
                  f"95% CI {100 * low:4.1f}-{100 * high:4.1f}%)")

        if "baseline" in counts:
            low, high = newcombe(*counts["finetuned"], *counts["baseline"])
            verdict = "no significant difference" if low <= 0 <= high else ("finetuned better" if low > 0 else "finetuned WORSE")
            records["difference"] = {"delta_points": round(100 * (low + high) / 2, 1),
                                     "ci95_points": [round(100 * low, 1), round(100 * high, 1)], "verdict": verdict}
            print(f"{'':<10}    difference {100 * (low + high) / 2:+5.1f} points "
                  f"(95% CI {100 * low:+.1f} to {100 * high:+.1f}) -> {verdict}\n")
        results[name] = records

    summary = {"battles_per_opponent": a.battles, "by_opponent": results,
               "overall": {label: kev.stats() for label, kev in players.items()}}
    emit(log, "evaluate.session_summary", **summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
