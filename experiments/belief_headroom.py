"""How much could knowing the opponent's team possibly be worth?

The gap between MCTS with the opponent's TRUE sets and MCTS with SAMPLED ones is the entire budget any future
belief model can ever spend. Measure it before building the model, not after: if the gap is small, a belief model
is decoration no matter how good it is; if it is large, it is the whole project.

Also reported: sampled - naive. If drawing from real set data barely beats knowing nothing, the sampler is broken
and the headline gap is not measuring what it claims to.

Every variant gets the same TOTAL iteration budget (see engine/player.py) -- otherwise the gap measures thinking
time as much as information.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # reuse evaluate.py's interval maths

from poke_env.player import MaxBasePowerPlayer, Player, RandomPlayer, SimpleHeuristicsPlayer

from evaluate import newcombe, wilson
from kev_showdown.engine.player import MODES, MctsPlayer

OPPONENTS = {"SimpleHeuristics": SimpleHeuristicsPlayer, "MaxBasePower": MaxBasePowerPlayer, "Random": RandomPlayer}


async def run(mode: str, opponent_class: type[Player], battles: int, concurrency: int,
              iterations: int, samples: int, fmt: str) -> tuple[int, int, dict]:
    player = MctsPlayer(mode=mode, iterations=iterations, samples=samples, battle_format=fmt,
                        max_concurrent_battles=concurrency)
    opponent = opponent_class(battle_format=fmt, max_concurrent_battles=concurrency)
    player.peer = opponent                       # only consulted by mode="perfect"
    await player.battle_against(opponent, n_battles=battles)
    return player.n_won_battles, player.n_finished_battles, player.stats()


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--battles", type=int, default=300)
    ap.add_argument("--concurrency", type=int, default=10)
    ap.add_argument("--iterations", type=int, default=60_000, help="TOTAL search iterations per decision, every mode")
    ap.add_argument("--samples", type=int, default=8, help="opponent teams sampled per decision in mode=sampled")
    ap.add_argument("--format", default="gen9randombattle")
    ap.add_argument("--opponents", default="SimpleHeuristics")
    a = ap.parse_args()

    results: dict[str, dict] = {}
    for name in a.opponents.split(","):
        opponent_class = OPPONENTS[name]
        print(f"\n=== vs {name} ({a.battles} battles each, {a.iterations:,} iterations per decision) ===")
        counts: dict[str, tuple[int, int]] = {}
        for mode in MODES:
            won, done, stats = await run(mode, opponent_class, a.battles, a.concurrency,
                                         a.iterations, a.samples, a.format)
            counts[mode] = (won, done)
            low, high = wilson(won, done)
            print(f"  {mode:<8} {won:>4}/{done:<4}  {100 * won / max(done, 1):5.1f}%  "
                  f"95% CI {100 * low:4.1f}-{100 * high:4.1f}%   fallback {stats['fallback_rate']:.3f}")
        results[name] = {m: {"won": w, "battles": d, "win_rate": round(w / max(d, 1), 3)}
                         for m, (w, d) in counts.items()}

        for a_mode, b_mode, label in (("perfect", "sampled", "HEADROOM for a belief model"),
                                      ("sampled", "naive", "value of sampling at all")):
            low, high = newcombe(*counts[a_mode], *counts[b_mode])
            mid = 100 * (low + high) / 2
            verdict = "not resolved" if low <= 0 <= high else ("real" if low > 0 else "NEGATIVE")
            print(f"  {a_mode} - {b_mode}: {mid:+5.1f} points  95% CI [{100 * low:+.1f}, {100 * high:+.1f}]"
                  f"  -> {verdict}   ({label})")
            results[name][f"{a_mode}_minus_{b_mode}"] = {
                "points": round(mid, 1), "ci95": [round(100 * low, 1), round(100 * high, 1)], "verdict": verdict}

    print("\n" + json.dumps(results, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
