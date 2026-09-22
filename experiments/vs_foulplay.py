"""Head-to-head against foul-play, the reference open-source bot for this format.

foul-play is the right yardstick because it is the same architecture -- poke-engine, MCTS, DUCT, sampled opponent
teams -- and it has a public ladder result to anchor against (gen9 RandomBattle: 88% GXE, peak 2341, Top 50). Our
own numbers to date are all against SimpleHeuristics, which stopped being informative once we crossed ~80%.

Run foul-play separately in accept_challenge mode against the same local server, then point this at its username.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poke_env import AccountConfiguration

from evaluate import wilson
from kev_showdown.engine.player import MctsPlayer


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--opponent", default="foulplay", help="foul-play's --ps-username")
    ap.add_argument("--battles", type=int, default=50)
    ap.add_argument("--iterations", type=int, default=100_000)
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--mode", default="sampled")
    ap.add_argument("--format", default="gen9randombattle")
    ap.add_argument("--concurrency", type=int, default=1, help="foul-play plays one battle at a time")
    ap.add_argument("--username", default="kevmcts", help="our name on the server; foul-play challenges this")
    ap.add_argument("--accept", action="store_true",
                    help="wait for foul-play to challenge us instead of challenging it (more reliable: a "
                         "poke-env player's own name is auto-generated, so the other side cannot find it)")
    a = ap.parse_args()

    me = MctsPlayer(mode=a.mode, iterations=a.iterations, samples=a.samples,
                    battle_format=a.format, max_concurrent_battles=a.concurrency,
                    account_configuration=AccountConfiguration(a.username, None))
    print(f"{'waiting for' if a.accept else 'challenging'} {a.opponent}: {a.battles} {a.format} battles "
          f"({a.iterations:,} iterations, {a.samples} samples, mode={a.mode}); we are '{a.username}'", flush=True)
    if a.accept:
        await me.accept_challenges(a.opponent, n_challenges=a.battles)
    else:
        await me.send_challenges(a.opponent, n_challenges=a.battles)

    won, done = me.n_won_battles, me.n_finished_battles
    low, high = wilson(won, done)
    print(f"\nresult: {won}/{done} = {100 * won / max(done, 1):.1f}%  95% CI {100 * low:.1f}-{100 * high:.1f}%")
    print(f"stats: {me.stats()}")
    if done:
        if high < 0.5:
            print("-> foul-play is stronger, and the interval excludes parity")
        elif low > 0.5:
            print("-> we are stronger, and the interval excludes parity")
        else:
            print("-> not resolved: the interval contains 50%, so no difference has been shown")


if __name__ == "__main__":
    asyncio.run(main())
