"""Generate Kev fine-tuning data by watching a teacher play.

Teacher = poke-env's SimpleHeuristicsPlayer. Every non-forced turn becomes one labelled record with two questions:
  action (choice): the option the teacher picked            -> imitation (behaviour cloning)
  win    (noul)  : whether the teacher's side won the battle -> a free value signal, labelled after the battle ends

Records are split into train / heldout BY BATTLE (kev_showdown.split.battle_split), so no heldout turn shares a
battle with a training turn -- see tests/test_split.py for the leakage guard on that guarantee.

    python collect.py --battles 300 --out data
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import TypedDict

from poke_env.battle import AbstractBattle, Battle
from poke_env.player import BattleOrder, MaxBasePowerPlayer, RandomPlayer, SimpleHeuristicsPlayer

from kev_showdown.common import build_options, build_request
from kev_showdown.split import battle_split
from kev_showdown.telemetry import configure, emit
from kev_showdown.types import JSONContent


class LabelledQuestion(TypedDict, total=False):
    type: str
    instructions: JSONContent
    criteria: JSONContent
    label: JSONContent


class LabelledRecord(TypedDict):
    state: JSONContent
    questions: dict[str, LabelledQuestion]


log = configure()


class TeacherLogger(SimpleHeuristicsPlayer):
    def __init__(self, *args: object, seed: int = 0, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self.rng = random.Random(seed)
        self.records: dict[str, list[LabelledRecord]] = defaultdict(list)   # battle_tag -> records (win label filled later)
        self.skipped = 0

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        assert isinstance(battle, Battle), "kev_showdown only models gen9 singles battles, not doubles"
        order: BattleOrder = super().choose_move(battle)
        options = build_options(battle)
        if len(options) >= 2:
            label = next((k for k, (o, _) in options.items() if o.message == order.message), None)
            if label is None:
                self.skipped += 1          # teacher chose something we don't model (e.g. a default order)
            else:
                record: LabelledRecord = build_request(battle, options, self.rng, ask_win=True)  # type: ignore[assignment]
                record["questions"]["action"]["label"] = label
                self.records[battle.battle_tag].append(record)
        return order


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--battles", type=int, default=300, help="battles per opponent type")
    ap.add_argument("--format", default="gen9randombattle")
    ap.add_argument("--heldout", type=float, default=0.15)
    ap.add_argument("--out", default="data")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    teacher = TeacherLogger(battle_format=a.format, max_concurrent_battles=10, seed=a.seed)
    opponents = [RandomPlayer(battle_format=a.format, max_concurrent_battles=10),
                 MaxBasePowerPlayer(battle_format=a.format, max_concurrent_battles=10),
                 SimpleHeuristicsPlayer(battle_format=a.format, max_concurrent_battles=10)]
    for opp in opponents:
        before_won, before_done = teacher.n_won_battles, teacher.n_finished_battles
        await teacher.battle_against(opp, n_battles=a.battles)
        emit(log, "collect.opponent_done", opponent=type(opp).__name__,
             won=teacher.n_won_battles - before_won, finished=teacher.n_finished_battles - before_done)

    tags = [t for t in teacher.records if teacher.battles[t].finished]
    splits = battle_split(tags, a.heldout, random.Random(a.seed))

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for name, split_tags in splits.items():
        n = 0
        with open(out / f"{name}.jsonl", "w") as f:
            for tag in split_tags:
                won = bool(teacher.battles[tag].won)
                for rec in teacher.records[tag]:
                    rec["questions"]["win"]["label"] = won
                    f.write(json.dumps(rec) + "\n"); n += 1
        emit(log, "collect.split_written", split=name, battles=len(split_tags), records=n, path=str(out / f"{name}.jsonl"))

    emit(log, "collect.done", skipped=teacher.skipped)


if __name__ == "__main__":
    asyncio.run(main())
