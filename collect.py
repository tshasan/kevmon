"""Generate Kev fine-tuning data. Two modes, same record format.

BEHAVIOUR CLONING (default) -- the teacher plays and the teacher is labelled:

    python collect.py --out data/pokebot-v2

DAgger (`--dagger URL`) -- the *student* plays and the *teacher* labels its turns:

    python collect.py --dagger http://127.0.0.1:8009 --out data/pokebot-v3 --merge data/pokebot-v2

Teacher = poke-env's SimpleHeuristicsPlayer. Every non-forced turn becomes one labelled record with two questions:
  action (choice): the option the teacher picked        -> imitation (behaviour cloning)
  win    (noul)  : whether our side won the battle      -> a free value signal, labelled after the battle ends

Why the second mode exists: behaviour cloning is trained only on the states *SimpleHeuristics* reaches, but a weaker
student drifts into positions the teacher rarely visits and has no label for, and its mistakes compound from there.
That is covariate shift, and it is the first thing that caps a cloned policy -- not the teacher's own skill. The fix
is unusually cheap here because the teacher is a local pure function rather than a human or a paid API: let the
student drive, call the teacher on the very same battle object for the label, and retrain on the union of the old
data and the new. One heuristic call per turn, no extra battles, no reward model, no RL.

Note what each label means in DAgger data, because they no longer come from the same policy: `action` is the
teacher's move (that is the point -- it is the correction), while `win` is the *student's* own outcome. That is the
right pairing for both heads. The action head learns what should have been played in a state the student actually
reaches; the win head gets calibrated against the policy that will actually be playing, which is the one thing
teacher-only data can never give it.

Two things keep the `win` label honest. Most battles are mirror matches logged from BOTH sides, which is free data
and perfectly anti-correlated; and the `win` question is then subsampled down to a 50/50 split (`--balance_win`).
Without this a lopsided win rate makes "yes" the right answer most of the time and the noul head learns the base
rate instead of the position. Action labels are never dropped.

Records are split into train / calibration / development BY BATTLE (kev_showdown.split.battle_split), so no
calibration or development turn shares a battle with a training turn -- and, because both perspectives of a mirror
match share one battle tag, no battle's two sides straddle a split either. See tests/test_split.py. The three file
names match what kev's own Modal fine-tune skill expects (../kev/skills/kev-finetune/scripts/kev_modal.py:
PARTITIONS = train/calibration/development), so the output directory can be pointed at directly:

    modal run --detach ../kev/skills/kev-finetune/scripts/kev_modal.py::train --data data/pokebot-v3 --name pokebot-v5
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Protocol, TypedDict

from poke_env.battle import AbstractBattle, Battle
from poke_env.player import (BattleOrder, MaxBasePowerPlayer, Player, RandomPlayer, SimpleHeuristicsPlayer,
                             SingleBattleOrder)

from kev_showdown.common import build_options, build_request, order_key
from kev_showdown.player import KevPlayer
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


Options = dict[str, tuple[SingleBattleOrder, str]]

log = configure()


class TurnLog:
    """The labelled turns of one collecting player, plus the reasons turns were skipped.

    Both collectors own one of these and neither builds a record any other way, so the behaviour-cloning run and the
    DAgger run cannot drift apart in format. They differ only in who picks the move that is *played*; the move that
    is *labelled* always comes from the teacher and always becomes a record through `add`."""

    def __init__(self, seed: int) -> None:
        self.seed = seed
        self.records: dict[str, list[LabelledRecord]] = defaultdict(list)   # battle_tag -> records (win label filled later)
        self.skipped_forced = 0
        self.skipped_unmodelled = 0

    def rng(self, battle: Battle, purpose: str = "options") -> random.Random:
        """Seeded per (run, battle, turn, purpose) rather than once per player: battles run concurrently, so a single
        shared Random is consumed in whatever order the event loop interleaves them and --seed would not reproduce a
        run. `purpose` keeps independent draws independent -- the DAgger coin must not be correlated with the option
        shuffle of the very same turn."""
        return random.Random(f"{self.seed}:{purpose}:{battle.battle_tag}:{battle.turn}")

    def add(self, battle: Battle, options: Options, order: BattleOrder) -> None:
        """Record this turn with `order` -- the teacher's move -- as the action label."""
        if len(options) < 2:
            self.skipped_forced += 1      # no decision to model on a turn with one legal action
            return
        label = order_key(order) if isinstance(order, SingleBattleOrder) else None
        if label is None or label not in options:
            self.skipped_unmodelled += 1  # teacher chose something we don't model (e.g. a default order)
            return
        record: LabelledRecord = build_request(battle, options, self.rng(battle), ask_win=True)  # type: ignore[assignment]
        record["questions"]["action"]["label"] = label
        self.records[battle.battle_tag].append(record)


class RecordingPlayer(Protocol):
    """What `harvest` needs of a collector: its labelled turns, and the battles to read outcomes from."""
    turns: TurnLog

    @property
    def battles(self) -> dict[str, AbstractBattle]: ...


class TeacherLogger(SimpleHeuristicsPlayer):
    """Behaviour cloning: the teacher plays and every move it makes is a label."""

    def __init__(self, *args: object, seed: int = 0, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self.turns = TurnLog(seed)

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        assert isinstance(battle, Battle), "kev_showdown only models gen9 singles battles, not doubles"
        order: BattleOrder = super().choose_move(battle)
        self.turns.add(battle, build_options(battle), order)
        return order


class StudentLogger(KevPlayer):
    """DAgger: the student plays, the teacher labels the position the student got itself into.

    The teacher here is an ordinary SimpleHeuristicsPlayer that never connects to the server and never joins a
    battle -- `choose_singles_move` reads only the battle it is handed and touches no `self` state at all, so one
    detached instance is a pure position -> move function. Labelling costs one call per turn and no extra games.

    `--beta` is the classic DAgger mixture: with that probability the teacher's move is *played* as well as
    recorded, which keeps early iterations from spending most of their turns in positions a bad student throws
    itself into and never recovers from. Those turns skip the Kev call entirely (the student's answer would be
    discarded anyway), so a high beta makes a run cheaper and faster, not slower. beta=0 is pure DAgger and is the
    right setting once the student is worth watching."""

    def __init__(self, *args: object, seed: int = 0, beta: float = 0.0, **kwargs: object) -> None:
        super().__init__(*args, seed=seed, **kwargs)  # type: ignore[arg-type]
        self.turns = TurnLog(seed)
        self.beta = beta
        self.teacher = SimpleHeuristicsPlayer(start_listening=False)
        self.followed_teacher = 0

    async def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        assert isinstance(battle, Battle), "kev_showdown only models gen9 singles battles, not doubles"
        options = build_options(battle)
        teacher_order: BattleOrder = self.teacher.choose_singles_move(battle)[0]
        self.turns.add(battle, options, teacher_order)   # label first: this is the state the student is in NOW
        if self.beta > 0 and self.turns.rng(battle, "beta").random() < self.beta:
            self.followed_teacher += 1
            return teacher_order
        return await super().choose_move(battle)


def harvest(loggers: list[RecordingPlayer]) -> dict[str, list[tuple[bool, LabelledRecord]]]:
    """All finished battles as {battle_tag: [(won, record), ...]}, keyed so that both perspectives of a mirror match
    land under one tag -- the two sides of a battle are the same trajectory and must never be split apart."""
    per_tag: dict[str, list[tuple[bool, LabelledRecord]]] = defaultdict(list)
    for logger in loggers:
        for tag, records in logger.turns.records.items():
            battle = logger.battles.get(tag)
            if battle is None or not battle.finished:
                continue
            won = bool(battle.won)
            for record in records:
                per_tag[tag].append((won, record))
    return per_tag


def summarise(records: list[LabelledRecord]) -> dict[str, JSONContent]:
    """Everything we were previously blind to, per split: label balance, option counts, and a shuffle sanity check."""
    win_labels: Counter[bool] = Counter()
    kinds: Counter[str] = Counter()
    n_options, turns, uniform, at_first = [], [], [], 0
    for record in records:
        questions = record["questions"]
        win = questions.get("win")
        if win is not None:
            win_labels[bool(win["label"])] += 1
        criteria = questions["action"]["criteria"]
        assert isinstance(criteria, dict)
        keys = list(criteria)
        label = questions["action"]["label"]
        n_options.append(len(keys))
        uniform.append(1 / len(keys))
        at_first += keys.index(label) == 0 if label in keys else 0
        kinds["switch" if str(label).startswith("switch") else "tera" if "with tera" in str(label) else "move"] += 1
        state = record["state"]
        assert isinstance(state, dict)
        turns.append(state["turn"])
    n = len(records)
    win_total = sum(win_labels.values())
    return {
        "records": n,
        "win_questions": win_total,
        "win_true_rate": round(win_labels[True] / win_total, 3) if win_total else None,
        "action_kinds": dict(kinds),
        "options_mean": round(statistics.mean(n_options), 2) if n_options else None,
        "options_min": min(n_options, default=None),
        "options_max": max(n_options, default=None),
        "random_guess_accuracy": round(statistics.mean(uniform), 3) if uniform else None,
        "label_at_first_option": round(at_first / n, 3) if n else None,   # must track random_guess_accuracy, or the shuffle is biased
        "turn_mean": round(statistics.mean(turns), 1) if turns else None,
        "turn_max": max(turns, default=None),
    }


def balance_win_questions(per_tag: dict[str, list[tuple[bool, LabelledRecord]]], rng: random.Random) -> dict[str, JSONContent]:
    """Drop the `win` question (never the `action` label) from a random share of the majority-outcome records, so the
    noul head sees a ~50/50 target instead of the raw win rate. A record whose win question is dropped is still a
    valid training record -- kev.data.load_records only requires a non-empty questions object."""
    counts = Counter(won for rows in per_tag.values() for won, _ in rows)
    majority = counts[True] >= counts[False]
    keep = (min(counts[True], counts[False]) / counts[majority]) if counts[majority] else 1.0
    dropped = 0
    for rows in per_tag.values():
        for won, record in rows:
            if won is majority and rng.random() > keep:
                record["questions"].pop("win", None)
                dropped += 1
    return {"before_true_rate": round(counts[True] / max(sum(counts.values()), 1), 3),
            "majority_outcome": majority, "keep_probability": round(keep, 3), "win_questions_dropped": dropped}


def load_split(directory: Path, name: str) -> list[LabelledRecord]:
    """An earlier run's split, to be carried into this one's. DAgger trains on the UNION of every iteration's data,
    not on the newest slice alone -- a student retrained only on its own corrections forgets the positions the
    previous round already taught it, and oscillates between two sets of mistakes.

    Merging happens per split and never re-partitions what it reads, which is what keeps it honest: an old record
    stays in the split it was assigned to, so no trajectory can move from train to development between iterations.
    (Two runs can reuse a battle-tag string -- the server numbers battles from 1 each time it restarts -- but those
    are genuinely different games, so a repeated tag across runs is not a leak. Within a run, tags are unique.)"""
    path = directory / f"{name}.jsonl"
    if not path.exists():
        raise SystemExit(f"--merge {directory}: no {name}.jsonl (expected train/calibration/development)")
    with open(path) as f:
        records: list[LabelledRecord] = [json.loads(line) for line in f if line.strip()]
    return records


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--battles", type=int, default=200, help="battles against each fixed opponent")
    ap.add_argument("--mirror", type=int, default=500, help="self-play battles, logged from both sides")
    ap.add_argument("--format", default="gen9randombattle")
    ap.add_argument("--concurrency", type=int, default=10, help="battles in flight at once")
    ap.add_argument("--calibration", type=float, default=0.10, help="fraction of battles held out to fit temperature")
    ap.add_argument("--development", type=float, default=0.15, help="fraction of battles held out to score the fine-tune")
    ap.add_argument("--balance_win", action=argparse.BooleanOptionalAction, default=True,
                    help="subsample the win question to a 50/50 target (on by default)")
    ap.add_argument("--dagger", metavar="KEV_URL", default="",
                    help="DAgger mode: the model at this URL plays, and the teacher labels the turns it reaches")
    ap.add_argument("--beta", type=float, default=0.0,
                    help="DAgger only: probability of PLAYING the teacher's move as well as recording it (0 = pure student)")
    ap.add_argument("--merge", metavar="DIR", default="",
                    help="carry an earlier dataset's records into each split file, so the fine-tune sees the union")
    ap.add_argument("--log_file", default="", help="DAgger only: JSONL file for the student's per-decision telemetry")
    ap.add_argument("--out", default="data")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if not 0.0 <= a.beta <= 1.0:
        ap.error("--beta is a probability")
    if a.beta and not a.dagger:
        ap.error("--beta only means anything with --dagger: without a student there is nothing to mix the teacher into")

    common = {"battle_format": a.format, "max_concurrent_battles": a.concurrency}

    def collector(seed: int) -> TeacherLogger | StudentLogger:
        if not a.dagger:
            return TeacherLogger(seed=seed, **common)
        return StudentLogger(seed=seed, beta=a.beta, kev_url=a.dagger, log_file=a.log_file or None, **common)

    protagonist = collector(a.seed)
    mirror = collector(a.seed + 1)

    # In DAgger mode the teacher is also a fixed opponent: the positions a strong opponent forces the student into
    # are exactly the ones behaviour cloning has no label for, and they are the reason this run exists.
    opponents: list[tuple[Player, int]] = [(RandomPlayer(**common), a.battles), (MaxBasePowerPlayer(**common), a.battles)]
    if a.dagger:
        opponents.append((SimpleHeuristicsPlayer(**common), a.battles))
    opponents.append((mirror, a.mirror))

    for opponent, n_battles in opponents:
        if n_battles <= 0:
            continue
        before_won, before_done = protagonist.n_won_battles, protagonist.n_finished_battles
        await protagonist.battle_against(opponent, n_battles=n_battles)
        won, done = protagonist.n_won_battles - before_won, protagonist.n_finished_battles - before_done
        emit(log, "collect.opponent_done", opponent=type(opponent).__name__, battles=n_battles, won=won, finished=done,
             win_rate=round(won / done, 3) if done else None, both_sides=opponent is mirror)

    per_tag = harvest([protagonist, mirror])
    if not per_tag:
        raise SystemExit("no finished battles were recorded -- is the Showdown server running?")
    for rows in per_tag.values():
        for won, record in rows:
            record["questions"]["win"]["label"] = won
    if a.balance_win:
        emit(log, "collect.win_balanced", **balance_win_questions(per_tag, random.Random(a.seed)))

    splits = battle_split(list(per_tag), {"calibration": a.calibration, "development": a.development}, random.Random(a.seed))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, tags in splits.items():
        fresh = [record for tag in tags for _, record in per_tag[tag]]
        carried = load_split(Path(a.merge), name) if a.merge else []
        with open(out / f"{name}.jsonl", "w") as f:
            for record in carried + fresh:
                f.write(json.dumps(record) + "\n")
        emit(log, "collect.split_written", split=name, battles=len(tags), path=str(out / f"{name}.jsonl"),
             merged_from=a.merge or None, merged_records=len(carried), new_records=len(fresh),
             **summarise(carried + fresh))

    emit(log, "collect.done", mode="dagger" if a.dagger else "clone", battles=len(per_tag),
         records=sum(len(r) for r in per_tag.values()),
         skipped_forced=protagonist.turns.skipped_forced + mirror.turns.skipped_forced,
         skipped_unmodelled=protagonist.turns.skipped_unmodelled + mirror.turns.skipped_unmodelled)
    if isinstance(protagonist, StudentLogger) and isinstance(mirror, StudentLogger):
        # The student's own numbers, from the same battles that produced the labels: a free read on where it stands
        # before the next fine-tune, including whether p(win) beats its own base rate under its own policy.
        emit(log, "collect.student", beta=a.beta,
             teacher_moves_played=protagonist.followed_teacher + mirror.followed_teacher,
             **{k: v for k, v in protagonist.stats().items()})


if __name__ == "__main__":
    asyncio.run(main())
