"""The DAgger collection path: the student plays, the teacher labels.

The load-bearing claim of this file is the first test. DAgger is only worth running if the teacher's move can
actually be found among the options we describe to Kev -- if `order_key` and `SimpleHeuristicsPlayer` disagree about
how a move is named, nothing crashes and nothing warns: every turn is quietly counted as `skipped_unmodelled` and
the run produces an empty dataset after however many battles it was asked to play. That is the failure this file
exists to make loud, and it is checked against a real poke-env Battle and a real teacher, with no stand-ins.
"""
from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest
from poke_env.battle import Battle
from poke_env.player import DefaultBattleOrder, SimpleHeuristicsPlayer

from collect import StudentLogger, TurnLog, load_split
from kev_showdown.common import build_options


@pytest.fixture
def teacher() -> SimpleHeuristicsPlayer:
    """The real teacher, detached: `choose_singles_move` reads only the battle handed to it (it touches no `self`
    state at all), so an instance that never connects is a pure position -> move function."""
    return SimpleHeuristicsPlayer(start_listening=False)


def test_the_teachers_move_is_one_of_the_options_we_offer(battle: Battle, teacher: SimpleHeuristicsPlayer) -> None:
    """The whole mechanism in one line: the teacher picks a move for a position the student is in, and that move has
    to come back as one of the option keys. If it does not, DAgger silently collects nothing."""
    options = build_options(battle)
    turns = TurnLog(seed=0)
    turns.add(battle, options, teacher.choose_singles_move(battle)[0])

    assert turns.skipped_unmodelled == 0 and turns.skipped_forced == 0
    records = turns.records[battle.battle_tag]
    assert len(records) == 1
    label = records[0]["questions"]["action"]["label"]
    assert label in options, f"teacher picked {label!r}, which is not an option we described"


def test_the_label_never_leaks_into_the_option_that_carries_it(battle: Battle, teacher: SimpleHeuristicsPlayer) -> None:
    """A correction is only useful if it is not already given away: the record must describe every option the same
    way, with nothing marking the one the teacher chose."""
    turns = TurnLog(seed=0)
    turns.add(battle, build_options(battle), teacher.choose_singles_move(battle)[0])
    record = turns.records[battle.battle_tag][0]
    criteria = record["questions"]["action"]["criteria"]
    assert isinstance(criteria, dict)
    label = record["questions"]["action"]["label"]
    assert str(label) not in str(record["state"])
    assert len(set(criteria)) == len(criteria) and len(criteria) > 1


def test_a_forced_turn_is_skipped_rather_than_recorded(battle_recharge: Battle, teacher: SimpleHeuristicsPlayer) -> None:
    """One legal action is not a decision. Recording it would teach the model to answer questions it will never be
    asked, and would inflate accuracy with turns that had no wrong answer available."""
    turns = TurnLog(seed=0)
    options = build_options(battle_recharge)
    turns.add(battle_recharge, options, teacher.choose_singles_move(battle_recharge)[0])
    assert len(options) < 2 and not turns.records and turns.skipped_forced == 1


def test_an_order_we_do_not_model_is_counted_not_recorded(battle: Battle) -> None:
    turns = TurnLog(seed=0)
    turns.add(battle, build_options(battle), DefaultBattleOrder())  # type: ignore[no-untyped-call]
    assert not turns.records and turns.skipped_unmodelled == 1


def test_the_beta_coin_is_reproducible_and_independent_of_the_option_shuffle(battle: Battle) -> None:
    """Both draws are seeded from the same (run, battle, turn), so they must be told apart by purpose -- otherwise
    the turns the teacher takes over are exactly the turns with one particular shuffle, and the two are correlated
    in the collected data for no reason anyone would ever think to look for."""
    turns = TurnLog(seed=7)
    assert turns.rng(battle, "beta").random() == turns.rng(battle, "beta").random()      # reproducible from --seed
    assert turns.rng(battle, "beta").random() != turns.rng(battle, "options").random()   # and independent


@pytest.mark.asyncio
async def test_beta_one_plays_the_teacher_without_ever_calling_kev(battle: Battle) -> None:
    """The mixture policy has to be free: on a turn the teacher takes over, the student's answer would be discarded,
    so asking for it would be pure latency and GPU time. Nothing here is mocked -- the assertion is that no HTTP
    client was ever constructed, which is only true if that code path was genuinely never reached."""
    student = StudentLogger(seed=0, beta=1.0, kev_url="http://127.0.0.1:1", start_listening=False)
    order = await student.choose_move(battle)

    assert student.followed_teacher == 1
    assert student._http is None, "beta=1.0 still opened an HTTP client, so it paid for an answer it threw away"
    assert order.message == student.teacher.choose_singles_move(battle)[0].message
    assert len(student.turns.records[battle.battle_tag]) == 1   # and the turn was still recorded


@pytest.mark.asyncio
async def test_the_student_records_the_turn_even_when_kev_is_unreachable(battle: Battle) -> None:
    """The label does not depend on the model answering. A run that loses the server mid-way should come back with
    every turn it played labelled, and fall back to a random move for the ones it could not ask about."""
    student = StudentLogger(seed=0, beta=0.0, kev_url="http://127.0.0.1:1", start_listening=False)
    await student.choose_move(battle)

    assert sum(student.fallbacks.values()) == 1                 # Kev was down, as arranged
    assert len(student.turns.records[battle.battle_tag]) == 1   # and the teacher's label survived it


def test_merging_reads_an_earlier_datasets_split(tmp_path: Path, make_battle: Callable[..., Battle],
                                                 teacher: SimpleHeuristicsPlayer) -> None:
    """DAgger trains on the union of every iteration, so a run must be able to carry the previous dataset forward."""
    turns = TurnLog(seed=0)
    turns.add(make_battle(), build_options(make_battle()), teacher.choose_singles_move(make_battle())[0])
    earlier = [r for rs in turns.records.values() for r in rs]
    (tmp_path / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in earlier))

    assert load_split(tmp_path, "train") == earlier
    with pytest.raises(SystemExit):
        load_split(tmp_path, "development")   # a partial merge directory is a mistake, not a default


def test_the_shuffle_is_not_biased_towards_the_teachers_move(make_battle: Callable[..., Battle],
                                                             teacher: SimpleHeuristicsPlayer) -> None:
    """Same guard as the collection summary's `label_at_first_option`, applied to the DAgger path: across turns the
    label must land in every option position, not systematically first."""
    positions = []
    for turn in range(60):
        battle = make_battle(battle_tag=f"battle-gen9randombattle-{turn}")
        options = build_options(battle)
        turns = TurnLog(seed=turn)
        turns.add(battle, options, teacher.choose_singles_move(battle)[0])
        record = turns.records[battle.battle_tag][0]
        criteria = record["questions"]["action"]["criteria"]
        assert isinstance(criteria, dict)
        positions.append(list(criteria).index(str(record["questions"]["action"]["label"])))
    assert len(set(positions)) > 1, "the label sat in the same option position every single time"


def test_seeding_makes_a_dagger_turn_reproducible(battle: Battle, teacher: SimpleHeuristicsPlayer) -> None:
    """Compared as option ORDER, not as records: two dicts with the same pairs are equal whatever order they were
    built in, so comparing the records themselves would pass no matter what the shuffle did."""
    def option_order(seed: int) -> list[str]:
        turns = TurnLog(seed=seed)
        turns.add(battle, build_options(battle), teacher.choose_singles_move(battle)[0])
        criteria = turns.records[battle.battle_tag][0]["questions"]["action"]["criteria"]
        assert isinstance(criteria, dict)
        return list(criteria)

    assert option_order(3) == option_order(3)   # --seed reproduces a run
    assert option_order(3) != option_order(4)   # ...and is actually reaching the shuffle
