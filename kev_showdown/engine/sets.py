"""Opponent set sampling from Showdown's own gen9 random-battle data.

poke-engine searches a single fully-specified state and has no notion of hidden information, so somebody has to
turn "they have shown me a Skarmory" into six concrete Pokemon. That is this module, and it is the seam a learned
belief model would eventually replace.

Item assignment is ported rather than approximated, deliberately. `getItem` in pokemon-showdown's
data/random-battles/gen9/teams.ts is a DETERMINISTIC decision tree over things we already know once a moveset is
drawn -- Choice item for mono-category movepools, Assault Vest with no status move, Boots when Rock-weak. Sampling
items uniformly instead would make sampled-belief play markedly worse than a real opponent distribution, which
would bias the headroom experiment toward the answer we are hoping for. The branches ported are listed in
`choose_item`; the unported tail falls through to Leftovers.
"""
from __future__ import annotations

import json
import random
from functools import lru_cache
from pathlib import Path

from poke_env.battle import Battle, Pokemon
from poke_env.data import GenData

from kev_showdown.engine.state import KnownSet
from kev_showdown.engine.stats import final_stats, randbats_evs_ivs

DEFAULT_SETS_PATH = Path.home() / "dev" / "pokemon-showdown" / "data" / "random-battles" / "gen9" / "sets.json"
PIVOT_MOVES = {"uturn", "flipturn", "voltswitch", "teleport", "partingshot", "chillyreception", "shedtail"}
_SLUG = str.maketrans({" ": "", "-": "", "'": "", ".": "", ":": ""})


def slug(name: str) -> str:
    return name.translate(_SLUG).lower()


@lru_cache(maxsize=1)
def load_sets(path: str = "") -> dict[str, dict]:
    p = Path(path) if path else DEFAULT_SETS_PATH
    if not p.exists():
        raise FileNotFoundError(
            f"gen9 randbats set data not found at {p}. It lives in the pokemon-showdown checkout "
            f"(data/random-battles/gen9/sets.json); set KEVMON_SETS_PATH or pass a path explicitly.")
    return json.loads(p.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _gen() -> GenData:
    return GenData.from_gen(9)


def _rock_weak(species: str) -> bool:
    """Rock effectiveness >= 1 in Showdown's terms means at least 2x -- the Heavy-Duty Boots trigger."""
    entry = _gen().pokedex.get(species, {})
    chart = _gen().type_chart
    mult = 1.0
    for t in entry.get("types", []):
        mult *= chart.get(t.upper(), {}).get("ROCK", 1)
    return mult >= 2


def choose_item(species: str, moves: tuple[str, ...], role: str, ability: str, rng: random.Random) -> str:
    """The ported head of teams.ts::getItem. Branches implemented, in the generator's own order:
    Choice Band/Scarf, Choice Specs/Scarf, Assault Vest, Substitute->Leftovers, Heavy-Duty Boots (Rock-weak),
    Protect->Leftovers, Bulky roles->Leftovers. Everything past that falls through to Leftovers."""
    cats = {m: _gen().moves.get(m, {}).get("category", "Status") for m in moves}
    physical = sum(c == "Physical" for c in cats.values())
    special = sum(c == "Special" for c in cats.values())
    status = sum(c == "Status" for c in cats.values())
    base = _gen().pokedex.get(species, {}).get("baseStats", {})
    spe = base.get("spe", 0)

    blockers = {"dragontail", "fakeout", "firstimpression", "flamecharge", "rapidspin", "trailblaze"}
    if physical >= len(moves) and not (set(moves) & blockers):
        scarf = (role != "Wallbreaker" and base.get("atk", 0) >= 100 and 60 <= spe <= 108)
        return "choicescarf" if (scarf and rng.random() < 0.5) else "choiceband"
    if special >= len(moves) or (special >= len(moves) - 1 and set(moves) & {"flipturn", "uturn"}):
        scarf = (role != "Wallbreaker" and base.get("spa", 0) >= 100 and 60 <= spe <= 108
                 and "uturn" not in moves)
        return "choicescarf" if (scarf and rng.random() < 0.5) else "choicespecs"
    if status == 0 and role not in {"Fast Attacker", "Wallbreaker", "Tera Blast user"}:
        return "assaultvest"
    if "substitute" in moves:
        return "leftovers"
    if _rock_weak(species):
        return "heavydutyboots"
    if "protect" in moves and ability != "speedboost":
        return "leftovers"
    return "leftovers"


def sample_set(species: str, rng: random.Random, observed: Pokemon | None = None) -> KnownSet | None:
    """One plausible set, pinned to whatever has already been revealed."""
    data = load_sets().get(species)
    if not data:
        return None
    level = data["level"]
    role = rng.choice(data["sets"])
    pool = [slug(m) for m in role["movepool"]]

    seen_moves = [m for m in (observed.moves if observed else {})]
    moves = list(dict.fromkeys(seen_moves))                      # revealed moves are pinned, never resampled
    remaining = [m for m in pool if m not in moves]
    rng.shuffle(remaining)
    moves += remaining[: max(0, 4 - len(moves))]
    moves = tuple(moves[:4])

    ability = slug(observed.ability) if (observed and observed.ability) else slug(rng.choice(role["abilities"]))
    tera = (observed.tera_type.name.lower() if (observed and observed.tera_type is not None)
            else slug(rng.choice(role["teraTypes"])) if role.get("teraTypes") else "typeless")
    item = (observed.item if (observed and observed.item and observed.item != "unknown_item")
            else choose_item(species, moves, role.get("role", ""), ability, rng))

    base = _gen().pokedex.get(species, {}).get("baseStats")
    if not base:
        return None
    evs, ivs = randbats_evs_ivs(frozenset(moves), {m: _gen().moves.get(m, {}).get("category", "") for m in moves},
                                species=species, ability=ability, base_stats=base)
    entry_types = [t.lower() for t in _gen().pokedex[species].get("types", [])]
    types = (entry_types[0] if entry_types else "typeless",
             entry_types[1] if len(entry_types) > 1 else "typeless")
    return KnownSet(species=species, level=level, types=types,
                    stats=final_stats(base, level, evs, ivs), moves=moves,
                    ability=ability or "none", item=item or "none", tera_type=tera)


def sample_opponent_team(battle: Battle, rng: random.Random) -> dict[str, KnownSet]:
    """Revealed species keep what we have seen; the rest of their six are drawn from the set data."""
    out: dict[str, KnownSet] = {}
    for mon in battle.opponent_team.values():
        s = sample_set(mon.species, rng, mon)
        if s is not None:
            out[mon.species] = s

    pool = [s for s in load_sets() if s not in out]
    rng.shuffle(pool)
    for species in pool:
        if len(out) >= 6:
            break
        s = sample_set(species, rng, None)
        if s is not None:
            out[species] = s
    return out


def naive_opponent_team(battle: Battle) -> dict[str, KnownSet]:
    """The floor: only what has literally been seen, with no set data filled in at all."""
    return {}
