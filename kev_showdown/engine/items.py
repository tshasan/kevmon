"""A port of gen9 random battles' item assignment, from pokemon-showdown/data/random-battles/gen9/teams.ts.

Items are not in sets.json. The generator picks them with `getPriorityItem` followed by `getItem`, a long
deterministic decision tree over the moveset, ability, role and base stats -- all things the sampler already knows
once it has drawn a set. Sampling items uniformly instead would make sampled-belief play markedly worse than a real
opponent distribution and would inflate the belief-headroom measurement, so this is ported rather than approximated.

Faithfully ported: every singles branch of `getPriorityItem` and all of `getItem`.

Deliberately not ported, and why:
  * doubles (`getDoublesItem`, `isDoubles` branches) -- we only play singles.
  * `teamDetails` branches (rain/snow already on the team) -- team-level state we do not model.
  * `isLead` branches -- we do not know whether an unrevealed Pokemon led; treated as not-lead, which costs a
    Focus Sash on some Fast Support leads.
  * `this.randomChance` coin flips are honoured as real coin flips, because they are.
"""
from __future__ import annotations

import random
from typing import Any

# poke-env's pokedex/move entries are heterogeneous JSON straight from Showdown's data files -- a species entry
# mixes ints, strings, lists and nested dicts. Pretending to a precise shape here would be a fiction that the next
# field added to the dex quietly breaks, so these stay Any.
DexEntry = dict[str, Any]
TypeChart = dict[str, dict[str, float]]

RECOVERY_MOVES = frozenset({"healorder", "milkdrink", "moonlight", "morningsun", "recover", "roost", "shoreup",
                            "slackoff", "softboiled", "strengthsap", "synthesis"})
PHYSICAL_SETUP = frozenset({"bellydrum", "bulkup", "coil", "curse", "dragondance", "honeclaws", "howl", "meditate",
                            "poweruppunch", "swordsdance", "tidyup", "victorydance"})
SPEED_SETUP = frozenset({"agility", "autotomize", "flamecharge", "rockpolish", "snowscape", "trailblaze"})
SETUP = frozenset({"acidarmor", "agility", "autotomize", "bellydrum", "bulkup", "calmmind", "clangoroussoul",
                   "coil", "cosmicpower", "curse", "dragondance", "flamecharge", "growth", "honeclaws", "howl",
                   "irondefense", "meditate", "nastyplot", "noretreat", "poweruppunch", "quiverdance",
                   "rockpolish", "shellsmash", "shelter", "shiftgear", "swordsdance", "tailglow", "takeheart",
                   "tidyup", "trailblaze", "workup", "victorydance"})
HAZARDS = frozenset({"spikes", "stealthrock", "stickyweb", "toxicspikes"})
PIVOT_MOVES = frozenset({"chillyreception", "flipturn", "partingshot", "shedtail", "teleport", "uturn",
                         "voltswitch"})
DEFENSIVE_TERA_BLAST_USERS = frozenset({"alcremie", "bellossom", "comfey", "fezandipiti", "florges"})


class Counter:
    """The subset of teams.ts's MoveCounter that the item tree actually branches on."""

    def __init__(self, moves: tuple[str, ...], dex: dict[str, DexEntry], ability: str) -> None:
        self.moves = frozenset(moves)
        entries = {m: dex.get(m, {}) for m in moves}
        self.physical = sum(e.get("category") == "Physical" for e in entries.values())
        self.special = sum(e.get("category") == "Special" for e in entries.values())
        self.status = sum(e.get("category") == "Status" for e in entries.values())
        self.priority = sum(1 for e in entries.values()
                            if (e.get("priority") or 0) > 0 and e.get("category") != "Status")
        self.recovery = len(self.moves & RECOVERY_MOVES)
        self.setup = len(self.moves & SETUP)
        self.speedsetup = len(self.moves & SPEED_SETUP)
        self.physicalsetup = len(self.moves & PHYSICAL_SETUP)
        self.hazards = len(self.moves & HAZARDS)
        self.recoil = sum(1 for e in entries.values() if e.get("recoil") or e.get("hasCrashDamage"))
        self.sheerforce = sum(1 for e in entries.values()
                              if e.get("secondary") or e.get("secondaries")) if ability == "sheerforce" else 0
        self.skilllink = sum(1 for e in entries.values()
                             if isinstance(e.get("multihit"), list) and e.get("multihit") == [2, 5])


def _eff(species_entry: DexEntry, chart: TypeChart, attacking: str) -> int:
    """Showdown's `getEffectiveness` scale, which the item tree compares against and which is NOT a multiplier:
    0 is neutral, 1 is 2x, 2 is 4x, -1 is resisted. poke-env's type chart gives multipliers, so convert.

    Getting this wrong is silent and one-directional: comparing a multiplier against `>= 2` fires on everything
    2x-weak rather than only 4x, which over-assigned Air Balloon to a fifth of the dex."""
    mult = 1.0
    for t in species_entry.get("types", []):
        mult *= chart.get(t.upper(), {}).get(attacking.upper(), 1)
    if mult == 0:
        return -6                                    # immune; no branch wants it, but keep it well below any test
    steps = 0
    while mult > 1:
        mult /= 2; steps += 1
    while mult < 1:
        mult *= 2; steps -= 1
    return steps


def priority_item(species: str, entry: DexEntry, moves: tuple[str, ...], role: str, ability: str,
                  counter: Counter, chart: TypeChart, rng: random.Random) -> str | None:
    """getPriorityItem, singles branches. Returns None to fall through to `regular_item`."""
    ms, types = counter.moves, {t.lower() for t in entry.get("types", [])}
    if role == "Fast Bulky Setup" and ability in {"quarkdrive", "protosynthesis"}:
        return "boosterenergy"
    if species == "lokix":
        return "silverpowder" if role == "Fast Attacker" else "lifeorb"
    # Showdown's dex carries `requiredItems` (array); poke-env's carries `requiredItem` (string). Accept either,
    # or a forme with a mandatory item silently falls through to the generic tree and gets Leftovers.
    required = entry.get("requiredItems") or ([entry["requiredItem"]] if entry.get("requiredItem") else [])
    if required:
        req: list[str] = [str(r).replace(" ", "").replace("-", "").lower() for r in required]
        return req[0] if entry.get("baseSpecies") == "Arceus" else rng.choice(req)
    if species == "pikachu":
        return "lightball"
    if role == "AV Pivot":
        return "assaultvest"
    if species == "regieleki":
        return "magnet"
    if "normal" in types and "doubleedge" in ms and "fakeout" in ms:
        return "silkscarf"
    if species == "froslass" or "populationbomb" in ms or (ability == "hustle" and counter.setup and rng.random() < 0.5):
        return "widelens"
    if species == "smeargle":
        return "focussash"
    if "clangoroussoul" in ms or (species == "toxtricity" and "shiftgear" in ms):
        return "throatspray"
    if entry.get("baseSpecies") == "Magearna" and role == "Tera Blast user":
        return "weaknesspolicy"
    if ms & {"dragonenergy", "lastrespects", "waterspout"}:
        return "choicescarf"
    if ability == "imposter" or (species == "magnezone" and role == "Fast Attacker"):
        return "choicescarf"
    if species == "rampardos" and role == "Fast Attacker":
        return "choicescarf"
    if species == "palkia" and counter.status:
        return "lustrousorb"
    if "courtchange" in ms or species == "luvdisc" or (species == "terapagos" and "rest" not in ms):
        return "heavydutyboots"
    if ability in {"cheekpouch", "cudchew", "harvest", "ripen"} or ms & {"bellydrum", "filletaway"}:
        return "sitrusberry"
    if ms & {"healingwish", "switcheroo", "trick"}:
        spe = entry.get("baseStats", {}).get("spe", 0)
        if 60 <= spe <= 108 and role != "Wallbreaker" and not counter.priority:
            return "choicescarf"
        return "choiceband" if counter.physical > counter.special else "choicespecs"
    if counter.status and species in {"latias", "latios"}:
        return "souldew"
    if species == "scyther":
        return "heavydutyboots"                     # isLead branch not modelled; the non-lead result
    if ability in {"poisonheal", "quickfeet"}:
        return "toxicorb"
    if entry.get("evos"):                            # species.nfe
        return "eviolite"
    if (ability == "guts" or "facade" in ms) and "sleeptalk" not in ms:
        return "toxicorb" if ("fire" in types or ability == "toxicboost") else "flameorb"
    if ability == "magicguard" or (ability == "sheerforce" and counter.sheerforce):
        return "lifeorb"
    if ability == "angershell":
        return rng.choice(["expertbelt", "lumberry", "scopelens", "sitrusberry"])
    if counter.skilllink and ability != "skilllink" and species != "breloom":
        return "loadeddice"
    if ability == "unburden":
        return "whiteherb" if ms & {"closecombat", "leafstorm"} else "sitrusberry"
    if "shellsmash" in ms and ability != "weakarmor":
        return "whiteherb"
    if "meteorbeam" in ms or "electroshot" in ms:
        return "powerherb"
    if "acrobatics" in ms and ability != "protosynthesis":
        return "none"                                # the generator returns '' -- deliberately no item
    if "auroraveil" in ms or ("lightscreen" in ms and "reflect" in ms):
        return "lightclay"
    if ability == "gluttony":
        return rng.choice(["aguavberry", "figyberry", "iapapaberry", "magoberry", "wikiberry"])
    if species == "giratina" and "rest" in ms and "sleeptalk" not in ms:
        return "leftovers"
    if "rest" in ms and "sleeptalk" not in ms and ability not in {"naturalcure", "shedskin"}:
        return "chestoberry"
    if species != "yanmega" and _eff(entry, chart, "rock") >= 2:      # 4x
        return "heavydutyboots"
    return None


def regular_item(species: str, entry: DexEntry, moves: tuple[str, ...], role: str, ability: str,
                 counter: Counter, chart: TypeChart, rng: random.Random) -> str:
    """getItem, in the generator's own order."""
    ms = counter.moves
    base = entry.get("baseStats", {})
    spe, n = base.get("spe", 0), len(moves)
    life_orb_ok = not (ms & {"flamecharge", "nuzzle", "rapidspin"})

    if species != "jirachi" and counter.physical >= n and not (
            ms & {"dragontail", "fakeout", "firstimpression", "flamecharge", "rapidspin", "trailblaze"}):
        scarf = (role != "Wallbreaker" and (base.get("atk", 0) >= 100 or ability in {"hugepower", "purepower"})
                 and 60 <= spe <= 108 and ability != "speedboost" and not counter.priority)
        return "choicescarf" if (scarf and rng.random() < 0.5) else "choiceband"
    if counter.special >= n or (counter.special >= n - 1 and ms & {"flipturn", "uturn"}):
        scarf = (role != "Wallbreaker" and base.get("spa", 0) >= 100 and 60 <= spe <= 108
                 and ability not in {"speedboost", "tintedlens"} and "uturn" not in ms and not counter.priority)
        return "choicescarf" if (scarf and rng.random() < 0.5) else "choicespecs"
    if counter.speedsetup and not counter.physicalsetup and role == "Bulky Setup":
        return "weaknesspolicy"
    if not counter.status and role not in {"Fast Attacker", "Wallbreaker", "Tera Blast user"}:
        return "assaultvest"
    if species == "golem":
        return "weaknesspolicy" if counter.speedsetup else "custapberry"
    if "substitute" in ms:
        return "leftovers"
    if _eff(entry, chart, "rock") >= 1:                              # 2x or more
        return "heavydutyboots"
    if "chillyreception" in ms or (role == "Fast Support" and (ms & (PIVOT_MOVES | {"defog", "mortalspin", "rapidspin"}))
                                   and "flying" not in {t.lower() for t in entry.get("types", [])}
                                   and ability != "levitate"):
        return "heavydutyboots"
    if "dragondance" in ms and role == "Bulky Setup":
        return "weaknesspolicy"
    if ability == "roughskin" or (
            ability == "regenerator" and role in {"Bulky Support", "Bulky Attacker"}
            and base.get("hp", 0) + base.get("def", 0) >= 180 and rng.random() < 0.5) or (
            ability != "regenerator" and not counter.setup and counter.recovery
            and _eff(entry, chart, "fighting") < 1 and base.get("hp", 0) + base.get("def", 0) > 200
            and rng.random() < 0.5):
        return "rockyhelmet"
    if "outrage" in ms and counter.setup:
        return "lumberry"
    if "protect" in ms and ability != "speedboost":
        return "leftovers"
    if not counter.setup and ability != "levitate" and _eff(entry, chart, "ground") >= 2:   # 4x
        return "airballoon"
    if role in {"Bulky Attacker", "Bulky Support", "Bulky Setup"}:
        return "leftovers"
    if species == "pawmot" and "nuzzle" in ms:
        return "leppaberry"
    if role in {"Fast Support", "Fast Bulky Setup"}:
        return "lifeorb" if (counter.physical + counter.special > counter.status and life_orb_ok) else "leftovers"
    if role == "Tera Blast user" and species in DEFENSIVE_TERA_BLAST_USERS:
        return "leftovers"
    if life_orb_ok and role in {"Fast Attacker", "Setup Sweeper", "Tera Blast user", "Wallbreaker"}:
        return "lifeorb"
    return "leftovers"


def choose_item(species: str, entry: DexEntry, moves: tuple[str, ...], role: str, ability: str,
                dex: dict[str, DexEntry], chart: TypeChart, rng: random.Random) -> str:
    counter = Counter(moves, dex, ability)
    return (priority_item(species, entry, moves, role, ability, counter, chart, rng)
            or regular_item(species, entry, moves, role, ability, counter, chart, rng))
