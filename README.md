# kev-showdown

A local Kev decision model plays Pokémon Showdown (gen9 random battles) through poke-env.
Each turn becomes one Kev request: the battle state goes in, every legal action is an option in a `choice`
question, a `noul` question asks "will we win?", and one prefill pass answers both. No text generation.

```
kev_showdown/types.py     typed /v1/systemone request (TypedDict, zero-cost) and response (pydantic, validates real HTTP input)
kev_showdown/common.py    battle -> Kev request (shared by training AND play, so the formats never drift)
kev_showdown/player.py    KevPlayer: a poke-env Player that calls Kev each turn (falls back to random if Kev is down)
kev_showdown/split.py     pure battle-level train/calibration/development partitioning (the thing that must never leak)
kev_showdown/telemetry.py structured JSON-lines logging for every decision (latency, confidence, p(win), fallbacks)
collect.py                 watches a heuristic teacher play and writes Kev training JSONL, split by battle
evaluate.py                Kev vs Random / MaxBasePower / SimpleHeuristics: win rates, latency, confidence
tests/                      unit tests (real poke-env Battle fixtures, no live services) + `-m live` (real model, no mocks)
```

## 0. Setup

```bash
uv sync                      # poke-env, httpx, pydantic, pytest, mypy (see pyproject.toml)

# local Showdown server (the public one rate-limits bots)
git clone https://github.com/smogon/pokemon-showdown.git && cd pokemon-showdown
npm install && cp config/config-example.js config/config.js
node pokemon-showdown start --no-security

# the production Kev model, served locally from the sibling `kev` repo:
git clone https://github.com/jaredpalmer/kev.git ../kev && cd ../kev && uv sync --extra serve
KEV_DTYPE=bf16 uv run --extra serve python -m kev.serve --run jaredpalmer/kev-4b --port 8009
```

The first `kev.serve` run downloads the adapter (`jaredpalmer/kev-4b`) and base model
(`Qwen/Qwen3.5-4B-Base`, ~4 GB) from the Hub, then serves on `http://127.0.0.1:8009`. Check it's up with:

```bash
curl -s localhost:8009/v1/models
```

## 1. Zero-shot baseline (before any training)

```bash
python evaluate.py --battles 50 --verbose
```

Write these numbers down. That's your "before" chart.

## 2. Collect teacher data

```bash
python collect.py --battles 300 --out data/showdown-v1     # 900 battles total, roughly 20k labelled turns
```

The teacher is poke-env's `SimpleHeuristicsPlayer`. Every non-forced turn becomes a record with the teacher's
action as the `action` label and the final result as the `win` label, in the exact format of `kev.data.load_records`.

Battles are split by `kev_showdown.split.battle_split` into three files, by whole battle (unit-tested in
`tests/test_split.py` so no calibration/development turn ever shares a battle with a training turn):

| File | Fraction | Used for |
| --- | --- | --- |
| `train.jsonl` | remainder (~75%) | the fine-tune itself |
| `calibration.jsonl` | `--calibration` (default 0.10) | fitting the response temperature |
| `development.jsonl` | `--development` (default 0.15) | scoring the fine-tune against the baseline |

These are exactly the three file names kev's own Modal fine-tune skill expects
(`../kev/skills/kev-finetune/scripts/kev_modal.py`: `PARTITIONS = ("train", "calibration", "development")`), so
`--out` can point straight at a directory `kev_modal.py::train --data` reads.

## 3. Fine-tune from the released weights (not from the base)

Two ways to run this -- pick based on how fast you need the loop and how much you want to spend.

### 3a. Modal (a rented cloud GPU, recommended for a real run)

Training a 4B+ model is slow on a laptop GPU (MPS/CPU), even with LoRA -- every forward/backward pass still runs
the full backbone. `../kev/skills/kev-finetune/scripts/kev_modal.py` rents a GPU per run (H100 by default, ~$3.95/hr;
cheaper tiers via `KEV_GPU`: L4 $0.80/hr, A10G $1.10/hr, A100 $2.10/hr) and handles the fine-tune, calibration and
scoring in one container. Needs a Modal account first (`uvx modal setup`) -- that's a real account with billing, so
set it up yourself before running this.

```bash
# in ../kev
modal run skills/kev-finetune/scripts/kev_modal.py::train \
  --data ../kevmon/data/showdown-v1 --name showdown-v1 --init-from jaredpalmer/kev-4b
```

A delta fine-tune on a few hundred to a few thousand records like this typically runs in ~12-15 minutes for about
$1-3. Reports land in `runs/showdown-v1/` on the Modal volume; pull them locally with
`modal run skills/kev-finetune/scripts/kev_modal.py::pull --name showdown-v1`.

### 3b. Local (`kev.train`, for a fast dry run before spending money on 3a)

```bash
# in ../kev
uv run python -m kev.train --data ../kevmon/data/showdown-v1/train.jsonl \
  --base Qwen/Qwen3.5-4B-Base --init_from jaredpalmer/kev-4b \
  --epochs 2 --lr 2e-5 --batch 1 --accum 8 --dtype bf16 --checkpointing 1 --device cuda --out runs/showdown

uv run python -m kev.benchmark --run runs/showdown --data ../kevmon/data/showdown-v1/development.jsonl --out runs/showdown-eval
```

On Apple Silicon, drop `--dtype bf16` (CUDA-only; use `--weights_dtype bf16` instead, which works everywhere) and
consider `jaredpalmer/kev-0.8b` for the fast-iteration loop before committing to a full run on `kev-4b`/`kev-9b`.
`kev.benchmark` gives you imitation accuracy and calibration on the development split. A useful reference point:
random guessing over ~10 options is around 10%.

## 4. Play with the fine-tuned model

```bash
KEV_DTYPE=bf16 uv run --extra serve python -m kev.serve --run runs/showdown --port 8009
python evaluate.py --battles 100 --save_replays replays --log_file kev.log.jsonl
```

## How to read the results

- **vs Random**: should be at or near 100%. If not, something is broken.
- **vs MaxBasePower**: shows whether it understands type matchups, not just raw power.
- **vs SimpleHeuristics**: this is the teacher, so behaviour cloning tops out around 50%. Beating it needs a
  better teacher (high-Elo human replays) or self-play: play Kev against itself, keep the actions from won
  battles, and retrain.
- **p(win) from the `win` question**: check it's calibrated (when it says 0.8, does it win about 80% of the time?).
  If it is, you can use it for a 1-ply lookahead later.

## Observability

Every `KevPlayer` decision and fallback emits one JSON line via `kev_showdown.telemetry` -- to stderr always, and
to a file when `--log_file` (or `KevPlayer(log_file=...)`) is set:

```json
{"event": "kev.decision", "battle_tag": "...", "turn": 5, "pick": "use earthquake with tera", "confidence": 0.47, "p_win": 0.43, "latency_ms": 3878.1, "ts": 1790048522.5, "level": "INFO"}
```

`kev.fallback` lines (level `WARNING`) fire whenever Kev is down, slow, or returns something the response model
rejects -- `evaluate.py`'s `fallbacks` count and `KevPlayer.stats()` summarise the same signal, but the JSONL is
what you `jq`/`grep` for a specific bad turn. No metrics backend to run; a log file is enough at this scale.

## Type safety

`kev_showdown/types.py` splits the `/v1/systemone` contract by which side of the wire it's on. The *response* is a
pydantic model: it validates a real HTTP reply from Kev, which is untrusted input, and that's where catching a
malformed or drifted shape earns its runtime cost. The *request* is a plain `TypedDict`: `build_request` runs once
per battle turn (thousands of times with no model latency to hide the cost, in `collect.py`) building a request
from data this process already computed, so there's nothing to validate -- mypy checks the shape statically instead,
at zero runtime cost. Both mirror kev's own `kev/api.py` contract rather than importing it, so this project never
has to install kev's torch/transformers dependencies just to get an HTTP client's types. `pyproject.toml` runs the
whole project under `mypy --strict`:

```bash
uv run mypy
```

## Tests

```bash
uv run pytest                # everything, including live tests if a server answers at KEV_BASE_URL
uv run pytest -m "not live"  # offline only: no network, no running model, no Showdown server
uv run pytest -m live        # only the real-model integration tests (skipped, not failed, if nothing's listening)
```

- **Unit tests** (`test_common.py`, `test_split.py`, `test_types.py`) build a real `poke_env.battle.Battle` via
  its own public parsing API (`parse_request`/`parse_message`) -- not a hand-mocked stand-in -- and check the
  properties that must hold for training data to be trustworthy: option keys never collide or silently drop a
  legal move, option order is actually shuffled (not learnable as position bias), the rendered state never leaks
  the label (and stays a fixed shape even with no active Pokemon yet), and the train/calibration/development split
  never lets one battle straddle two of the three.
- **`test_live_server.py`** (`-m live`) makes real HTTP calls to a running `kev.serve` instance -- there is no
  mock in that code path, so this file cannot be made to pass by faking a response; it only passes if an actual
  model returns well-formed, internally consistent probabilities for a real battle position it has never seen
  hardcoded anywhere. It's skipped (not failed) when nothing answers at `KEV_BASE_URL`
  (default `http://127.0.0.1:8009`).

## Notes

- Options are shuffled on every request, both at train and play time, because Kev can be option-order sensitive.
- Type effectiveness, STAB, and HP % are pre-computed into the option text. Small models are bad at arithmetic, and
  giving them the numbers helps a lot (see kev issue #8).
- Only gen9 singles is modelled. Mega, Z-moves, and Dynamax orders are skipped. Tera is supported.
