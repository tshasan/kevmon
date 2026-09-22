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
collect.py                 writes Kev training JSONL, split by battle: the teacher plays (cloning), or --dagger lets
                           the fine-tuned student play and the teacher label the positions it reaches
preflight.py               free CPU-only check that every record fits Kev's context, BEFORE you rent a GPU
train.sh                   launches the Modal fine-tune detached, so sleeping the laptop cannot kill the GPU run
evaluate.py                Kev vs Random / MaxBasePower / SimpleHeuristics: win rates with intervals, latency, calibration
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
python collect.py --out data/pokebot-v2     # 900 battles (200 vs Random, 200 vs MaxBasePower, 500 mirror), ~25k turns
```

The teacher is poke-env's `SimpleHeuristicsPlayer`. Every non-forced turn becomes a record with the teacher's
action as the `action` label and the final result as the `win` label, in the exact format of `kev.data.load_records`.

**Most battles are mirror matches, logged from both sides.** Teacher-vs-teacher games are the most competitive
positions available, both sides produce a good action label, and their outcomes are exactly anti-correlated, so the
data is free and self-balancing. Both perspectives of a mirror match share one battle tag and therefore always land
in the same split — two sides of one game are the same trajectory, and splitting them apart would leak it.

**The `win` question is then subsampled to a 50/50 target** (`--balance_win`, on by default). This matters more than
it sounds: the teacher beats Random and MaxBasePower badly, so raw collection lands near an 80% win rate and "yes"
becomes the right answer four times out of five. A noul head trained on that scores ~0.80 by ignoring the board
entirely, which looks like a working value signal and is not one. Balancing drops the `win` question from a random
share of the majority-outcome records; it never drops an `action` label, and a record carrying only `action` is
still valid input (`kev.data.load_records` requires a non-empty questions object, not every question).

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

Every run prints a `collect.split_written` line per split with the numbers that are worth seeing *before* renting a
GPU — label balance, options per record, the random-guess floor to compare accuracy against, and the share of
records whose label sits at the first option (which must track the random-guess floor, or the shuffle is biased):

```bash
python collect.py --out data/pokebot-v2 2>&1 | grep split_written | jq .
```

## 3. Preflight: check the data fits before renting anything

```bash
cd ../kev && uv run python ../kevmon/preflight.py ../kevmon/data/pokebot-v2
```

```
train         23027 records      0 would be dropped ( 0.00%)  state tokens median  256 p95  286 max  325 / 384
```

**Always run this.** It is free, CPU-only, and takes seconds. A record whose state exceeds Kev's 384-token budget
(`kev.model.MAX_STATE`) is not an error — the trainer prints a `warning:` line and silently drops it. So an
over-long `render_state` does not fail a run, it quietly deletes part of your training set and charges you full
price for whatever survived. This actually happened here: adding a raw base-stat block to the state pushed the
median from 256 to 369 tokens and put **26% of records over the limit**, which was only visible in the GPU log ~40
minutes and one H100 into a run.

Every field in `render_state` is paid for by the packed encoding too (`MAX_PACKED` = 2048 covers the state *and*
every option branch), so adding one is never free. Measure, don't guess.

## 4. Fine-tune from the released weights (not from the base)

Two ways to run this -- pick based on how fast you need the loop and how much you want to spend.

### Spend in this order

Never let an H100 be the thing that discovers a bug a laptop could have found:

| Step | Cost | Catches |
| --- | --- | --- |
| `preflight.py` | free | records that do not fit, missing partitions, malformed records |
| `pytest` + `mypy` | free | format drift between collection and play |
| `kev.train` locally on `kev-0.8b`, a few hundred records | free | the training loop itself, end to end |
| Modal on a cheap GPU (`KEV_GPU=L4`, $0.80/hr) with `kev-0.8b` | ~$1 | the full Modal path: volumes, scoring, calibration, reports |
| Modal H100 + `kev-4b` | ~$6-8 | the actual result |

### 4a. Modal (a rented cloud GPU, recommended for a real run)

Training a 4B+ model is slow on a laptop GPU (MPS/CPU), even with LoRA -- every forward/backward pass still runs
the full backbone. `../kev/skills/kev-finetune/scripts/kev_modal.py` rents a GPU per run (H100 by default, ~$3.95/hr;
cheaper tiers via `KEV_GPU`: L4 $0.80/hr, A10G $1.10/hr, A100 $2.10/hr) and handles the fine-tune, calibration and
scoring in one container. Needs a Modal account first (`uvx modal setup`) -- that's a real account with billing, so
set it up yourself before running this. Everything here runs under the Modal app name `pokebot`
(`KEV_APP_NAME=pokebot`, exported once below); run names (`--name`) are immutable, so a retry needs a new suffix
(`pokebot-v2`, ...).

Use `./train.sh`, in this repo. It is the raw `modal run` below with the four things that actually decide whether
a run survives the afternoon already set, and nothing else:

```bash
./train.sh launch pokebot-v6 data/pokebot-v2   # preflight, then launch detached and let go
./train.sh logs pokebot-v6                     # follow the GPU log (Ctrl-C is safe -- it only stops the tail)
./train.sh status                              # every pokebot app and its state
./train.sh pull pokebot-v6                     # reports into ../kev/runs/<name>/ once it finishes
./train.sh resume pokebot-v4                   # continue an interrupted run from its last checkpoint
./train.sh cancel pokebot-v6                   # the only thing that stops a detached run
```

What it sets, and why each one is the difference between a finished run and a wasted H100 hour:

- **`--detach`.** Without it the run dies the moment the local client disconnects — sleeping the laptop, dropping
  wifi, or Ctrl-C at the wrong moment all cancel a GPU you are paying for. This is not hypothetical: `pokebot-v4`
  and `pokebot-v5` both died exactly this way, at step 360 and step 10 of 3129, and the only trace is a
  `Received a cancellation signal` line at the end of `modal app logs`. Modal prints this advice in the logs, after
  it has already happened.
- **`caffeinate -i` for the whole run, not just the launch.** Two separate things need it. The upload of the split
  files and the handoff to Modal happen on this machine, and that window is the one place a sleep can still lose you
  the *launch*. After that, caffeinate stays because it wraps the modal client, and the client is what streams
  `logs/<name>.log` — let the Mac idle and the local log simply stops updating mid-run. So the launcher holds an
  idle-sleep assertion for the full run (a 100-minute fine-tune means 100 minutes of held-open sleep) and says so at
  launch, because an unexplained wide-awake laptop is its own kind of bug. `KEEP_AWAKE=0 ./train.sh launch ...` turns
  it off; the run is equally safe, you just lose the live local log and read it back with `./train.sh logs` instead.
  Closing the lid sleeps the Mac regardless, and that has always been fine — `--detach` is what protects the run.
- **`--checkpoint-every 200`.** An interruption then costs minutes instead of the whole run: `resume` reads the
  recipe back from the run's own `config.json` and finishes the checkpoint that run would have produced. A run that
  dies before its first checkpoint (as `pokebot-v5` did, at step 10) has no `train_state.pt` and is simply gone.
- **A name check before the upload.** Run names are immutable on the volume, so a reused name fails — and finding
  that out *after* the upload wastes the launch.

**A detached run looks dead from this machine, and mostly isn't.** It outlives every local process, so a closed
terminal, a finished `tail`, a slept laptop and a killed client all look identical to a crash from here -- and none
of them are one. Nothing local is evidence. `./train.sh status` asks Modal instead and prints `RUNNING on Modal` or
`stopped` per run, with the app id and the last line of whatever local log exists (that line goes stale the moment
the local streamer dies, which again says nothing about the run). `./train.sh logs <name>` reattaches to the live
GPU log from Modal at any point, from any terminal, however the previous one ended.

The reports are the one thing detaching costs you: the local client is what downloads them, so a detached run leaves
them on the volume for `./train.sh pull`.

Raw equivalent, if you want to see what it runs:

```bash
# in ../kev
export KEV_APP_NAME=pokebot
modal run --detach skills/kev-finetune/scripts/kev_modal.py::train \
  --data ../kevmon/data/pokebot-v2 --name pokebot-v4 --init-from jaredpalmer/kev-4b \
  --checkpoint-every 200 --eval-every 100
```

**Budget ~1.5-2 hours and $6-8, not 15 minutes.** At this data size (~25k records plus a 2,000-record replay slice)
the fine-tune is ~2,100 optimizer steps at roughly 0.20s/record, which is about 80 minutes on an H100 at $3.95/hr,
and calibration plus development scoring run afterwards in the same container. The run prints a live `eta` on every
tenth step — trust that number over this paragraph, and do not kill a run for being slower than you expected before
reading it.

Reports land in `runs/pokebot-v4/` on the Modal volume; pull them locally with
`modal run skills/kev-finetune/scripts/kev_modal.py::pull --name pokebot-v4`. Run names are immutable, so a retry
needs a fresh suffix. To check on a detached run or clear out dead ones:

```bash
modal app list                              # state of every pokebot run
modal app logs <app-id>                     # live training log
modal volume ls kev-finetune-runs           # what is taking up space
modal volume rm -r kev-finetune-runs <name> # delete an abandoned run (never kev-hf-cache: that is the model download)
```

### 4b. Local (`kev.train`, for a fast dry run before spending money on 4a)

```bash
# in ../kev
uv run python -m kev.train --data ../kevmon/data/pokebot-v2/train.jsonl \
  --base Qwen/Qwen3.5-4B-Base --init_from jaredpalmer/kev-4b \
  --epochs 2 --lr 2e-5 --batch 1 --accum 8 --dtype bf16 --checkpointing 1 --device cuda --out runs/pokebot

uv run python -m kev.benchmark --run runs/pokebot --data ../kevmon/data/pokebot-v2/development.jsonl --out runs/pokebot-eval
```

On Apple Silicon, drop `--dtype bf16` (CUDA-only; use `--weights_dtype bf16` instead, which works everywhere) and
consider `jaredpalmer/kev-0.8b` for the fast-iteration loop before committing to a full run on `kev-4b`/`kev-9b`.
`kev.benchmark` gives you imitation accuracy and calibration on the development split. A useful reference point:
random guessing over ~10 options is around 10%.

## 5. Play with the fine-tuned model

Also two ways, matching however you trained it in step 4.

### 5a. Modal serve (matches 4a; sidesteps the MPS slowness entirely)

`kev_modal.py`'s `Serve` endpoint (`KEV_SERVE_GPU`, default L4 $0.80/hr) scales to zero when idle
(5-minute window) -- it only bills while a battle is actually in flight, not around the clock. Deploy once, point
`evaluate.py`/`KevPlayer` at the resulting URL:

```bash
# in ../kev
KEV_APP_NAME=pokebot KEV_SERVE_RUN=pokebot-v4 modal deploy skills/kev-finetune/scripts/kev_modal.py
```

The deploy prints the actual URL (labelled `pokebot-api`, so it looks like `https://<your-workspace>--pokebot-api.modal.run`) -- use exactly what it prints:

```bash
# in kevmon, once deployed
python evaluate.py --kev <url from the deploy output> --battles 800 --concurrency 20 --log_file kev.log.jsonl
```

`--concurrency` is the difference between an eval you can afford to run and one you cannot: at one battle in flight
every decision is serialised behind ~4s of model latency, so 800 battles takes about ten hours. Twenty in flight
lets the server batch and brings that back to something you can run over lunch.

### 5b. Local (`kev.serve`, for quick checks without a deploy)

```bash
KEV_DTYPE=bf16 uv run --extra serve python -m kev.serve --run runs/pokebot --port 8009
python evaluate.py --battles 100 --save_replays replays --log_file kev.log.jsonl
```

## How to read the results

- **Read the confidence interval before the win rate.** `evaluate.py` prints a 95% Wilson interval on every number.
  At 100 battles around a 50% win rate that interval is about ±10 points, so resolving a 5-point improvement needs
  roughly **800 battles per opponent**, not 50. Two runs whose intervals overlap have not been shown to differ, and
  reporting the midpoints as progress is the easiest way to fool yourself in this whole project.
- **vs Random**: should be at or near 100%. If not, something is broken.
- **vs MaxBasePower**: shows whether it understands type matchups, not just raw power.
- **vs SimpleHeuristics**: this is the teacher, so behaviour cloning tops out around 50%.
- **p(win) from the `win` question**: `evaluate.py` reports `p_win_brier` against the real outcome of the battle each
  prediction was made in, next to `p_win_brier_if_constant_base_rate` — the score a model gets for ignoring the board
  and always answering the base rate. **If those two numbers are close, the value head is worthless** no matter how
  good its accuracy looks. Only once it clearly beats the constant baseline is p(win) worth using for a 1-ply lookahead.

### What it would take to beat the teacher

Imitation cannot exceed its teacher, and nothing in this repo or in kev is reinforcement learning — `kev_modal.py`
exposes `train`, `evaluate`, `compare`, `publish` and `serve`, all supervised. Three routes out, cheapest first:

1. **DAgger (implemented -- see step 6).** The immediate problem is not the ceiling, it is covariate shift: training
   states are the ones *SimpleHeuristics* reaches, while a weaker student drifts into positions the teacher rarely
   visited and has no label for. The fix is unusually cheap here because the teacher is a local pure function, not a
   human or a paid API: let the student drive, call `SimpleHeuristicsPlayer.choose_singles_move` on the same battle
   object for the label, and retrain on the union. One heuristic call per turn, no extra battles.
2. **A better teacher.** High-Elo human replays, same pipeline, same record format.
3. **Self-play with an outcome signal.** Play Kev against itself, keep the actions from won battles, retrain. Note
   what that is: reward-filtered imitation, which your supervised loss can already express as a 0/1 weight. Anything
   finer (weighting by an advantage rather than filtering on the outcome) needs a per-sample weight, and there is
   nowhere to put one -- `kev/train.py`'s `question_loss` is plain cross-entropy and the record contract in
   `kev/data.py` has no weight field. This is also the first point where the `win` head has to be genuinely
   calibrated, which is why balancing it in step 2 matters.

## 6. DAgger: retrain on the positions the student actually reaches

Behaviour cloning only ever sees states the *teacher* reached. The student is weaker, so it wanders into positions
the teacher almost never visited, has no label for, and misplays -- and each misplay takes it somewhere stranger
still. That compounding is usually what caps a cloned policy first, before the teacher's own skill does.

So: deploy the fine-tune, let it play, and label every turn with what the teacher *would* have done there.

```bash
# the student needs to be serving somewhere -- Modal (5a) or local (5b), either is fine
python collect.py --dagger <url from the deploy output> --out data/pokebot-v3 --merge data/pokebot-v2 \
    --battles 200 --mirror 500 --concurrency 20 --log_file dagger.log.jsonl
```

- **`--merge` is not optional in spirit.** DAgger trains on the *union* of every iteration, never on the newest
  slice alone: a student retrained only on its own corrections forgets what the previous round taught it and
  oscillates between two sets of mistakes. Merging is per split and never re-partitions what it reads, so an old
  record stays in the split it was assigned to and no trajectory can drift from train to development between
  iterations.
- **The teacher is also an opponent in this mode** (it is not in cloning mode, where it is the one playing). The
  positions a strong opponent forces the student into are exactly the ones cloning has no label for.
- **`--beta` is the classic DAgger mixture**: the probability of *playing* the teacher's move as well as recording
  it. Those turns skip the Kev call entirely, so a high beta is cheaper and faster, not slower. Use it only if the
  first student is bad enough that most battles end before anything interesting happens; `--beta 0` is the default
  and is what you want once the student is worth watching.
- The two labels now come from two different policies, on purpose. `action` is the teacher's move -- that is the
  correction. `win` is the *student's* own outcome, which is the only way the value head ever gets calibrated
  against the policy that will actually be playing.

Then preflight, train and score exactly as before, from the released weights again rather than from the v2 run:

```bash
cd ../kev && uv run python ../kevmon/preflight.py ../kevmon/data/pokebot-v3
KEV_APP_NAME=pokebot modal run --detach skills/kev-finetune/scripts/kev_modal.py::train \
  --data ../kevmon/data/pokebot-v3 --name pokebot-v5 --init-from jaredpalmer/kev-4b
```

The result to look at is not imitation accuracy -- it will *drop*, because the new states are harder ones the old
model has no idea about, and that is the point. It is the win rate against SimpleHeuristics, over enough battles to
resolve it (see the interval note above: ~800, not 50).

## Observability

Every `KevPlayer` decision and fallback emits one JSON line via `kev_showdown.telemetry` -- to stderr always, and
to a file when `--log_file` (or `KevPlayer(log_file=...)`) is set:

```json
{"event": "kev.decision", "battle_tag": "...", "turn": 5, "pick": "use earthquake with tera", "confidence": 0.47, "p_win": 0.43, "latency_ms": 3878.1, "ts": 1790048522.5, "level": "INFO"}
{"event": "kev.battle_end", "battle_tag": "...", "won": false, "turns": 23, "decisions": 21, "mean_p_win": 0.61, "ts": 1790048699.2, "level": "INFO"}
```

`kev.fallback` lines (level `WARNING`) fire whenever Kev is down, slow, or returns something the response model
rejects -- `evaluate.py`'s `fallbacks` count and `KevPlayer.stats()` summarise the same signal, but the JSONL is
what you `jq`/`grep` for a specific bad turn. No metrics backend to run; a log file is enough at this scale.

`kev.battle_end` exists so that every `p_win` has an outcome to be scored against. Without it the decision lines are
unfalsifiable — you can see what the model predicted but never whether it was right. `KevPlayer.calibration()` joins
the two and is what produces `p_win_brier` in the evaluate summary. This is also the only measurement that catches
the subtle version of the problem step 2 fixes at the data end: a `win` head can be perfectly calibrated *for the
teacher's policy* and badly overconfident for the weaker student actually playing the battles, and only scoring
predictions against the student's own results will show it.

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
- **`test_dagger.py`** covers the student-drives/teacher-labels path against a real `Battle` and a real
  `SimpleHeuristicsPlayer`. Its first test is the load-bearing one: the teacher's chosen order has to come back as
  one of the option keys we describe to Kev. If `order_key` and the teacher ever disagree about how a move is
  named, nothing crashes and nothing warns -- every turn is silently counted as `skipped_unmodelled` and the run
  produces an empty dataset after however many battles it was told to play. The rest pin the properties that make
  the data usable: forced turns are skipped rather than recorded, the label never appears in the state, the
  mixture coin is independent of the option shuffle, `--beta 1.0` never opens an HTTP connection at all (so a
  teacher-played turn really is free), and a turn is still labelled when Kev is unreachable.
- **`test_live_server.py`** (`-m live`) makes real HTTP calls to a running `kev.serve` instance -- there is no
  mock in that code path, so this file cannot be made to pass by faking a response; it only passes if an actual
  model returns well-formed, internally consistent probabilities for a real battle position it has never seen
  hardcoded anywhere. It's skipped (not failed) when nothing answers at `KEV_BASE_URL`
  (default `http://127.0.0.1:8009`).

## Notes

- Options are shuffled on every request, both at train and play time, because Kev can be option-order sensitive.
  `collect.py`'s `label_at_first_option` telemetry is the check that this is actually happening: it must stay close
  to `random_guess_accuracy`, or the model can learn a position instead of reading the descriptions.
- Type effectiveness, STAB, and HP % are pre-computed into the option text. Small models are bad at arithmetic, and
  giving them the numbers helps a lot (see kev issue #8).
- **The base-speed comparison and the atk/def vs spa/spd ratio are rendered too**, for a sharper reason: the teacher
  is a deterministic function, and both of those are inputs to it (`_estimate_matchup`'s speed term, and the
  physical/special ratio in `choose_singles_move`). A types-and-HP-only state cannot represent the teacher's own
  decision rule, so imitation accuracy was capped below 100% by the input format rather than by the model or the
  data volume. Raw base stats are deliberately *not* rendered on top of those two: at ~22 tokens per side they were
  the single most expensive thing in the state, and the preflight story in step 3 is what they cost.
- Tera options describe the position *after* terastallizing: Tera Blast becomes the tera type, and STAB is granted
  on the tera type in addition to the original types. Describing a tera option from the pre-tera types makes the
  option text argue against the very option it is offering.
- Only gen9 singles is modelled. Mega, Z-moves, and Dynamax orders are skipped. Tera is supported.
- Collection is reproducible from `--seed`: the option shuffle is seeded per (run, battle, turn) rather than once per
  player, because battles run concurrently and a single shared `Random` is consumed in whatever order the event loop
  happens to interleave them.
