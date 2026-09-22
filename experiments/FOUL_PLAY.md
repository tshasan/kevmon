# Benchmarking against foul-play

`foul-play` (github.com/pmariglia/foul-play) is the right yardstick for this project: it is the **same
architecture** we arrived at independently -- poke-engine, MCTS, DUCT for simultaneous moves, opponent teams filled
in by sampling -- and it has a published ladder result to anchor against.

> gen9 RandomBattle: **88% GXE, peak ELO 2341, Top 50**

Everything else we measure is against `SimpleHeuristicsPlayer`, which stops discriminating once a bot is past ~80%.

## Setup

```bash
git clone --depth 1 https://github.com/pmariglia/foul-play.git ../foul-play
cd ../foul-play && uv venv .venv
VIRTUAL_ENV=.venv uv pip install requests websockets python-dateutil
# reuse the wheel we already built -- foul-play pins poke-engine 0.0.48 with terastallization,
# which is exactly what scripts/build_poke_engine.sh produces, so there is no second Rust build
VIRTUAL_ENV=.venv uv pip install ../kevmon/third_party/wheels/poke_engine-0.0.48-*.whl
```

### The local-login patch you will need

foul-play always fetches a login assertion from `play.pokemonshowdown.com`, even for a guest login. A local server
started with `--no-security` issues a challstr the public auth server knows nothing about, so the assertion is
rejected and the bot silently never gets a username:

```
|nametaken|foulplay|Your authentication token was invalid.
|popup|You must choose a username before you challenge someone.
```

It still logs `Successfully logged in`, which is what makes this take a while to spot. In `fp/websocket_client.py`,
short-circuit `login()` for localhost and claim the name directly with an empty assertion:

```python
if "localhost" in (self.address or "") or "127.0.0.1" in (self.address or ""):
    await self.send_message("", ["/trn " + self.username + ",0,"])
    await asyncio.sleep(3)
    return self.username
```

### Direction matters

**foul-play must be the one issuing the challenge.** A poke-env player's username is auto-generated, so foul-play
cannot find us unless we pin ours with `AccountConfiguration`; and challenging *foul-play* fails with
`The user 'foulplay' was not found` until it has successfully taken its name. Pin our name and accept.

## Running

```bash
# terminal 1 -- us, waiting
python experiments/vs_foulplay.py --accept --opponent foulplay --username kevmcts \
    --battles 60 --iterations 240000 --samples 8

# terminal 2 -- foul-play, challenging
cd ../foul-play && .venv/bin/python run.py \
    --websocket-uri ws://localhost:8000/showdown/websocket \
    --ps-username foulplay --bot-mode challenge_user --user-to-challenge kevmcts \
    --pokemon-format gen9randombattle --run-count 60
```

## Matching compute

foul-play's defaults are `--search-time-ms 100`, `--search-parallelism 1`, `--search-threads 1`, i.e. roughly
230k iterations per decision at poke-engine's ~2.3M/s single-threaded. `--iterations 240000` on our side is the
comparable budget. Report the settings with any result: at different budgets the comparison means something else.

foul-play plays one battle at a time, so use `--concurrency 1` and expect ~20s per battle.
