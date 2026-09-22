#!/usr/bin/env bash
# Build poke-engine for gen9 and install it into kevmon's venv.
#
#   ./scripts/build_poke_engine.sh
#
# poke-engine is published to PyPI as an sdist only, and its generations are MUTUALLY EXCLUSIVE Cargo features.
# Two things about this build fail SILENTLY -- they produce a working import that is quietly wrong, which is worse
# than a build error, so this script asserts against both at the end:
#
#   1. The Python binding crate declares `default = ["poke-engine/gen4"]`. A plain `pip install poke-engine` gives
#      you a GEN 4 engine that happily accepts gen9 states. `--no-default-features` is what stops that; without it
#      the build dies with ~40 redefinition errors ending in `features "gen4" and "gen9" cannot be used together`.
#   2. `terastallization` is not cosmetic: without it `Side::can_use_tera()` is hard-coded false, tera never appears
#      among the engine's legal moves, and you are searching a subtly non-gen9 game.
#   3. Build from the sdist ROOT, never from poke-engine-py/. There is no pyproject.toml in that subdirectory, so
#      maturin falls back to defaults, drops the root's `python-source = "python"`, and ships a 127-byte __init__.py
#      instead of the real wrapper -- the package imports fine but has no monte_carlo_tree_search.
set -euo pipefail

VERSION="${POKE_ENGINE_VERSION:-0.0.48}"
KEVMON="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WHEELHOUSE="$KEVMON/third_party/wheels"
BUILD="${BUILD_DIR:-$KEVMON/.build/poke-engine}"

command -v cargo >/dev/null || { echo "cargo not found; install Rust (rustup default stable)" >&2; exit 1; }
rustup toolchain list 2>/dev/null | grep -qv "no installed toolchains" || {
  echo "rustup has no toolchain installed. Run: rustup default stable" >&2; exit 1; }

mkdir -p "$BUILD" "$WHEELHOUSE"
SRC="$BUILD/poke_engine-$VERSION"

if [ ! -d "$SRC" ]; then
  echo "downloading poke-engine $VERSION sdist"
  curl -fsSL "https://files.pythonhosted.org/packages/source/p/poke-engine/poke_engine-$VERSION.tar.gz" \
    -o "$BUILD/poke_engine-$VERSION.tar.gz"
  tar xzf "$BUILD/poke_engine-$VERSION.tar.gz" -C "$BUILD"
fi

# maturin goes in its own throwaway venv so it never lands in kevmon's dependency set
MATURIN_VENV="$BUILD/.maturin-venv"
[ -d "$MATURIN_VENV" ] || uv venv "$MATURIN_VENV" >/dev/null
VIRTUAL_ENV="$MATURIN_VENV" uv pip install --quiet maturin

echo "building gen9 + terastallization from the sdist root"
( cd "$SRC" && VIRTUAL_ENV="$MATURIN_VENV" "$MATURIN_VENV/bin/maturin" build --release \
    --no-default-features --features "poke-engine/gen9,poke-engine/terastallization" )

WHEEL="$(ls -t "$SRC"/target/wheels/poke_engine-*.whl | head -1)"
[ -n "$WHEEL" ] || { echo "no wheel produced" >&2; exit 1; }
cp "$WHEEL" "$WHEELHOUSE/"
echo "wheel -> $WHEELHOUSE/$(basename "$WHEEL")"

VIRTUAL_ENV="$KEVMON/.venv" uv pip install --quiet --force-reinstall "$WHEELHOUSE/$(basename "$WHEEL")"

# The canary. Each assert corresponds to one of the silent failures above; a wheel that imports but fails these
# would otherwise surface only as "the search player mysteriously isn't very good".
"$KEVMON/.venv/bin/python" - <<'PY'
import sys
import poke_engine as pe

assert hasattr(pe, "monte_carlo_tree_search"), \
    "no monte_carlo_tree_search: built from poke-engine-py/ instead of the sdist root, so the Python wrapper was dropped"

s = pe.State(
    side_one=pe.Side(pokemon=[pe.Pokemon(id="venusaur", level=84, types=("grass", "poison"),
        hp=280, maxhp=280, attack=180, defense=182, special_attack=205, special_defense=205, speed=180,
        ability="overgrow", moves=[pe.Move(id="gigadrain", pp=16)])], active_index="0"),
    side_two=pe.Side(pokemon=[pe.Pokemon(id="skarmory", level=84, types=("steel", "flying"),
        hp=260, maxhp=260, attack=190, defense=240, special_attack=120, special_defense=160, speed=170,
        ability="sturdy", moves=[pe.Move(id="bodypress", pp=16)])], active_index="0"),
)
r = pe.monte_carlo_tree_search(s, duration_ms=200)
assert r.total_visits > 0, "MCTS returned no visits"

# gen9 + terastallization: a -tera choice must be offered, or the wrong generation/feature set was compiled in
choices = {i.move_choice for i in r.side_one}
assert any(c.endswith("-tera") for c in choices), \
    f"no tera option in {choices}: built without --features poke-engine/terastallization (engine is not gen9)"

print(f"poke-engine ok: gen9 + tera, {r.total_visits:,} visits in 200ms")
PY
