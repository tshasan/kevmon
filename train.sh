#!/usr/bin/env bash
# Detached Modal training that survives this laptop: sleeping, losing wifi, closing the terminal, Ctrl-C.
#
#   ./train.sh launch pokebot-v6 data/pokebot-v2   # preflight, then launch and let go
#   ./train.sh status                              # what is running right now
#   ./train.sh logs pokebot-v6                     # follow the GPU log (safe to Ctrl-C)
#   ./train.sh resume pokebot-v4                   # continue an interrupted run from its checkpoint
#   ./train.sh pull pokebot-v6                     # reports into ../kev/runs/<name>/
#   ./train.sh cancel pokebot-v6                   # actually stop it (nothing else does)
#
# Why a script: `modal run` without --detach ties the GPU to this terminal, and the client dying -- which is what
# sleeping does to it -- cancels the run server-side. That is exactly how pokebot-v4 and v5 died, at step 360 and
# step 10 of 3129, mid-H100.
#
# This holds your Mac awake for the WHOLE run, not just the launch: caffeinate wraps the modal client, and the client
# lives until the run ends, so the local log keeps streaming instead of stalling the moment the machine idles. The run
# itself does not need this -- --detach is what protects it -- so KEEP_AWAKE=0 turns it off and only costs you the
# live local log. Closing the lid sleeps the Mac either way; that is still safe.
set -euo pipefail

KEVMON="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KEV="${KEV_REPO_DIR:-$KEVMON/../kev}"
SCRIPT="skills/kev-finetune/scripts/kev_modal.py"
LOGS="$KEVMON/logs"
export KEV_APP_NAME="${KEV_APP_NAME:-pokebot}"

# Every run is the same recipe unless you say otherwise. checkpoint-every is not optional here: it is the difference
# between an interruption costing you minutes and costing you the whole run.
INIT_FROM="${INIT_FROM:-jaredpalmer/kev-4b}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-200}"
EVAL_EVERY="${EVAL_EVERY:-100}"
TIMEOUT="${TIMEOUT:-14400}"   # 4h; the run is ~100 min of training plus calibration and scoring in the same container
# Hold the Mac awake for the run so the local log keeps streaming. 0 = let it idle-sleep; the detached run is
# unaffected either way, you just stop receiving log lines locally until you ./train.sh logs again.
KEEP_AWAKE="${KEEP_AWAKE:-1}"
if [ "$KEEP_AWAKE" = "0" ]; then AWAKE=(); else AWAKE=(caffeinate -i); fi

[ -d "$KEV" ] || { echo "no kev repo at $KEV (set KEV_REPO_DIR)" >&2; exit 1; }
KEV="$(cd "$KEV" && pwd)"
mkdir -p "$LOGS"

m() { (cd "$KEV" && uv run modal "$@"); }

app_id_file() { echo "$LOGS/$1.app-id"; }

require_app_id() {
  local f; f="$(app_id_file "$1")"
  [ -s "$f" ] || { echo "no app id recorded for '$1' (launched outside this script? use: ./train.sh status)" >&2; exit 1; }
  cat "$f"
}

# Hold this machine awake only until the run is provably server-side, then let go. An app id in the log is not
# enough on its own: modal creates the app and prints its id before the local entrypoint has finished its own
# checks, so a rejected name or a missing partition still exits *after* the id appears. Wait for the container to
# say something before calling it launched.
await_start() {
  local name="$1" log="$2" id="" i
  echo -n "waiting for the run to reach the GPU " >&2
  for i in $(seq 1 240); do
    [ -n "$id" ] || id="$(grep -oE 'ap-[A-Za-z0-9]{16,}' "$log" | head -1 || true)"
    if grep -qE 'SystemExit|Traceback|^Error|already exists|is not on the volume|missing data files' "$log"; then
      echo >&2; tail -20 "$log" >&2
      echo "launch failed locally -- no GPU was rented, nothing was billed" >&2; exit 1
    fi
    # the first line the container itself writes: the recipe it is about to train
    if [ -n "$id" ] && grep -qE 'training:|resuming|Qwen/|uploaded ' "$log"; then
      echo "$id" > "$(app_id_file "$name")"; echo " ok" >&2; echo "$id"; return 0
    fi
    echo -n "." >&2; sleep 2
  done
  echo >&2
  [ -n "$id" ] && { echo "$id" > "$(app_id_file "$name")"; echo "still starting after 8 min; app $id -- check ./train.sh logs $name" >&2; echo "$id"; return 0; }
  echo "no app id after 8 min; check $log" >&2; exit 1
}

cmd_launch() {
  local name="${1:-}"; shift || true
  local data="data/pokebot-v2"
  # a second bare argument is the data dir; anything starting with - is passed straight to kev_modal.py
  if [ "${1:-}" ] && [ "${1#-}" = "${1:-}" ]; then data="$1"; shift; fi
  [ -n "$name" ] || { echo "usage: ./train.sh launch <name> [data-dir] [extra modal flags]" >&2; exit 1; }
  [ -d "$KEVMON/$data" ] || [ -d "$data" ] || { echo "no data directory: $data" >&2; exit 1; }
  local abs_data; abs_data="$(cd "$KEVMON" && cd "$(dirname "$data")" && pwd)/$(basename "$data")"

  # Names are immutable on the volume. Finding that out after the upload wastes the launch, so check first.
  if m volume ls kev-finetune-runs 2>/dev/null | grep -qx "$name"; then
    echo "/runs/$name already exists on the volume; pick a new name (or: ./train.sh resume $name)" >&2; exit 1
  fi

  # Free, CPU-only, seconds. Never let an H100 discover what a laptop could have.
  echo "preflight $abs_data"
  (cd "$KEV" && uv run python "$KEVMON/preflight.py" "$abs_data")

  local log="$LOGS/$name.launch.log"
  echo "launching $name on ${KEV_GPU:-H100} (detached); launch log: $log"
  [ "$KEEP_AWAKE" = "0" ] || echo "holding this Mac awake for the whole run so the local log keeps streaming (KEEP_AWAKE=0 to disable)"
  # caffeinate covers the upload+handoff only. nohup+setsid-style backgrounding means closing this terminal is fine.
  # The redirect goes on the subshell, not just the command inside it: a background subshell that inherits this
  # script's stdout holds the caller's pipe open, so the terminal hangs long after the run is safely detached.
  ( cd "$KEV" && exec "${AWAKE[@]}" uv run modal run --detach "$SCRIPT::train" \
      --data "$abs_data" --name "$name" --init-from "$INIT_FROM" \
      --checkpoint-every "$CHECKPOINT_EVERY" --eval-every "$EVAL_EVERY" --timeout "$TIMEOUT" \
      "$@" ) </dev/null >"$log" 2>&1 &
  disown 2>/dev/null || true

  local id; id="$(await_start "$name" "$log")"

  cat <<EOF

  $name is running detached as $id

  The laptop is free now -- sleep it, close it, close this terminal. Nothing here holds the run up.

    ./train.sh logs $name      follow the GPU log
    ./train.sh status          is it still alive
    ./train.sh pull $name      reports, once it finishes (the detached run cannot download them itself)
EOF
}

cmd_resume() {
  local name="${1:-}"; shift || true
  [ -n "$name" ] || { echo "usage: ./train.sh resume <name>" >&2; exit 1; }
  if ! m volume ls kev-finetune-runs "$name/checkpoint" 2>/dev/null | grep -q train_state.pt; then
    echo "/runs/$name has no train_state.pt -- it died before its first checkpoint and cannot be resumed; launch a new name" >&2; exit 1
  fi
  local log="$LOGS/$name.resume.log"
  echo "resuming $name from its checkpoint (detached); launch log: $log"
  ( cd "$KEV" && exec "${AWAKE[@]}" uv run modal run --detach "$SCRIPT::resume" \
      --name "$name" --timeout "$TIMEOUT" --eval-every "$EVAL_EVERY" "$@" ) </dev/null >"$log" 2>&1 &
  disown 2>/dev/null || true
  local id; id="$(await_start "$name" "$log")"
  echo "resumed as $id -- laptop is free; follow with ./train.sh logs $name"
}

# The question this answers is "is my GPU still running", and the only honest source for that is Modal, not anything
# on this machine. A detached run outlives every local process, so a dead tail, a closed terminal and a slept laptop
# all look exactly like a dead run from here -- and none of them are one.
cmd_status() {
  local apps; apps="$(m app list 2>/dev/null)"
  local f n id state line
  for f in "$LOGS"/*.app-id; do
    [ -e "$f" ] || continue
    n="$(basename "$f" .app-id)"; id="$(cat "$f")"
    state="$(printf '%s\n' "$apps" | grep -F "$id" | grep -oE 'ephemeral|deployed|running|stopped' | head -1)"
    case "$state" in
      ephemeral|deployed|running) state="RUNNING on Modal";;
      stopped)                    state="stopped";;
      *)                          state="unknown (not in app list)";;
    esac
    # last progress line from whichever local log exists; it is only a convenience -- it goes stale the moment the
    # local streamer dies, which says nothing about the run.
    line="$(tail -n 40 "$LOGS/$n".*.log 2>/dev/null | grep -E '^(ep[0-9]+ step|eval step|checkpoint:)' | tail -1 || true)"
    printf '%-16s %-22s %s\n' "$n" "$state" "$id"
    [ -n "$line" ] && echo "                 last local log line: ${line:0:110}"
  done
  echo
  printf '%s\n' "$apps"
}

cmd_logs()   { m app logs "$(require_app_id "${1:?usage: ./train.sh logs <name>}")"; }
cmd_pull()   { (cd "$KEV" && uv run modal run "$SCRIPT::pull" --name "${1:?usage: ./train.sh pull <name>}" "${@:2}"); }
cmd_cancel() { m app stop "$(require_app_id "${1:?usage: ./train.sh cancel <name>}")" --yes; }

case "${1:-}" in
  launch|resume|status|logs|pull|cancel) c="$1"; shift; "cmd_$c" "$@";;
  *) sed -n '2,12p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 1;;
esac
