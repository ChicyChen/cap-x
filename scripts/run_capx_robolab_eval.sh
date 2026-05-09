#!/usr/bin/env bash
# Single-shot CaP-X evaluation on a robolab LH task.
#
# Sequence:
#   1. (optional) wipe stale outputs
#   2. run capx.envs.launch with the privileged YAML
#   3. post-process: for each cap-x trial dir, move the matching
#      gt_state.jsonl into it and run the offline Aspect-1 scorer
#
# Result: one directory per episode under
#   outputs/<model>/franka_robolab_privileged/trial_NN_sandboxrc_*_reward_*_taskcompleted_*/
# containing: code.py, summary.txt, raw_response.sh, all_responses.json,
# prompts_and_responses/, gt_state.jsonl, video.mp4 (if frames captured),
# task_failures.jsonl, metadata.json.
#
# Usage:
#   conda activate capx-robolab
#   ./scripts/run_capx_robolab_eval.sh [--clean] [--trials N] [--config <yaml>]

set -euo pipefail

CAP_X_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$CAP_X_ROOT"

CONFIG="env_configs/robolab/franka_robolab_privileged.yaml"
TRIALS=1
WORKERS=1
CLEAN=0
MODEL="aws/anthropic/bedrock-claude-opus-4-7"
SERVER_URL="https://inference-api.nvidia.com/v1/chat/completions"
TIMEOUT_S=1500
OUTPUT_DIR=""
SCENE_NAME_OVERRIDE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --clean) CLEAN=1; shift ;;
    --trials) TRIALS="$2"; shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    --config) CONFIG="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --server-url) SERVER_URL="$2"; shift 2 ;;
    --timeout) TIMEOUT_S="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --scene-name) SCENE_NAME_OVERRIDE="$2"; shift 2 ;;
    -h|--help)
      sed -n '2,18p' "$0"; exit 0 ;;
    *) echo "Unknown arg: $1"; exit 2 ;;
  esac
done

# Derive base output_dir from YAML config if not overridden.
if [[ -z "$OUTPUT_DIR" ]]; then
  OUTPUT_DIR=$(grep -E '^output_dir:' "$CONFIG" | head -1 | sed -E 's/output_dir:\s*//; s/^\.\///')
fi
# --scene-name CLI flag overrides the YAML; otherwise extract from YAML.
if [[ -n "$SCENE_NAME_OVERRIDE" ]]; then
  SCENE_NAME="$SCENE_NAME_OVERRIDE"
else
  SCENE_NAME=$(grep -E '^\s+scene_name:' "$CONFIG" | head -1 | awk '{print $2}')
fi
if [[ -z "$SCENE_NAME" ]]; then
  echo "[run_capx] WARNING: scene_name not found in YAML; falling back to '_unknown_scene'"
  SCENE_NAME="_unknown_scene"
fi

# Cap-x's _setup_output_dir mangles output_dir by inserting the model
# name as the SECOND-TO-LAST path component. We pre-build the per-scene
# pre-mangle path here, then compute the post-mangle path the wrapper
# uses for log + pending_gt + post-process so all artifacts end up in
# ONE tree:
#   outputs/<model>/<config>/<scene>/{launch.log, trial_NN/, ...}
PRE_MANGLE_OUTPUT_DIR="$OUTPUT_DIR/$SCENE_NAME"
MODEL_DIR_FRAGMENT="${MODEL//\//_}"
RESULTS_DIR="${OUTPUT_DIR}/${MODEL_DIR_FRAGMENT}/${SCENE_NAME}"
PENDING_GT_DIR="$RESULTS_DIR/_pending_gt"
LOG_FILE="$RESULTS_DIR/launch.log"

if [[ "$CLEAN" -eq 1 ]]; then
  echo "[run_capx] cleaning $RESULTS_DIR ..."
  rm -rf "$CAP_X_ROOT/$RESULTS_DIR" 2>/dev/null || true
  rm -rf "$CAP_X_ROOT/$PRE_MANGLE_OUTPUT_DIR" 2>/dev/null || true
fi
mkdir -p "$CAP_X_ROOT/$RESULTS_DIR"
mkdir -p "$CAP_X_ROOT/$PENDING_GT_DIR"

# Safety: never run two trials in parallel against the same output dir
# (HDF5 file-lock contention on robolab's recorder).
if pgrep -f 'capx\.envs\.launch' >/dev/null; then
  echo "[run_capx] another capx.envs.launch is running; abort." >&2
  exit 3
fi

PYTHON="${PYTHON:-/home/rescuedog/miniconda3/envs/capx-robolab/bin/python}"

echo "[run_capx] launching trial(s) → $LOG_FILE (scene=$SCENE_NAME)"
# `set -e` is on, so disable it just for the launch — timeout returns
# 124 on cap, 137 on SIGKILL, ~143 on SIGTERM, and we want to
# post-process regardless of which one fires (Isaac Sim's shutdown
# often hangs through SIGTERM).
#
# We pass --output-dir to override the YAML so cap-x's mangler produces
# `outputs/<model>/<config>/<scene>/`. CAPX_GT_STATE_DUMP_DIR points the
# adapter's per-trial gt_state.jsonl staging at the same tree.
set +e
CAPX_GT_STATE_DUMP_DIR="$CAP_X_ROOT/$PENDING_GT_DIR" \
CAPX_SCENE_NAME="$SCENE_NAME" \
timeout --kill-after=30 "$TIMEOUT_S" "$PYTHON" -m capx.envs.launch \
    --config-path "$CONFIG" \
    --output-dir "$PRE_MANGLE_OUTPUT_DIR" \
    --server-url "$SERVER_URL" \
    --api-key "${NVIDIA_API_KEY:-}" \
    --model "$MODEL" \
    --visual-differencing-model-api-key "${NVIDIA_API_KEY:-}" \
    --total-trials "$TRIALS" \
    --num-workers "$WORKERS" \
    > "$LOG_FILE" 2>&1
LAUNCH_RC=$?
set -e
echo "[run_capx] cap-x launcher exit=$LAUNCH_RC"

# Belt-and-braces: if any python child survived (Isaac Sim destructor
# can ignore SIGTERM), kill it explicitly so post-process can proceed.
pkill -KILL -f 'capx\.envs\.launch' 2>/dev/null || true
sleep 2

# Post-process: move pending gt_state files into their trial dirs and
# run the Aspect-1 scorer per-dir.  Each trial dir is named
# ``trial_<NN>_sandboxrc_*_reward_*_taskcompleted_*``; pending files
# are ``trial_<NN>.jsonl`` (matching ``<NN>``).
if [[ -d "$RESULTS_DIR" ]]; then
  echo "[run_capx] post-processing trial dirs under $RESULTS_DIR"
  shopt -s nullglob

  # The adapter monkey-patches cap-x to write each trial to a stable
  # ``trial_<NN>`` dir. We also support the upstream
  # ``trial_<NN>_sandboxrc_*`` pattern for back-compat (e.g. when running
  # without our adapter's bootstrap).
  for trial_dir in "$RESULTS_DIR"/trial_*; do
    [[ -d "$trial_dir" ]] || continue
    base=$(basename "$trial_dir")
    nn=$(echo "$base" | sed -E 's/^trial_([0-9]+).*/\1/')
    pending="$PENDING_GT_DIR/trial_${nn}.jsonl"
    if [[ -f "$pending" ]]; then
      mv "$pending" "$trial_dir/gt_state.jsonl"
      echo "[run_capx]   moved gt_state for trial $nn → $trial_dir"
    else
      echo "[run_capx]   no pending gt_state for trial $nn"
    fi

    # (video_native.mp4 disabled — set CAPX_RECORD_NATIVE_VIDEO=1 in
    # the launch env to re-enable per the adapter's flag.)
    pending_native_video="$PENDING_GT_DIR/trial_${nn}_native.mp4"
    if [[ -f "$pending_native_video" ]]; then
      mv "$pending_native_video" "$trial_dir/video_native.mp4"
    fi

    # Per-trial Aspect-1 scoring lands task_failures.jsonl + metadata.json
    # right next to code.py / summary.txt / video.mp4.
    if [[ -f "$trial_dir/gt_state.jsonl" ]]; then
      "$PYTHON" "$CAP_X_ROOT/scripts/score_capx_with_aspect1.py" \
        --capx-output-dir "$trial_dir" \
        --out-root "$trial_dir/_aspect1_tmp" >> "$LOG_FILE" 2>&1 || true
      # Flatten: scorer writes to <out-root>/<task_slug>/episode_000/{*}.
      # We want those two files at trial_dir level.
      ep_dir=$(find "$trial_dir/_aspect1_tmp" -type d -name 'episode_*' | head -1)
      if [[ -n "$ep_dir" ]]; then
        mv "$ep_dir/task_failures.jsonl" "$trial_dir/task_failures.jsonl" 2>/dev/null || true
        mv "$ep_dir/metadata.json"        "$trial_dir/metadata.json"      2>/dev/null || true
      fi
      rm -rf "$trial_dir/_aspect1_tmp" 2>/dev/null || true
    fi
  done
fi

# LOG_FILE was redirected directly into RESULTS_DIR — no copy needed.

# Drop the empty staging dir if nothing's left in it.
rmdir "$CAP_X_ROOT/$PENDING_GT_DIR" 2>/dev/null || true

# Remove any stale legacy pre-mangle dir (left over from older runs
# that staged log + pending_gt outside the model tree).
[[ -d "$CAP_X_ROOT/$PRE_MANGLE_OUTPUT_DIR" ]] && \
    rmdir "$CAP_X_ROOT/$PRE_MANGLE_OUTPUT_DIR" 2>/dev/null || true

echo "[run_capx] done. Per-episode artifacts in $RESULTS_DIR/trial_*/"
exit "$LAUNCH_RC"
