#!/usr/bin/env bash
# Two-task smoke sweep — used to verify the per-task / per-episode
# folder layout produced by run_capx_robolab_eval.sh.
#
# Layout produced:
#   outputs/franka_robolab_skill_library/<model>/<scene>/
#       launch.log
#       all_responses.json, summaries.txt, initial_prompt.txt
#       trial_01/, trial_02/   ← episodes
#
# Each trial dir contains: code.py, summary.txt, raw_response.sh,
# all_responses.json, prompts_and_responses/, visual_feedback_*.png,
# video_combined.mp4, video_turn_*.mp4, video_native.mp4,
# gt_state.jsonl, task_failures.jsonl, metadata.json.

set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$(pwd)"

# Two distinct LH common-sense tasks, each runs 2 trials.
SCENES=(InferClearTableTask SortFoodVsNonFoodTask)
TRIALS_PER_SCENE=2

for scene in "${SCENES[@]}"; do
  echo "=========================================================="
  echo "[sweep] starting $scene × $TRIALS_PER_SCENE trials"
  echo "=========================================================="
  ./scripts/run_capx_robolab_eval.sh \
      --clean \
      --config env_configs/robolab/franka_robolab_skill_library.yaml \
      --scene-name "$scene" \
      --trials "$TRIALS_PER_SCENE" \
      || echo "[sweep] WARN: $scene exited non-zero"
done

echo "=========================================================="
echo "[sweep] done. Result tree:"
find "$ROOT/outputs/franka_robolab_skill_library" -maxdepth 4 -type d 2>/dev/null \
  | sed "s|$ROOT/||"
