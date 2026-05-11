# CaP-X osmo workflows

Reference: cluster `https://us-west-2-aws.osmo.nvidia.com`,
Lustre `/mnt/amlfs-04/home/<user>/`, pool `isaac-srl-l40-04`.

## Prerequisites

- `osmo` CLI installed + logged in (`osmo login https://us-west-2-aws.osmo.nvidia.com`)
- `robolab` and `vlm-orchestrator` already on Lustre at the standard paths:
  `/mnt/amlfs-04/home/<user>/{robolab, vlm-orchestrator}`
- Three osmo credentials registered (verify with `osmo credential list`):
  `github-pat`, `hf-token`, `nvidia-api-key`
- A syncer on the same pool to pull results back. Submit
  `vlm-orchestrator/osmo/syncer-isaac-srl-l40-04.yaml` if you don't have
  one running. (See `vlm-orchestrator/docs/osmo-sync-results-to-local.md`.)

## Two baselines

The same yaml supports both via the `mode` parameter; workflow names
include the mode prefix so they're easy to disambiguate in
`osmo workflow list`:

| `mode` | What runs | LLM cost / turn | GPUs / pod | Workflow name |
|---|---|--:|--:|---|
| `single` | Claude opus-4-7 only | 1× | **2** | `capx-robolab-single-<batch>` |
| `ensemble` | 9 calls (gpt-5.4 + gemini-3.1-pro + claude-opus-4-5, 3 temps each) + 1 synthesis | ~10× | **2** | `capx-robolab-ensemble-<batch>` |

Both modes run **with Molmo2-8B at inference time** by default. The
cap-x pod spawns a `vllm serve allenai/Molmo2-8B` on GPU 1
(`CUDA_VISIBLE_DEVICES=1`); Isaac Sim + SAM3 + ContactGraspNet + cap-x
agent + the LLM client all share GPU 0. Molmo weights are pre-warmed
to `$LUSTRE_DIR/hf_cache` (cached across pods).

Both baselines use cap-x's upstream defaults:
- `TRIAL_TIMEOUT_SECONDS = 1000` (1000 s wallclock per trial)
- `MAX_TRIAL_RETRIES = 1` (one shot per episode — cap-x default would be 3,
  but our framing here is **single-attempt-per-episode for cross-baseline
  parity** with orchestrator / policy-only runs)

## Submit a sweep

10 batched pods cover all 30 LH common-sense scenes (3 scenes × 3 trials per pod):

```bash
# v2: single Claude + Molmo
for i in 0 1 2 3 4 5 6 7 8 9; do
  osmo workflow submit osmo/run-capx-skill-library.yaml \
      --pool isaac-srl-l40-04 \
      --set-string mode=single batch_name=cs-batch-$i \
      --set trials=3
done

# v3: paper-faithful ensemble
for i in 0 1 2 3 4 5 6 7 8 9; do
  osmo workflow submit osmo/run-capx-skill-library.yaml \
      --pool isaac-srl-l40-04 \
      --set-string mode=ensemble batch_name=cs-batch-$i \
      --set trials=3
done
```

**`osmo` CLI quirk**: `--set-string` and `--set` each accept multiple
`key=val` pairs **space-separated under one flag**. Repeating the flag
(`--set-string A=x --set-string B=y`) silently overwrites — only the last
value sticks. Same for `--set`.

## Single-scene smoke

```bash
osmo workflow submit osmo/run-capx-skill-library.yaml \
    --pool isaac-srl-l40-04 \
    --set-string mode=single batch_name=smoke \
    --set trials=1
```

`smoke` runs just `InferClearTableTask` × 1 trial. Use to verify a yaml/code change before a full sweep.

## Output layout on Lustre

Per-pod-per-scene:

```
/mnt/amlfs-04/home/<user>/capx-results/<scene>/<run_ts>/
├── _servers/
│   ├── sam3.log, cgn.log, pyroki.log, molmo.log
└── aws_anthropic_bedrock-claude-opus-4-7/<scene>/
    ├── launch.log, all_responses.json, initial_prompt.txt
    └── trial_NN/
        ├── code.py, summary.txt, raw_response.sh
        ├── all_responses.json, prompts_and_responses/
        ├── visual_feedback_*.png
        ├── video_combined.mp4, video_turn_*.mp4
        ├── gt_state.jsonl
        ├── task_failures.jsonl
        └── metadata.json
```

`<run_ts>` is set once per pod and shared across all scenes the pod runs.
All scenes in the same batch use the same `<run_ts>`. Ensemble and
single submissions of the same scene set produce distinct `<run_ts>`
dirs (so they coexist).

`metadata.json` carries the eval verdict (`success: bool`, `num_steps`,
…); `task_failures.jsonl` carries Aspect-1 events
(`object_complete`, `wrong_object_picked`, `stuck`, `recovery`, …).

## Pulling results to local

```bash
# 1. Get a port-forward to the syncer
osmo workflow port-forward syncer-isaac-srl-l40-04-3 syncer --port 12224:22 &

# 2. Metadata-only pull (no videos — fast, ~80 s for 30 scenes)
ssh -o StrictHostKeyChecking=no -p 12224 root@localhost \
  "cd /mnt/amlfs-04/home/<user>/capx-results && \
   find . -maxdepth 3 -type d -name '<run_ts_glob>' -print0 | \
   xargs -0 -I{} find {} -type f \\( -name '*.json' -o -name '*.jsonl' \
       -o -name '*.log' -o -name '*.py' -o -name '*.txt' -o -name '*.sh' \
       -o -name '*.png' \\) -print0 | \
   tar --null -czf - -T -" | tar -xzf -

# 3. Full pull including mp4s (if you want videos — much larger):
scp -r -P 12224 root@localhost:/mnt/amlfs-04/.../capx-results/<scene>/<run_ts> ./
```

## Per-pod runtime + cost

| mode | per-trial wallclock | per-pod wallclock (3 scenes × 3 trials) | GPU-hours / pod |
|---|--:|--:|--:|
| single (no Molmo) | ~700 s | ~2 h | ~2 |
| single + Molmo | ~1100 s | ~3 h | ~6 (2 GPUs) |
| ensemble (no Molmo) | ~1500 s | ~4 h | ~4 |
| ensemble + Molmo | ~1800 s | ~5 h | ~10 (2 GPUs) |

All within osmo's default workflow timeout. 10 parallel pods → wallclock
matches single-pod time; GPU-hours scale 10×.

## Adding a new LH benchmark suite

When a new task suite (e.g., `lh_office` or `lh_kitchen`) is registered
in robolab, follow this pattern:

1. **Enumerate the scenes.** Get the task class names from
   `robolab/robolab/tasks/long_horizon/<suite>/`. Each scene = one Python
   file with one task class.

2. **Add batch arms to `osmo/run-capx-skill-library.yaml`.** Edit the
   `case "{{batch_name}}" in ...` block to add `<suite>-batch-N` arms,
   one per group of 3 scenes:

   ```yaml
   <suite>-batch-0)
     SCENE_ARR=(<Scene1> <Scene2> <Scene3>) ;;
   <suite>-batch-1)
     ...
   ```

   For 30 scenes that's 10 arms. For 24, that's 8 arms.

3. **Verify auto-registration** for the suite in
   `capx/envs/simulators/robolab.py:_bootstrap_isaac_sim()`. The current
   call is:

   ```python
   auto_register_droid_envs(task_dirs=["long_horizon/common_sense"])
   ```

   Add your suite's path to `task_dirs` if it isn't there.

4. **Smoke-test one scene × 1 trial first** before submitting 10
   parallel pods:

   ```bash
   osmo workflow submit osmo/run-capx-skill-library.yaml \
       --pool isaac-srl-l40-04 \
       --set-string mode=single batch_name=<suite>-batch-0 \
       --set trials=1
   ```

   Confirm the trial dir + `metadata.json` land on Lustre, then submit
   the full sweep.

5. **Pull + summarize.** Adapt the bash loop from
   `vlm-orchestrator/docs/osmo-sync-results-to-local.md` to the new
   `run_ts` glob.

## Wrapper-timeout / Isaac-Sim hang (fixed 2026-05-11)

Earlier sweeps lost a handful of trials per run to a wrapper SIGKILL —
the wrapper's `timeout` was firing while cap-x was stuck in Isaac Sim's
destructor (60-90 min hang at process exit on Isaac Lab 2.2.0). The
trials cap-x had already written were preserved; the in-progress trial
was lost.

Root cause was patched in `capx/envs/launch.py:main` with `os._exit(0)`
after the trial loop, which bypasses Python's atexit handlers (including
Isaac Sim's). All trial artifacts (`trial_NN/` dirs, `gt_state.jsonl`)
are flushed to disk before that point, so the force-exit is safe.

With that patch in place, the wrapper formula
`WRAPPER_TIMEOUT_S = trials × 1100 + 1200` is right-sized: it covers
`N × cap-x's 1000 s per-trial cap + small setup/safety` and no longer
needs a giant buffer for the shutdown hang.

If you later see the wrapper SIGKILL'ing cap-x mid-trial again, either
the `os._exit(0)` patch was reverted, or per-trial wallclock genuinely
exceeded cap-x's 1000 s cap (which shouldn't happen — that's enforced
by cap-x itself via SIGALRM). Inspect `launch.log` for `Trial X timed
out after N seconds` lines first.

## Notes

- HF model weights (`facebook/sam3`, `allenai/Molmo2-8B`) are pre-cached
  on Lustre at `$LUSTRE_DIR/hf_cache` on first run via
  `hf_transfer` + Xet-disabled (`HF_HUB_DISABLE_XET=1`,
  `HF_HUB_ENABLE_HF_TRANSFER=1`). First run pays ~2 min for ~20 GB;
  subsequent runs are instant cache hits.
- All `aws/anthropic/bedrock-claude-*` model IDs are accessible to our
  NVIDIA inference key. For the ensemble panel, use the *prefixed* IDs
  (`azure/openai/gpt-5.4`, `gcp/google/gemini-3.1-pro-preview`,
  `aws/anthropic/claude-opus-4-5`) — the bare upstream IDs return
  `key_model_access_denied`.
- `--enable-subtask` equivalent is automatic — the adapter sets
  `robolab.constants.ENABLE_SUBTASK_PROGRESS_CHECKING = True` before
  `create_env`.
- Aspect-1 scorer is invoked per-trial after the cap-x trial loop exits,
  so `task_failures.jsonl` + `metadata.json` always land before the pod
  terminates.
