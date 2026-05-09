# CaP-X osmo workflows

Reference: cluster `https://us-west-2-aws.osmo.nvidia.com`,
Lustre `/mnt/amlfs-04/home/<user>/`.

## Prerequisites

- `osmo` CLI installed + logged in (`osmo login https://us-west-2-aws.osmo.nvidia.com`)
- `robolab` and `vlm-orchestrator` already on Lustre at the standard paths
  used by `robolab/osmo/experiments/run-eval-pi05-general.yaml`:
  `/mnt/amlfs-04/home/<user>/{robolab, vlm-orchestrator}`
- Three osmo credentials registered on your account (verify with
  `osmo credential list`): `github-pat`, `hf-token`, `nvidia-api-key`

## V1: single-task / single-episode smoke

Verifies the full pipeline (S4 + M2 + M3, **no Molmo**) on osmo.

```bash
osmo workflow submit osmo/run-capx-skill-library.yaml \
    --set-string scene_name=InferClearTableTask \
    --set trials=1
```

Per-trial outputs land on Lustre:
```
/mnt/amlfs-04/home/<user>/capx-results/<scene>/<run_ts>/
└── franka_robolab_skill_library/
    └── aws_anthropic_bedrock-claude-opus-4-7/
        └── <scene>/
            ├── launch.log
            ├── all_responses.json, summaries.txt, initial_prompt.txt
            └── trial_NN/
                ├── code.py, summary.txt, raw_response.sh
                ├── all_responses.json, prompts_and_responses/
                ├── visual_feedback_*.png
                ├── video_combined.mp4, video_turn_*.mp4
                ├── gt_state.jsonl
                ├── task_failures.jsonl
                └── metadata.json
```

Sync results to your local machine via the syncer attached to the same
osmo pool the job ran on (per the
[Osmo pools and syncers](../../vlm-orchestrator/CLAUDE.md) note).

## V2 (planned): + Molmo sidecar

Add a second task running `vllm serve allenai/Molmo-7B-D-0924
--served-model-name allenai/Molmo2-8B` on the same pod. Wire via
`{{host:molmo-server}}` so the cap-x process hits port 8122 directly.
Skipped in V1 because cap-x's molmo client falls back gracefully when
the server isn't reachable.

## V3 (planned): scene matrix

Submit one workflow per LH common-sense scene × N seeds. Each run gets
its own Lustre directory; aggregate via a post-process script that
walks `capx-results/*/*/franka_robolab_skill_library/.../trial_*/
{task_failures.jsonl, metadata.json}`.

## Notes

- Cap-x's main process and the SAM3/CGN/PyRoKi servers all share the
  one L40 GPU (~37 GB used at peak; L40 has 48 GB). Adding Molmo will
  likely require either an H100 platform or splitting servers across
  GPUs via `CUDA_VISIBLE_DEVICES`.
- `--enable-subtask` equivalent is automatic — the adapter sets
  `robolab.constants.ENABLE_SUBTASK_PROGRESS_CHECKING = True` before
  `create_env`.
- The wrapper's offline Aspect-1 scorer is invoked per-trial after the
  cap-x trial loop exits, so `task_failures.jsonl` + `metadata.json`
  always land in the trial dir before the pod terminates.
