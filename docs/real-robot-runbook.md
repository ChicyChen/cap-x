# Real-robot runbook: CaP-X (and TiPToP) on the franky_service Franka cell

How to bring up the CaP-X real-robot baseline, what every knob is for, and the
failure modes that have actually bitten us. The TiPToP half of the same cell is
documented in `tiptop/docs/real-robot-runbook.md` in the tiptop repo; both
baselines run **simultaneously** on one machine and one GPU.

Everything below was verified on `siyi-hugo` (NVIDIA L40S, 47.5 GB, driver 550).

---

## 1. Endpoints

| baseline | driver connects to | dashboard (browser) |
|---|---|---|
| **CaP-X** | `ws://<host>:8041/` | `http://<host>:8300/` |
| **TiPToP** | `ws://<host>:8042/` | `http://<host>:8302/` |

`8041`/`8042` are **WebSocket** ports. Opening them in a browser returns
`HTTP 426 Upgrade Required` — that is correct behaviour, not a fault. Only
`8300`/`8302` are browsable.

Supporting services (internal): SAM3 `8114`, ContactGraspNet `8115`,
PyRoKi `8116`, Molmo2-8B `8122`, M2T2 `8123`.

---

## 2. Bring-up, from cold

Molmo takes ~5 min to load, so start it first and wait for `:8122`.

```bash
tmux new-session -d -s molmo  "~/baseline-launchers/start_molmo.sh > /tmp/molmo.log 2>&1"
# wait for :8122 to listen, then
tmux new-session -d -s capx   "~/baseline-launchers/run_capx.sh   > /tmp/capx.log   2>&1"
tmux new-session -d -s tiptop "PLAN_ONLY=false TIPTOP_PORT=8042 TIPTOP_MONITOR_PORT=8302 \
                               ~/baseline-launchers/run_tiptop.sh > /tmp/tiptop.log 2>&1"
```

CaP-X needs ~5 min more for SAM3 + ContactGraspNet + PyRoKi. Wait for three
`API server on 127.0.0.1:81xx is ready` lines in `/tmp/capx.log`.

> **Each `tmux new-session` must be its own SSH invocation.** Combining a
> `pkill` and a `new-session` in one remote command kills the session you just
> created. This cost several confusing "the ports are down again" cycles.

### Shutdown

```bash
tmux kill-session -t capx;   pkill -9 -f 'real_launch|sam3_server|graspnet_server|pyroki_server'
tmux kill-session -t tiptop; pkill -9 -f serve_franky
tmux kill-session -t m2t2;   pkill -9 -f m2t2_server
tmux kill-session -t molmo;  pkill -9 -f 'vllm serve'
```

---

## 3. THE most common failure: the driver never reconnects

`openpi_client` connects **once** in `__init__`; `policy.reset()` only clears
local chunk state and never re-opens the socket. So:

> **After ANY server restart you must restart the driver PROCESS on the robot
> host — not just start a new episode.**

Symptom when you forget: the driver reports

```
Episode thread error: no close frame received or sent
```

and the dashboard stays empty. That message means "my socket died", nothing more;
it is **not** diagnostic of the cause. Always check `/tmp/capx.log` for the real
reason before assuming a network fault.

---

## 4. Molmo sizing is load-bearing

`start_molmo.sh` runs vLLM **0.15.1** with:

```
VLLM_USE_V1=0 --max-model-len 4096 --max-num-batched-tokens 8192
--gpu-memory-utilization 0.55
```

Every one of those was forced by a real failure:

| symptom | cause | fix |
|---|---|---|
| `driver too old (found 12040)` | vllm ≥0.16 ships torch cu130; siyi-hugo has driver 550 = CUDA 12.4 | **vllm==0.15.1** (torch 2.9.1+cu128) |
| `TokenizersBackend has no attribute all_special_tokens_extended` | vllm 0.10 too OLD for Molmo2 | 0.15.1 is the only window that satisfies both |
| `Available KV cache memory: -9.15 GiB` | `--max-num-batched-tokens 32768` sized a huge activation buffer | **8192** |
| `max_tokens_per_mm_item (4067) > max_num_batched_tokens` | value too small | must EXCEED 4067 |
| CUDA device-side assert in `topk_topp_sampler` | vllm V1 engine on this driver | **`VLLM_USE_V1=0`** |
| ContactGraspNet `/plan` → HTTP 500, `CUDA out of memory ... 177 MiB free` | Molmo at `0.75` starved the rest of the stack | **0.55** |

GPU budget at steady state (models load LAZILY — a fresh start shows ~19 GB free
and settles to ~4 GB once SAM3/CGN have run once):

| service | GB |
|---|---|
| Molmo2-8B (util 0.55) | 23.3 |
| CaP-X: SAM3 + CGN + PyRoKi | ~18.7 |
| TiPToP: M2T2 + SAM2 | ~5.0 |

**If `/plan` starts returning 500s, LOWER Molmo to 0.50 — do not raise it.**

---

## 5. Model configuration

`env_configs/real/real_franky.yaml` sets `use_img_differencing: true`, so
`visual_differencing_model` is called **once per trial before code generation**.
A dead model there stalls every episode.

Two constraints that must hold together:

1. The model must be **reachable**. `bedrock-claude-opus-4-7` was returning
   `litellm.ServiceUnavailableError: BedrockException` (HTTP 503) while
   `opus-4-6` answered in ~2 s.
2. The model must be in **`VLM_MODELS`** (`capx/llm/client.py`), because
   `trial.py:699` asserts it. Changing the config without registering the model
   crashes the whole process on the first driver frame with
   `AssertionError: Image/video differencing model must be in the list of VLM
   models` — which the driver reports only as `no close frame received or sent`.

`tests/test_llm_retry.py::TestDifferencingModelIsAcceptedByTheAssertion` pins
both, so this cannot regress silently.

### LLM retry behaviour

`query_model` originally retried **forever** with 150–330 s blind sleeps, so a
provider outage was indistinguishable from a hang. It is now bounded:

| env var | default | meaning |
|---|---|---|
| `CAPX_LLM_MAX_RETRIES` | 6 | give up (loudly) after this many |
| `CAPX_LLM_RETRY_SLEEP_S` | 20 | base backoff, capped at 60 s |

Each retry is published to the dashboard as `LLM retry n/m: HTTP 503 …`, and a
final failure raises rather than hanging.

---

## 6. Output layout

```
outputs/franky_real/<model>/<stamp>/
├── episodes.jsonl              one row per plan attempt (the join key)
├── episode_007/
│   ├── episode.json            task, attempt count, timestamps
│   ├── attempt_1/              code.py, summary.txt, all_responses.json,
│   │                           prompts_and_responses/, video_*.mp4
│   └── attempt_2/
└── no_episode_trials/          trials that never received a driver frame
```

`python scripts/show_real_episodes.py` renders the mapping.

> **`reward` and `taskcompleted` in trial dir names are meaningless on
> hardware.** There is no ground-truth success signal: `task_completed()`
> returns False by construction and reward is always 0.000. Real trials are
> scored **manually by the operator**. Do not aggregate those fields.

---

## 7. Dashboard

`http://<host>:8300/` — episode-first, one row per operator episode with task,
attempt count, per-tool counts and errors. Click a row to filter the feed.

Per-tool views: the input frame **and** the annotated result (Molmo points,
SAM2/SAM3 mask overlays, ContactGraspNet candidates with the chosen grasp
ringed). Zero-result cases are red rows, e.g. `NO GRASP CANDIDATES returned`.

The dashboard keeps events **in memory**, so deleting result files does not
clear it — restart the service for a clean slate before a real session.

---

## 8. Known-bad states and what they mean

| what you see | actual cause |
|---|---|
| driver: `no close frame received or sent` | CaP-X restarted (driver must restart) **or** CaP-X crashed — check `/tmp/capx.log` |
| `/plan` 500 + `CUDA out of memory` | GPU oversubscribed; lower Molmo utilisation |
| `/plan` 500 + `device-side assert triggered` | a CUDA assert **poisons the context permanently**; the grasp server must be restarted, every later grasp fails identically |
| `Trial N timed out after 1000 seconds` | no driver frame arrived for that trial (normal between episodes) |
| `residual 3.2 rad > tol 0.01` | the arm is far from its commanded target; motion may not have executed even though the code prints success |
