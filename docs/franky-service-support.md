# Running CaP-X on a `franky_service` real Franka cell

CaP-X already ships real-Franka support (`docs/real-franka.md`), but it targets
[`robots_realtime`](https://github.com/uynitsuj/robots_realtime): raw-TCP
length-framed msgpack on `:9000`, different observation keys, and a control loop
that blocks until the arm converges.

Some cells instead run a `franky_service`-style driver
(`gitlab-master.nvidia.com/srl/py/franky_service`, branch
`hugohadfield/siyi_project`). That driver is shared with other systems and must
not be modified, so this support is a **separate low-level env** — it does not
reuse or modify `franka_real.py`.

**Sim/robolab behaviour is unchanged.** Everything here is new files plus one
registration line; no simulator config or shared class is touched.

---

## 1. The transport problem, and the fix

`franky_service` is a **policy client**
(`droid_plus/eval/episode_runner.py`): it sends one observation, expects an
action chunk back promptly, then executes that chunk at a fixed control rate.
It never waits for us.

CaP-X is a **closed-loop controller**: its generated Python calls
`move_to_joints_blocking(...)` and expects the arm to arrive.

Blocking inside the request handler deadlocks the two — the driver waits for
actions while CaP-X waits for the arm.

`FrankyWsLowLevel` uses the **trajectory-cursor** pattern that
vlm-orchestrator uses for its own grasp/place primitives
(`grasp/tool.py::_step_trajectory`):

```python
chunk = []
for _ in range(action_horizon):
    if cursor < len(trajectory):
        last_cmd = trajectory[cursor]; cursor += 1
    chunk.append(last_cmd)      # buffer empty -> repeat = hold position
return {"actions": np.array(chunk)}
```

A motion is interpolated into waypoints and appended to the buffer; each
observation pops the next `action_horizon` of them. `move_to_joints_blocking`
enqueues and then waits for the *buffer* to drain — so the driver keeps being
answered throughout, and CaP-X still observes motions completing.

## 2. The driver's schema (read from the driver, not the contract doc)

Source: `droid_plus/policies/pi05_with_depth.py::_build_request`. Two key names
differ from vlm-orchestrator's `eval-client-contract.md`, and the driver's own
comments say so — trusting the doc over the code is what broke the first attempt.

| key | notes |
|---|---|
| `prompt` | **sent on EVERY frame** — the task is live, not a launch arg |
| `episode_id` | plain int. **Not** `__episode_id` |
| `observation/exterior_image_1_left` | 224×224×3 uint8 |
| `observation/exterior_image_1_left_raw` | **native ZED res** — shares resolution with depth + K |
| `observation/wrist_image_left` | 224×224×3 uint8 |
| `observation/joint_position` | (7,) float64 rad |
| `observation/gripper_position` | (1,) float64, 0=open 1=closed |
| **`observation/depth_external`** | native res float32 **metres, NaN where stereo failed**. Doc says `depth_exterior_image_1_left`; every real consumer reads `depth_external` |
| `observation/camera_K` | (9,) row-major, native res |
| `observation/camera_extrinsic` | (16,) row-major, **cam→BASE**, OpenCV |
| `observation/ee_pos` | (3,) **raw flange** position, base frame |
| `observation/ee_quat` | (4,) (w,x,y,z) |

`__step` is not sent — do not require it. Both depth and episode-id spellings
are accepted.

## 3. Conversions that must not be got wrong

| quantity | rule |
|---|---|
| **gripper** | wire `0=open, 1=closed` ⇄ CaP-X `0=closed, 1=open` → **invert**, and binarise at 0.5 when commanding (matches `robolab.py`) |
| **camera extrinsic** | already OpenCV → **pass through**. The sim path applies an OpenGL→OpenCV flip; doing it again would corrupt every grasp pose |
| **depth** | metres, keep NaN. Downstream masks it explicitly (`control.py`: `~np.isnan(depth)`); filling NaN with 0.0 would put phantom geometry at the camera origin |
| **idle chunk** | repeat the last commanded pose. Zeros would command a violent move to the zero configuration |

## 4. API tier: matches the sim baseline

`env_configs/real/real_franky.yaml` uses
**`FrankaRealReducedSkillLibraryControlApi`** — already registered in
`capx/integrations/__init__.py:132`:

```python
FrankaControlApiReducedSkillLibrary(env, tcp_offset=[0.0, 0.0, -0.157], real=True)
```

| | sim baseline | real |
|---|---|---|
| perception | SAM3 text/point prompt | **same** |
| grasp synthesis | `plan_grasp` (ContactGraspNet) | **same** |
| skill library | 9 geometry helpers | **same** |
| motion | cuRobo | `solve_ik` + `move_to_joints` + `traj_plan` |
| `tcp_offset` | `[0,0,-0.107]` | `[0,0,-0.157]` |
| `real` | `False` | `True` |

The geometry difference is required (the physical gripper is 50 mm longer and
needs the `real=True` wrist correction). The motion difference is deliberate:
cuRobo on hardware needs a validated robot/world model, and the comparison
system likewise deploys its `linear` planner on this cell.

**Do not use `FrankaRealControlApi`** — it exposes only five tools
(`get_object_pose`, `sample_grasp_pose`, `goto_pose`, `open_gripper`,
`close_gripper`), a much weaker agent than the sim baseline, which would
understate CaP-X. The authors chose it to cut code-gen time for a live demo.
(`get_object_pose` is *not* privileged — it is SAM3 + depth back-projection —
so the reason to avoid it is tool count, not sim-only state.)

## 5. The task comes from the wire

`CodeExecutionEnvBase.__init__` builds `_full_prompt` **once** from YAML. Editing
that shared base would change every simulator env, so
`capx/envs/tasks/franka/franka_real_wire_task.py` subclasses it and overrides
three hooks: `reset()` (block for the first frame, adopt its `prompt`),
`_get_observation()` (re-adopt a changed `prompt`/`episode_id`), and
`task_completed()` (always `False`).

**No success signal is invented.** The comparison system's real-robot logs carry
no success field and trials are scored manually by the operator; adding automatic
success detection for the baseline only would bias the comparison.

## 6. Files

| path | role |
|---|---|
| `capx/envs/simulators/franky_ws.py` | the env: WS server + trajectory cursor + CaP-X control surface |
| `capx/envs/tasks/franka/franka_real_wire_task.py` | code env that takes its task from the wire |
| `env_configs/real/real_franky.yaml` | config |
| `tests/test_franky_ws.py` | 19 tests |
| `tests/test_franky_wire_task.py` | 15 tests |

One line added to `capx/envs/simulators/__init__.py` to register
`franky_ws_low_level`. Nothing else in the repo is modified.

## 7. Sim isolation, enforced by tests

- `franky_ws.py` imports neither `franka_real` nor `msgpack_server_client_utils`
  (checked by parsing its imports, not its comments);
- no config outside `env_configs/real/` may reference `FrankyWsLowLevel`;
- the shared `CodeExecutionEnvBase` must gain no wire hooks;
- the real code env must override only the expected members.

Full suite: 11 pre-existing failures (missing LIBERO/robosuite extras) both with
and without this work — no regression.

## 8. Running it

```bash
uv sync --active --extra contactgraspnet    # or the capx-robolab conda env
python -m capx.envs.launch --config-path env_configs/real/real_franky.yaml \
    --server-url https://inference-api.nvidia.com/v1/chat/completions \
    --api-key "$NVIDIA_API_KEY" --model aws/anthropic/bedrock-claude-opus-4-7 \
    --visual-differencing-model-api-key "$NVIDIA_API_KEY" --web-ui False
```

Then point the driver at `ws://<this-host>:8041/`. CaP-X blocks until the first
observation arrives, since that frame carries the task. SAM3 (`:8114`),
ContactGraspNet (`:8115`) and PyRoKi (`:8116`) are launched by the config and
need GPU — stop other GPU services on a shared cell.

## 8b. Real-time monitor (`capx/monitor/`)

A purpose-built, **episode-aware** dashboard for real-robot sessions. Starts
automatically with the env; override the port with `CAPX_MONITOR_PORT`
(default 8300).

```
http://<host>:8300/
```

| pane | shows |
|---|---|
| left sidebar | one row per episode: task, duration, LLM calls, code blocks, waypoints, per-tool counts, error count. Click to filter the feed. |
| main feed | task adoption, LLM query start/end, **generated code**, every tool step with its images (SAM3 masks, Molmo points, grasp renders), waypoints enqueued, errors |
| filters | everything / tools only / code only / errors only |

**Why not CaP-X's own `capx/web` UI:** it assumes one trial per session, and it
force-sets `enable_render` + `viser_debug`, which this env does not implement
(there is no simulator to render). It also routes env calls through a
single-worker executor that the blocking `move_to_joints_blocking` has not been
validated against.

**Design rules:**
- *Episode-first.* The driver increments `episode_id` per episode, so the monitor
  keys everything on it — a session is many episodes, not one run.
- *Cannot affect the robot.* `EventBus.publish` never raises and never blocks:
  each subscriber has a bounded queue and loses old events rather than stalling a
  producer on the action path. Tested with a deliberately un-`repr`-able payload
  and a 4-slot queue fed 50 events.
- *No CaP-X internals changed.* Tool steps arrive by wrapping the existing
  `capx.utils.execution_logger.log_step`, which every API already calls.
- *Stdlib only* (`http.server` + SSE), so it cannot conflict with the vLLM/torch
  pins in this env, and SSE reconnects on its own.

## 8c. Where results land (and the three numbering schemes)

Three independent counters overlap, which is genuinely confusing:

| Path element | What it counts | Matches your episodes? |
|---|---|---|
| `outputs/franky_real/<stamp>/<model>/` | one launcher invocation | no |
| `episode_NNN/` | the **operator's** episode, from the driver's `episode_id` | **yes** |
| `attempt_N/` | one CaP-X **plan attempt** (a "trial"). When a plan finishes, the runner starts another on the SAME episode | it's a sub-unit of the episode |

Run with `python -m capx.envs.real_launch` (NOT `capx.envs.launch`) to get this
layout; see `capx/envs/real_episode_runner.py`. Trials execute in a hidden
`.staging_trial_NNN/` dir and are filed under their episode afterwards, because
the `episode_id` only arrives with the driver's first frame — which happens
*inside* `reset()`. Attempt numbering is seeded from disk, so a process restart
onto the same root continues at `attempt_3` rather than colliding.

`episodes.jsonl` (written by `_record_attempt`, one row per plan attempt) is the
join key: `(episode_id, attempt) -> trial number + output dir`. Render it with:

```bash
python scripts/show_real_episodes.py          # newest run
python scripts/show_real_episodes.py --all    # every run
```

```
  episode 24  'pick up the larger yellow block'
    attempt 1  23:26:10  trial=1  trial_01_..._reward_0.000  code.py=yes
    attempt 2  23:26:10  trial=2  trial_02_..._reward_0.000
  episode 25  'pick up the smaller yellow block'
    attempt 1  23:26:10  trial=3  trial_03_..._reward_0.000
```

> **`reward` and `taskcompleted` in the folder names are meaningless on
> hardware.** There is no ground-truth success signal on the real cell:
> `task_completed()` returns False by construction and reward is always 0.000.
> Real-robot trials are scored **manually by the operator**. Do not aggregate
> those fields into a success rate.

Per-attempt artefacts inside each `trial_NN_...` dir: `code.py` (all generated
blocks), `summary.txt`, `all_responses.json`, `prompts_and_responses/`
(the exact prompt sent, including the `Goal:` line), `visual_feedback_*.png`.

## 9. Safety

CaP-X plans its own motions with no environment collision checking. Reduced
speed, clear workspace, supervision, reachable E-stop. `max_joint_step_rad`
(default 0.01) caps per-waypoint joint motion; lower it to go slower.
