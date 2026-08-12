# CaP-X + TiPToP real-robot baselines on siyi-hugo

Both run SIMULTANEOUSLY. Point the driver at whichever endpoint you want; no
restart needed to switch baselines.

| baseline | driver connects to | dashboard (browser) |
|---|---|---|
| CaP-X  | ws://10.57.232.29:8041/ | http://10.57.232.29:8300/ |
| TiPToP | ws://10.57.232.29:8042/ | http://10.57.232.29:8302/ |

8041/8042 are WEBSOCKET ports -- a browser gets HTTP 426 there, which is correct.

## Start from cold

    tmux new-session -d -s molmo  "~/baseline-launchers/start_molmo.sh > /tmp/molmo.log 2>&1"
    # wait ~5 min for :8122, then:
    tmux new-session -d -s capx   "~/baseline-launchers/run_capx.sh > /tmp/capx.log 2>&1"
    tmux new-session -d -s tiptop "PLAN_ONLY=false TIPTOP_PORT=8042 TIPTOP_MONITOR_PORT=8302 ~/baseline-launchers/run_tiptop.sh > /tmp/tiptop.log 2>&1"

Each `tmux new-session` must be its OWN ssh command -- combining a `pkill` and a
`new-session` in one remote command kills the session you just made.

## GPU budget (47.5 GB card, this is the tight part)

| service | GB |
|---|---|
| Molmo2-8B (vLLM, util 0.55) | 23.3 |
| CaP-X (SAM3 + CGN + PyRoKi) | ~18.7 |
| TiPToP M2T2 | 1.4 |
| TiPToP (SAM2 etc) | 3.6 |
| desktop daemon | 0.4 |

**--gpu-memory-utilization 0.55 is load-bearing.** 0.75 starved ContactGraspNet
("CUDA out of memory. Tried to allocate 470 MiB ... 177 MiB free" -> HTTP 500 on
/plan). 0.42 was too small ("No available memory for the cache blocks").
0.55 gives 4.56 GiB KV cache and leaves the rest alive. Verified end-to-end:
CGN returned (200, 4, 4) grasps.

Headroom is ~1 GB, so if /plan starts 500-ing again, lower Molmo to 0.50 rather
than raising it.

## Driver-side reminder

openpi_client connects ONCE and never reconnects: after ANY service restart the
driver must be restarted too.

## Restoring the VoLo stack

These baselines replaced the VoLo real-robot services. Bring them back with:

    ~/restore_volo_services.sh      # openpi 8000, orch 8001, grasp 8003, tool_chain 8011

Stop the baselines first (they hold the GPU).
