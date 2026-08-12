#!/bin/bash
# CaP-X as a persistent real-robot SERVICE (Option B).
#
# FOUR services, matching osmo/run-capx-skill-library.yaml:
#   8114 SAM3 · 8115 ContactGraspNet · 8116 PyRoKi   (launched by the config)
#   8122 Molmo2-8B via vLLM                          (launched here)
# The skill-library API exposes point_prompt_molmo, so the generated code can
# call :8122 -- without it the trial dies and the driver's socket closes.
# siyi-hugo: conda is not installed here, so the interpreter comes from a pixi
# env at ~/capx-env (see ~/.pi notes). Everything else is identical to the dev box.
CAPX_PY="$HOME/.pixi/bin/pixi run --manifest-path $HOME/capx-env/pixi.toml python"
cd ~/cap-x
export CAPX_MONITOR_MODEL=aws/anthropic/bedrock-claude-opus-4-6
export NVIDIA_API_KEY=$(grep -m1 '^export NVIDIA_API_KEY' ~/.bashrc | sed 's/.*="\(.*\)"/\1/')

# ── Molmo2-8B on :8122 (idempotent: reuse if already serving) ──
if ! (echo > /dev/tcp/127.0.0.1/8122) 2>/dev/null; then
  echo "[svc] starting Molmo2-8B on :8122 ..."
  tmux kill-session -t molmo 2>/dev/null
  tmux new-session -d -s molmo \
    "HF_HUB_DISABLE_XET=1 /tmp/molmo_venv3/bin/vllm serve allenai/Molmo2-8B \
       --host 127.0.0.1 --port 8122 \
       --trust-remote-code --enforce-eager \
       --max-model-len 4096 --max-num-batched-tokens 32768 \
       --gpu-memory-utilization 0.60 \
       > /tmp/capx_molmo.log 2>&1; sleep infinity"
  for i in $(seq 1 90); do
    (echo > /dev/tcp/127.0.0.1/8122) 2>/dev/null && { echo "[svc] :8122 ready after ${i}0s"; break; }
    sleep 10
  done
else
  echo "[svc] :8122 already serving — reusing"
fi
(echo > /dev/tcp/127.0.0.1/8122) 2>/dev/null || echo "[svc] WARNING: :8122 NOT up; point_prompt_molmo will fail"

STAMP=$(date +%Y%m%d_%H%M%S)
ROOT=outputs/franky_real/$STAMP
mkdir -p "$ROOT"
echo "[svc] CaP-X real-robot service | model=aws/anthropic/bedrock-claude-opus-4-6"
echo "[svc] endpoint: ws://$(hostname -I | awk '{print $1}'):8041/"
echo "[svc] output:   $ROOT"
i=0
while true; do
  i=$((i+1))
  echo "[svc] ---- pass $i ($(date +%H:%M:%S)) ----"
  $CAPX_PY -m capx.envs.real_launch \
    --config-path env_configs/real/real_franky.yaml \
    --output-dir "$ROOT" \
    --server-url https://inference-api.nvidia.com/v1/chat/completions \
    --api-key "$NVIDIA_API_KEY" \
    --model aws/anthropic/bedrock-claude-opus-4-6 \
    --visual-differencing-model-api-key "$NVIDIA_API_KEY" \
    --total-trials 500 --num-workers 1 --web-ui False 2>&1 | tee -a /tmp/capx_live.log
  # --total-trials 500: the endpoint is a process-wide singleton, so many
  # trials inside ONE process keep port 8041 (and the 3 model servers) up
  # across episodes. With --total-trials 1 the process exited after every
  # episode, closing 8041; openpi_client never reconnects, so the driver died
  # with "no close frame received or sent".
  echo "[svc] pass $i finished — restarting in 5s (endpoint stays up)"
  sleep 5
done
