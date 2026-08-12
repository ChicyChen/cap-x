#!/usr/bin/env bash
# Molmo2-8B for CaP-X on siyi-hugo (L40S, driver 550 => CUDA 12.4 max).
#
# vLLM 0.15.1 (torch 2.9.1+cu128). Newer vllm ships cu130, which driver 550
# cannot run: "The NVIDIA driver on your system is too old (found version 12040)".
#
# Sizing found empirically on this box:
#   --max-num-batched-tokens must EXCEED max_tokens_per_mm_item (4067), else
#     vLLM refuses to start when chunked MM input is disabled. 8192 gives room.
#   The dev box's 32768 is too large here: it sized an activation buffer that
#     produced "Available KV cache memory: -9.15 GiB".
#   --gpu-memory-utilization 0.55 (~36 GB) leaves room for SAM3 + ContactGraspNet.
export VLLM_USE_V1=0
export HF_HUB_DISABLE_XET=1
exec /tmp/molmo_venv3/bin/vllm serve allenai/Molmo2-8B \
  --host 127.0.0.1 --port 8122 --trust-remote-code --enforce-eager \
  --max-model-len 4096 --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.55
