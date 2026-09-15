#!/usr/bin/env bash
# Run on the RTX 3060 machine. Proves the GPU is visible from inside containers.
set -u
ok() { printf '\e[32mOK\e[0m   %s\n' "$1"; }
bad() { printf '\e[31mFAIL\e[0m %s\n' "$1"; }

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader \
  && ok "host driver" || bad "host driver: install NVIDIA drivers first"

command -v nvidia-ctk >/dev/null \
  && ok "nvidia-container-toolkit installed" \
  || bad "nvidia-container-toolkit missing (Arch: pacman -S nvidia-container-toolkit; Ubuntu: see NVIDIA docs), then: sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker"

docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi \
  && ok "GPU visible inside container" || bad "GPU not visible inside container"

# Same torch build the drone-flyby image uses (cu126 wheel, Python 3.14).
docker run --rm --gpus all python:3.14-slim sh -c \
  "pip install -q --root-user-action=ignore --index-url https://download.pytorch.org/whl/cu126 torch >/dev/null && python -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'" \
  && ok "torch.cuda works in python:3.14-slim" || bad "torch cannot see CUDA in container"

for t in sleep.target suspend.target hibernate.target hybrid-sleep.target; do
  [ "$(systemctl is-enabled $t 2>/dev/null)" = masked ] || { bad "$t not masked: sudo systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target"; break; }
done
