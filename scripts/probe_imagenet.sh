#!/usr/bin/env bash
# Frozen attentive probe. Reads IMAGENET_TRAIN / IMAGENET_VAL from .env.
#   bash scripts/probe_imagenet.sh --init --out runs/init/probe.json
#   bash scripts/probe_imagenet.sh --ckpt runs/baseline/<stamp>/checkpoints/last.ckpt --out runs/baseline/probe.json
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
exec uv run python probe_imagenet.py "$@"
