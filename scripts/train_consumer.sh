#!/usr/bin/env bash
# Consumer ViT-Tiny Walking Tours run on GPU 0 (RTX PRO 5000 Blackwell here).
# Extra args are Hydra overrides, for example:
#   bash scripts/train_consumer.sh disreg.enabled=true run_name=disreg
#   bash scripts/train_consumer.sh trainer.max_steps=50 loader.batch_size=4 run_name=smoke
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export LEVJEPA_DATA_ROOT="${LEVJEPA_DATA_ROOT:-data}"
export PYTHONUNBUFFERED=1

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader -i "${CUDA_VISIBLE_DEVICES}" || true

exec uv run python main.py --config-name consumer "$@"
