#!/usr/bin/env bash
# Consumer ViT-Tiny Walking Tours run on GPU 0 (RTX PRO 5000 Blackwell here).
# Extra args are Hydra overrides, for example:
#   bash scripts/train_consumer.sh disreg.enabled=true run_name=disreg
#   bash scripts/train_consumer.sh trainer.max_steps=50 loader.batch_size=4 run_name=smoke
#
# A full run (max_steps >= 1000) launches the ImageNet probe after training.
# If runs/probe_baseline_before_disreg exists, a DISReg launch probes the
# finished baseline checkpoint first and writes results/baseline.md.
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

RUN_NAME=baseline
MAX_STEPS=39063
DISREG=0
for arg in "$@"; do
  case "$arg" in
    run_name=*) RUN_NAME="${arg#run_name=}" ;;
    trainer.max_steps=*) MAX_STEPS="${arg#trainer.max_steps=}" ;;
    disreg.enabled=true) DISREG=1 ;;
  esac
done

AVAIL=$(df -B1 --output=avail . | tail -1 | tr -d ' ')
if [[ "$AVAIL" -lt 20000000000 ]]; then
  mkdir -p results
  printf '# BLOCKED\n\nLess than 20GB free on the repo filesystem (%s bytes). Training was not started.\n' "$AVAIL" > results/BLOCKED.md
  echo "out of disk: ${AVAIL} bytes free"
  exit 1
fi

mkdir -p runs results
LOG="runs/${RUN_NAME}_console.log"
MARKER=runs/probe_baseline_before_disreg

run_train() {
  set +e
  set -o pipefail
  uv run python main.py --config-name consumer "$@" 2>&1 \
    | uv run python scripts/finish_experiment.py log-filter \
    | tee -a "$LOG"
  TRAIN_EC=$?
  set +o pipefail
  set -e
}

latest_ckpt() {
  find "runs/${RUN_NAME}" -name 'last.ckpt' -printf '%T@ %p\n' 2>/dev/null \
    | sort -n | tail -1 | cut -d' ' -f2-
}

if [[ "$DISREG" -eq 1 && -f "$MARKER" ]]; then
  rm -f "$MARKER"
  echo "PIPELINE probe baseline $(date -Is)"
  uv run python scripts/finish_experiment.py baseline-probe || echo "BASELINE_PROBE_FAILED"
fi

if [[ "$DISREG" -eq 1 ]]; then
  date -Is > results/disreg_attempted
fi

run_train "$@"
if [[ "$TRAIN_EC" -ne 0 ]]; then
  CKPT=""
  FOUND=$(latest_ckpt || true)
  if [[ -n "${FOUND}" ]]; then
    CKPT=$(realpath "$FOUND")
  fi
  if [[ "$DISREG" -eq 1 && -n "$CKPT" ]] && grep -q "non-finite" "$LOG" && [[ ! -f results/disreg_change.log ]]; then
    echo "Non-finite loss. Halved disreg.lambda_disreg from 1.0 to 0.5 and resumed from ${CKPT}." \
      | tee results/disreg_change.log
    run_train "$@" disreg.lambda_disreg=0.5 "resume.ckpt_path=${CKPT}"
  elif grep -q "OutOfMemoryError" "$LOG" && [[ ! -f "results/${RUN_NAME}_change.log" ]]; then
    echo "CUDA OOM. Retried once at loader.batch_size=64 (bf16-mixed was already on)." \
      | tee "results/${RUN_NAME}_change.log"
    if [[ -n "$CKPT" ]]; then
      run_train "$@" loader.batch_size=64 "resume.ckpt_path=${CKPT}"
    else
      run_train "$@" loader.batch_size=64
    fi
  fi
fi

if [[ "$TRAIN_EC" -ne 0 ]]; then
  echo "TRAIN_FAILED ${RUN_NAME} ${TRAIN_EC}"
  exit "$TRAIN_EC"
fi

if [[ "$MAX_STEPS" -lt 1000 ]]; then
  exit 0
fi

if [[ "$DISREG" -eq 1 ]]; then
  echo "PIPELINE probe disreg $(date -Is)"
  uv run python scripts/finish_experiment.py disreg-probe || echo "DISREG_PROBE_FAILED"
else
  echo "PIPELINE probe baseline $(date -Is)"
  uv run python scripts/finish_experiment.py baseline-probe || echo "BASELINE_PROBE_FAILED"
fi
