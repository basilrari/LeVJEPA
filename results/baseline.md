# Baseline

ViT-Tiny, batch 128, seed 0, 39063 steps, DISReg off. Data is the 10-video Walking Tours Lance store (776,576 frames).

## Smoke

- steps: 400 / 400
- gpu: NVIDIA RTX PRO 5000 Blackwell
- peak allocated VRAM: 12612.1 MiB
- last epoch loss: 0.536
- last epoch cls_std: 0.676
- checkpoints: `step-000200.ckpt` and `step-000400.ckpt` under the smoke run
- gate: finite loss, cls_std above 1e-3, checkpoint files present. Batch stayed 128. Precision was already bf16-mixed.

## Training

- run dir: `runs/baseline/2026-09-27_17-59-50`
- checkpoint: `runs/baseline/2026-09-27_17-59-50/runs/20260927/175950/38a3e7d11fc1/checkpoints/last.ckpt`
- global_step: 39063 / max_steps: 39063
- gpu: NVIDIA RTX PRO 5000 Blackwell
- peak allocated VRAM: 12612.1 MiB
- wall hours (run-dir stamp to last checkpoint): 32.22
- seed: 0
- batch size: 128
- model: vit_tiny
- precision: bf16-mixed
- frames in the Lance store: 776576 (10 videos, not resampled)
- last epoch loss: 0.191
- last epoch pred_loss: 0.112
- last epoch sigreg_loss: 3.968
- last epoch cls_std: 0.960

## ImageNet attentive probe

- file: `results/baseline_probe.json`
- val_top1_best: 31.05
- val_top1_last: 31.05
- weight_source: state_dict_ema
- epochs: 20
- micro-batch: 64 x accum 16 = 1024
- limit: 0 (0 means the full split)
- probe hours: 75.17
- gpu: NVIDIA RTX PRO 5000 Blackwell
- peak allocated VRAM: 1399.0 MiB

## Paper consumer number

The paper reports about 8.9% at initialization and 25.2% after pretraining ViT-Tiny on eight Walking Tours videos. This run uses ten videos. The probe number above is the measurement for this checkpoint, not a claim of matching 25.2%.
