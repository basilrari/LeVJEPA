# DISReg

Same ViT-Tiny consumer recipe as the baseline, with MotionJEPA DISReg added. LeVJEPA SIGReg stays on. Default weights: lambda_disreg 1, lambda_z 0.25, lambda_pred 0.5, lambda_sigreg_d 2.

## Training

- run dir: `runs/disreg/2026-10-02_05-26-23`
- checkpoint: `runs/disreg/2026-10-02_05-26-23/runs/20261002/052624/e7fd109cbc79/checkpoints/last.ckpt`
- global_step: 39063 / max_steps: 39063
- gpu: NVIDIA RTX PRO 5000 Blackwell
- peak allocated VRAM: 19810.6 MiB
- wall hours (run-dir stamp to last checkpoint): 32.02
- seed: 0
- batch size: 128
- model: vit_tiny
- precision: bf16-mixed
- frames in the Lance store: 776576 (10 videos, not resampled)
- last epoch loss: 1.793
- last epoch pred_loss: 0.129
- last epoch sigreg_loss: 4.467
- last epoch cls_std: 0.944
- last epoch disreg: 1.575
- last epoch disreg_pred: 0.142
- last epoch sigreg_z: 0.701
- last epoch sigreg_d: 0.664
- last epoch z_std: 0.994
- last epoch d_std: 0.993

## ImageNet attentive probe

- file: `results/disreg_probe.json`
- val_top1_best: 29.45
- val_top1_last: 29.45
- weight_source: state_dict_ema
- epochs: 20
- micro-batch: 64 x accum 16 = 1024
- limit: 0 (0 means the full split)
- probe hours: 74.72
- gpu: NVIDIA RTX PRO 5000 Blackwell
- peak allocated VRAM: 1399.0 MiB

