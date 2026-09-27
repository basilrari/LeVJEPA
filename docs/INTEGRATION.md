# Grafting MotionJEPA DISReg onto LeVJEPA

One page. Code: `disreg.py`, wired from `main.py` `multiview_forward`. Off unless `disreg.enabled=true`.

## What each repo already does

LeVJEPA (`main.py` `multiview_forward`, `module.py` `SIGReg` / `VisionTransformer`, `data/loader.py`):

- One 16-frame clip becomes 1 global 224 crop and 10 local 96 crops (`conf/config.yaml`).
- Shared video ViT. Training drops 95% of patch tokens. The `[cls]` token is kept.
- Projector maps `[cls]` to 256-d. Invariance is MSE of every view to the global view, gradients through both sides.
- SIGReg (Epps–Pulley on 1024 random projections, 17 knots) on those projected clip embeddings, weight `0.02`.
- EMA of the encoder is saved as `state_dict_ema` and is not in the loss. ImageNet eval uses that copy.

There is no ImageNet probe in the upstream repo. `probe_imagenet.py` is the V-JEPA attentive probe the paper says it uses (one cross-attention query, residual, two-layer GELU MLP, linear classifier; 20 epochs, lr 0.001, weight decay 0.001, cosine to 0). Images are repeated to 8 frames so a tubelet-1 encoder is probed on 8 temporal slots, which is the paper's τ=1 eval setup.

MotionJEPA (`src/models/motionjepa/jepa.py`, built in `src/models/registry.py`):

- Separate image ViT encodes each frame. A second image ViT (DiffEnc) encodes `o_{t+1} - o_t`.
- `inv_pred` is an MLP, depth 3, hidden 512: `[z_t, z_{t+1}] -> d_hat`.
- `L = L_pred + 0.5 * MSE(d_hat, d) + 0.25 * SIGReg(z) + 2.0 * SIGReg(d)`.
- Those three coefficients are `diff_pred_weight`, `sigreg_weight`, and `diff_sigreg_weight` in `configs/pong_motionjepa_default.yaml` (same values for dino and golf). The user note "λ_d = 0.5" is the difference-prediction weight. SIGReg on `d` is 2.0, not 0.5.

## Loss that is trained

LeVJEPA is not replaced. With DISReg off the loss is unchanged:

```
L_LeVJEPA = L_inv + 0.02 * SIGReg(z_cls)
```

`L_inv` is the global/local MSE. `z_cls` is the projected clip `[cls]`.

With DISReg on, two frames are taken from the global view of the same clip. Default `disreg.gap=clip` uses the first and last frame. `disreg.gap=1` uses the first two clip frames (the loader already sampled them at `frame_stride` 2).

```
d_img = o_{t+1} - o_t          # [0, 1] pixels, uint8 / 255
z_t, z_{t+1} = Proj_frame(Enc(o))     # LeVJEPA encoder, one frame at a time, no token drop
d = Proj_diff(DiffEnc(d_img))         # separate ViT-Tiny image encoder unless disreg.share_encoder=true
d_hat = DiffPred([z_t, z_{t+1}])
L_DISReg = 0.5 * MSE(d_hat, d) + 0.25 * SIGReg(z) + 2.0 * SIGReg(d)
L = L_LeVJEPA + lambda_disreg * L_DISReg
```

`lambda_disreg` defaults to 1, so the MotionJEPA coefficients are used as released. Token dropping still applies to the 16-frame global and local views. It is turned off only for the two DISReg frame forwards, then restored. DiffEnc, both new projectors, and DiffPred are training-only. The ImageNet probe loads `encoder.*` from `state_dict_ema` and does not construct them.

Logged keys when DISReg is on: `inv`, `sigreg_loss` (clip CLS), `disreg_pred`, `sigreg_z`, `sigreg_d`, `disreg`, `loss`. `z_std` and `d_std` are there to see a constant embedding.

## Consumer run

`conf/consumer.yaml`. One GPU, ViT-Tiny, batch 128, 39063 steps (5,000,064 clips). GPU 0 is the RTX PRO 5000 Blackwell (`CUDA_VISIBLE_DEVICES=0` in `scripts/train_consumer.sh`).

```
bash scripts/train_consumer.sh
bash scripts/train_consumer.sh disreg.enabled=true run_name=disreg
```

`stable-pretraining` saves `last.ckpt` at `runs/<run_name>/<stamp>/runs/<YYYYMMDD>/<HHMMSS>/<id>/checkpoints/last.ckpt`. Resume with `resume.ckpt_path` pointed at that file. SIGReg multiplies its statistic by the batch size, same as upstream LeVJEPA and MotionJEPA, so at batch 128 the DISReg term is tens of units while the invariance MSE stays below 1. That scale is the released MotionJEPA weighting (`lambda_disreg=1`), not a divergence. Halve `disreg.lambda_disreg` only when the loss is non-finite.
