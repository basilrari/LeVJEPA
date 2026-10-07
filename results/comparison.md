# Comparison

Frozen ImageNet attentive probe. Same probe for init, baseline, and DISReg. Encoder weights are the EMA copy when the checkpoint has `state_dict_ema`.

| run | ImageNet top-1 | train hours | GPU | peak VRAM MiB | batch | steps |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| init | not run | n/a | n/a | n/a | n/a | 0 |
| baseline | 31.05 | 32.22 | NVIDIA RTX PRO 5000 Blackwell | 12612.1 | 128 | 39063 |
| +DISReg | 29.45 | 32.02 | NVIDIA RTX PRO 5000 Blackwell | 19810.6 | 128 | 39063 |

The paper's consumer ViT-Tiny run is 8.9% at init and 25.2% after pretraining on eight Walking Tours videos. This table is the 10-video store (776,576 frames) and does not claim 25.2%.

The init row is empty because `results/init_probe.json` was not written yet. That probe was still running when this file was committed.


Rerun probes:

```bash
bash scripts/probe_imagenet.sh --init --out results/init_probe.json
bash scripts/probe_imagenet.sh --ckpt "$(find runs/baseline -name 'last.ckpt' -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)" --out results/baseline_probe.json
bash scripts/probe_imagenet.sh --ckpt "$(find runs/disreg -name 'last.ckpt' -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)" --out results/disreg_probe.json
```
