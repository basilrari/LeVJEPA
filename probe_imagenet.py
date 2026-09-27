"""Frozen ImageNet attentive probe for a LeVJEPA encoder.

The upstream repo does not ship a probe. This follows the protocol named in
the paper (V-JEPA attentive probing, Bardes et al.):

- one learnable query, cross-attention over the frozen encoder tokens
- residual connection, two-layer GELU MLP, layer norm, then a linear classifier
- AdamW lr 0.001, weight decay 0.001, cosine to 0, 20 epochs, no warmup
- effective batch 1024 (V-JEPA's 8 nodes x 8 GPUs x 16). Default here is
  micro-batch 64 and grad accumulation 16 on one GPU.
- the image is repeated to 8 frames (paper section 4.3: tau=1 eval uses 8
  temporal slots)

Eval reads only the encoder. DISReg weights are ignored. The EMA copy
(``state_dict_ema``) is loaded when the checkpoint has one.

  uv run python probe_imagenet.py --init --out runs/init/probe.json
  uv run python probe_imagenet.py --ckpt runs/baseline/<stamp>/checkpoints/last.ckpt --out runs/baseline/probe.json
  uv run python probe_imagenet.py --init --limit 1000 --epochs 1 --out runs/smoke/probe.json
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2

import module as vit_models

_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


class AttentiveProbe(nn.Module):
    """V-JEPA CrossAttentionBlock + linear classifier, one query."""

    def __init__(self, embed_dim: int, num_classes: int = 1000, num_heads: int = 3):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim {embed_dim} not divisible by num_heads {num_heads}")
        self.num_heads = num_heads
        self.query = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.query, std=0.02)
        self.norm_x = nn.LayerNorm(embed_dim)
        self.q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.kv = nn.Linear(embed_dim, embed_dim * 2, bias=False)
        self.norm_q = nn.LayerNorm(embed_dim)
        hidden = embed_dim * 4
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, embed_dim),
        )
        self.fc = nn.Linear(embed_dim, num_classes)

    def _cross_attn(self, query: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        b, n, c = query.shape
        q = self.q(query).reshape(b, n, self.num_heads, c // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(tokens).reshape(b, tokens.shape[1], 2, self.num_heads, c // self.num_heads)
        kv = kv.permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        out = F.scaled_dot_product_attention(q, k, v)
        return out.transpose(1, 2).reshape(b, n, c)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        q = self.query.expand(tokens.shape[0], -1, -1)
        q = q + self._cross_attn(q, self.norm_x(tokens))
        q = q + self.mlp(self.norm_q(q))
        return self.fc(q[:, 0])


def build_encoder(args) -> nn.Module:
    factory = getattr(vit_models, args.model)
    encoder = factory(
        img_size=224,
        patch_size=16,
        num_frames=args.num_frames,
        tubelet_size=1,
        use_rope=True,
        token_drop_rate=0.0,
        attn_mode="block_causal",
    )
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    return encoder


def load_encoder_weights(encoder: nn.Module, ckpt_path: str) -> str:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "state_dict_ema" in ckpt and ckpt["state_dict_ema"]:
        source = ckpt["state_dict_ema"]
        which = "state_dict_ema"
    else:
        source = ckpt["state_dict"]
        which = "state_dict"
    stripped = {}
    for key, value in source.items():
        if key.startswith("encoder."):
            stripped[key[len("encoder.") :]] = value
    missing, unexpected = encoder.load_state_dict(stripped, strict=False)
    # RoPE models have no pos_embed. Anything else missing is a real mismatch.
    missing = [key for key in missing if key != "pos_embed"]
    if missing or unexpected:
        raise RuntimeError(f"encoder load missing={missing[:8]} unexpected={unexpected[:8]}")
    return which


def make_loader(root: str, train: bool, batch_size: int, workers: int, limit: int):
    transform = v2.Compose(
        [
            v2.RandomResizedCrop(224, interpolation=v2.InterpolationMode.BICUBIC, antialias=True)
            if train
            else v2.Resize(256, interpolation=v2.InterpolationMode.BICUBIC, antialias=True),
            v2.CenterCrop(224) if not train else nn.Identity(),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(_MEAN, _STD),
        ]
    )
    folder = ImageFolder(root, transform=transform)
    if limit > 0:
        gen = torch.Generator().manual_seed(0)
        pick = torch.randperm(len(folder), generator=gen)[:limit].tolist()
        folder = Subset(folder, pick)
    return DataLoader(
        folder,
        batch_size=batch_size,
        shuffle=train,
        num_workers=workers,
        pin_memory=True,
        drop_last=train,
    )


@torch.no_grad()
def encode(encoder: nn.Module, images: torch.Tensor, num_frames: int) -> torch.Tensor:
    video = images.unsqueeze(2).repeat(1, 1, num_frames, 1, 1)
    return encoder(video)


def evaluate(encoder, probe, loader, device, num_frames: int) -> float:
    probe.eval()
    correct = 0
    seen = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            tokens = encode(encoder, images, num_frames)
        logits = probe(tokens.float())
        correct += (logits.argmax(-1) == labels).sum().item()
        seen += labels.numel()
    probe.train()
    return 100.0 * correct / max(seen, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--init", action="store_true", help="probe a freshly initialized encoder")
    parser.add_argument("--model", default="vit_tiny")
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--train-root", default=os.environ.get("IMAGENET_TRAIN", ""))
    parser.add_argument("--val-root", default=os.environ.get("IMAGENET_VAL", ""))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--accum", type=int, default=16, help="64*16=1024 effective batch")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="subsample this many images per split; 0 = all")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if not args.train_root or not args.val_root:
        raise SystemExit("Set IMAGENET_TRAIN and IMAGENET_VAL, or pass --train-root and --val-root.")
    if bool(args.ckpt) == bool(args.init):
        raise SystemExit("Pass exactly one of --ckpt and --init.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = build_encoder(args).to(device)
    weight_source = "init"
    if args.ckpt:
        weight_source = load_encoder_weights(encoder, args.ckpt)
    probe = AttentiveProbe(encoder.embed_dim).to(device)
    train_loader = make_loader(args.train_root, True, args.batch_size, args.workers, args.limit)
    val_loader = make_loader(args.val_root, False, args.batch_size, args.workers, args.limit)
    steps_per_epoch = max(len(train_loader) // args.accum, 1)
    total_steps = steps_per_epoch * args.epochs
    opt = torch.optim.AdamW(probe.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(total_steps, 1), eta_min=0.0)

    started = time.time()
    history = []
    best = -1.0
    opt.zero_grad(set_to_none=True)
    for epoch in range(args.epochs):
        probe.train()
        micro = 0
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.no_grad(), torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                tokens = encode(encoder, images, args.num_frames)
            logits = probe(tokens.float())
            loss = F.cross_entropy(logits, labels) / args.accum
            loss.backward()
            micro += 1
            if micro % args.accum == 0:
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
        if micro % args.accum != 0:
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
        top1 = evaluate(encoder, probe, val_loader, device, args.num_frames)
        best = max(best, top1)
        history.append({"epoch": epoch, "val_top1": top1})
        print(f"epoch {epoch} val_top1 {top1:.2f} best {best:.2f}", flush=True)

    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        peak = round(torch.cuda.max_memory_allocated() / (1024**2), 1)
    else:
        gpu, peak = "cpu", 0.0
    payload = {
        "val_top1_last": history[-1]["val_top1"] if history else None,
        "val_top1_best": best if history else None,
        "history": history,
        "weight_source": weight_source,
        "ckpt": args.ckpt,
        "model": args.model,
        "num_frames": args.num_frames,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "accum": args.accum,
        "effective_batch": args.batch_size * args.accum,
        "limit": args.limit,
        "hours": round((time.time() - started) / 3600, 3),
        "gpu": gpu,
        "peak_vram_mib": peak,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({k: payload[k] for k in ("val_top1_last", "val_top1_best", "hours", "gpu")}))


if __name__ == "__main__":
    main()
