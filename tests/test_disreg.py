"""Dummy-clip check: DISReg backward works, and eval does not run DiffEnc."""

import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from disreg import DISReg, DiffPredictor  # noqa: E402
from module import Projector, SIGReg, vit_tiny  # noqa: E402


def _head(encoder, share, device):
    diff = None
    if not share:
        diff = vit_tiny(
            img_size=64,
            patch_size=16,
            num_frames=1,
            tubelet_size=1,
            use_rope=True,
            token_drop_rate=0.0,
            attn_mode="full",
        ).to(device)
    head = DISReg(
        frame_projector=Projector(encoder.embed_dim, hidden_dim=64, output_dim=32),
        diff_projector=Projector(
            encoder.embed_dim if share else diff.embed_dim,
            hidden_dim=64,
            output_dim=32,
        ),
        predictor=DiffPredictor(32, hidden_dim=64, depth=3),
        diff_encoder=diff,
        gap="clip",
    ).to(device)
    return head


def _run(share: bool) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    encoder = vit_tiny(
        img_size=64,
        patch_size=16,
        num_frames=4,
        tubelet_size=1,
        use_rope=True,
        token_drop_rate=0.95,
        attn_mode="block_causal",
    ).to(device)
    head = _head(encoder, share, device)
    sigreg = SIGReg(knots=17, num_proj=16).to(device)
    frames = torch.randint(0, 256, (2, 4, 3, 64, 64), dtype=torch.uint8, device=device)

    encoder.train()
    head.train()
    calls = {"n": 0}

    def hook(_module, _inputs, _output):
        calls["n"] += 1

    target = encoder if share else head.diff_encoder
    handle = target.register_forward_hook(hook)
    loss, parts = head.training_loss(encoder, sigreg, frames)
    loss.backward()
    handle.remove()
    assert torch.isfinite(loss), loss
    assert parts["z_std"] > 0, "frame embeddings collapsed to a constant"
    assert parts["d_std"] > 0, "difference embeddings collapsed to a constant"
    assert encoder.cls_token.grad is not None
    assert calls["n"] >= 1
    assert encoder.token_drop_rate == 0.95

    calls["n"] = 0
    handle = target.register_forward_hook(hook)
    encoder.eval()
    head.eval()
    # The train loop only calls training_loss while module.training is true.
    if encoder.training and head.training:
        head.training_loss(encoder, sigreg, frames)
    handle.remove()
    assert calls["n"] == 0, "DiffEnc ran under eval"


if __name__ == "__main__":
    _run(share=False)
    _run(share=True)
    print("test_disreg ok")
