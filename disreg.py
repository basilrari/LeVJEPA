"""MotionJEPA DISReg grafted onto a LeVJEPA training step.

Training-only. ImageNet eval loads the LeVJEPA encoder and never constructs
this module. See docs/INTEGRATION.md for the loss that is actually computed.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

import module as vit_models

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class DiffPredictor(nn.Module):
    """MotionJEPA ``inv_pred``: MLP on [z_t, z_{t+1}] -> d_hat.

    Depth 3, hidden 512, LayerNorm + GELU between the linear layers.
    Input is the concatenation of the two projected frame embeddings.
    """

    def __init__(self, embed_dim: int, hidden_dim: int = 512, depth: int = 3):
        super().__init__()
        layers: list[nn.Module] = []
        dim = embed_dim * 2
        for _ in range(max(depth - 1, 0)):
            layers.extend(
                [nn.Linear(dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()]
            )
            dim = hidden_dim
        layers.append(nn.Linear(dim, embed_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z_t: torch.Tensor, z_tp1: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z_t, z_tp1], dim=-1))


def _as_unit_interval(frames: torch.Tensor) -> torch.Tensor:
    """uint8 clip frames -> float in [0, 1]. Float input is returned as float."""
    if frames.dtype == torch.uint8:
        return frames.to(dtype=torch.float32).div(255.0)
    return frames.float()


def _imagenet_normalize(unit: torch.Tensor) -> torch.Tensor:
    """[0, 1] frames -> ImageNet norm. Accepts (B, C, H, W) or (B, T, C, H, W)."""
    if unit.ndim == 4:
        view = (1, -1, 1, 1)
    elif unit.ndim == 5:
        view = (1, 1, -1, 1, 1)
    else:
        raise ValueError(f"expected 4D or 5D frames, got {tuple(unit.shape)}")
    mean = unit.new_tensor(_IMAGENET_MEAN).view(*view)
    std = unit.new_tensor(_IMAGENET_STD).view(*view)
    return (unit - mean) / std


def frame_pair(frames: torch.Tensor, gap: str | int) -> tuple[torch.Tensor, torch.Tensor]:
    """Pick (o_t, o_{t+1}) from a clip (B, T, C, H, W).

    ``gap`` ``1`` / ``"1"``: first two clip frames (consecutive in the clip,
    which is already temporally strided by the loader).
    ``"clip"``: first and last frame of the clip.
    """
    if frames.ndim != 5 or frames.shape[1] < 2:
        raise ValueError(f"DISReg needs a clip (B, T>=2, C, H, W), got {tuple(frames.shape)}")
    key = str(gap)
    if key in {"1", "consecutive"}:
        return frames[:, 0], frames[:, 1]
    if key in {"clip", "ends"}:
        return frames[:, 0], frames[:, -1]
    raise ValueError(f"disreg.gap must be '1' or 'clip', got {gap!r}")


class DISReg(nn.Module):
    """Difference-image prediction plus SIGReg on frame and difference embeddings.

    ``diff_encoder`` is a separate image ViT (MotionJEPA's DiffEnc). When
    ``share_encoder`` is set, ``diff_encoder`` is None and the LeVJEPA encoder
    encodes the difference image as well.
    """

    def __init__(
        self,
        frame_projector: nn.Module,
        diff_projector: nn.Module,
        predictor: DiffPredictor,
        *,
        diff_encoder: nn.Module | None = None,
        lambda_disreg: float = 1.0,
        lambda_z: float = 0.25,
        lambda_pred: float = 0.5,
        lambda_sigreg_d: float = 2.0,
        gap: str = "clip",
    ):
        super().__init__()
        self.diff_encoder = diff_encoder
        self.frame_projector = frame_projector
        self.diff_projector = diff_projector
        self.predictor = predictor
        self.lambda_disreg = float(lambda_disreg)
        self.lambda_z = float(lambda_z)
        self.lambda_pred = float(lambda_pred)
        self.lambda_sigreg_d = float(lambda_sigreg_d)
        self.gap = gap
        self.share_encoder = diff_encoder is None

    def halve_outer_weight(self) -> float:
        self.lambda_disreg *= 0.5
        return self.lambda_disreg

    def training_loss(
        self,
        encoder: nn.Module,
        sigreg: nn.Module,
        frames: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """DISReg on one clip batch. Does not touch the clip-level LeVJEPA loss.

        Returns ``lambda_disreg * (lambda_pred * MSE + lambda_z * SIGReg(z) + lambda_sigreg_d * SIGReg(d))``.
        """
        o_t, o_tp1 = frame_pair(frames, self.gap)
        unit_t = _as_unit_interval(o_t)
        unit_tp1 = _as_unit_interval(o_tp1)
        # MotionJEPA: d_img = o_{t+1} - o_t. Walking Tours frames are uint8,
        # so the subtraction is in [0, 1] pixel space, not ImageNet-normalized space.
        d_img = unit_tp1 - unit_t
        b, c, h, w = unit_t.shape
        pair = _imagenet_normalize(torch.stack([unit_t, unit_tp1], dim=1))  # (B, 2, C, H, W)
        z_in = pair.reshape(b * 2, c, h, w).unsqueeze(2)  # (2B, C, 1, H, W)

        prev_drop = getattr(encoder, "token_drop_rate", 0.0)
        encoder.token_drop_rate = 0.0
        try:
            z_tokens = encoder(z_in)
            if self.share_encoder:
                d_tokens = encoder(d_img.unsqueeze(2))
            else:
                d_tokens = self.diff_encoder(d_img)
        finally:
            encoder.token_drop_rate = prev_drop

        z = self.frame_projector(z_tokens[:, 0]).view(b, 2, -1)
        d = self.diff_projector(d_tokens[:, 0])  # (B, D)
        d_hat = self.predictor(z[:, 0], z[:, 1])

        disreg_pred = F.mse_loss(d_hat, d)
        # SIGReg expects (T, B, D), matching MotionJEPA's emb.transpose(0, 1).
        sigreg_z = sigreg(z.transpose(0, 1))
        sigreg_d = sigreg(d.unsqueeze(0))
        weighted = (
            self.lambda_pred * disreg_pred
            + self.lambda_z * sigreg_z
            + self.lambda_sigreg_d * sigreg_d
        )
        total = self.lambda_disreg * weighted
        parts = {
            "disreg_pred": disreg_pred,
            "sigreg_z": sigreg_z,
            "sigreg_d": sigreg_d,
            "disreg": total,
            "z_std": z.detach().std(),
            "d_std": d.detach().std(),
        }
        return total, parts


def build_disreg(cfg, embed_dim: int) -> DISReg | None:
    """Build the training-only head. ``embed_dim`` is the LeVJEPA encoder width."""
    block = cfg.get("disreg")
    if block is None or not bool(block.get("enabled", False)):
        return None
    share = bool(block.get("share_encoder", False))
    out_dim = int(cfg.projector.output_dim)
    hidden = int(cfg.projector.hidden_dim)
    diff_encoder = None
    if not share:
        diff_encoder = vit_models.vit_tiny(
            img_size=int(cfg.model.img_size),
            patch_size=int(cfg.model.patch_size),
            num_frames=1,
            tubelet_size=1,
            use_rope=True,
            token_drop_rate=0.0,
            attn_mode="full",
        )
    frame_projector = vit_models.Projector(
        input_dim=embed_dim,
        hidden_dim=hidden,
        output_dim=out_dim,
        norm_layer=nn.BatchNorm1d,
    )
    diff_in = embed_dim if share else diff_encoder.embed_dim
    diff_projector = vit_models.Projector(
        input_dim=diff_in,
        hidden_dim=hidden,
        output_dim=out_dim,
        norm_layer=nn.BatchNorm1d,
    )
    return DISReg(
        frame_projector=frame_projector,
        diff_projector=diff_projector,
        predictor=DiffPredictor(out_dim),
        diff_encoder=diff_encoder,
        lambda_disreg=float(block.get("lambda_disreg", 1.0)),
        lambda_z=float(block.get("lambda_z", 0.25)),
        lambda_pred=float(block.get("lambda_pred", 0.5)),
        lambda_sigreg_d=float(block.get("lambda_sigreg_d", 2.0)),
        gap=block.get("gap", "clip"),
    )
