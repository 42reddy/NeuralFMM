import math
import torch


def apply_rope(x: torch.Tensor, pos: torch.Tensor, base: float = 10000.0) -> torch.Tensor:
    """Rotary position embedding applied along the last (feature) dim.

    x: (..., F) with F even
    pos: (...,) matching x's leading dims -- here, the Morton code of the box
        each row belongs to, giving the operators in fmm/blocks.py spatial
        awareness the way the paper's per-level MLPs need (Sec 3.2).
    """
    f = x.shape[-1]
    assert f % 2 == 0, "RoPE requires an even feature dimension"
    half = f // 2
    freqs = torch.exp(
        -math.log(base) * torch.arange(0, half, device=x.device, dtype=x.dtype) / half
    )
    angles = pos.unsqueeze(-1) * freqs  # (..., half)
    cos, sin = torch.cos(angles), torch.sin(angles)
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
