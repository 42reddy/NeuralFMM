import math
import torch


def cosine_cutoff(r, r_cut):
    """Smooth envelope, 1 at r=0, 0 at r=r_cut, zero derivative at both ends."""
    x = (r / r_cut).clamp(max=1.0)
    return 0.5 * (torch.cos(math.pi * x) + 1.0) * (r < r_cut)


def bessel_rbf(r, r_cut, n_basis):
    """DimeNet-style radial Bessel basis: sqrt(2/r_cut) * sin(n*pi*r/r_cut) / r.

    Safe at r -> 0 (limit is n*pi/r_cut).
    """
    n = torch.arange(1, n_basis + 1, device=r.device, dtype=r.dtype)
    r = r.unsqueeze(-1)  # (..., 1)
    arg = n * math.pi * r / r_cut
    eps = 1e-8
    safe_r = torch.where(r < eps, torch.full_like(r, eps), r)
    out = math.sqrt(2.0 / r_cut) * torch.sin(arg) / safe_r
    # exact r -> 0 limit for the atoms that got clamped
    limit = math.sqrt(2.0 / r_cut) * (n * math.pi / r_cut)
    out = torch.where(r < eps, limit.expand_as(out), out)
    return out
