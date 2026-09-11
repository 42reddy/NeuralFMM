import math
import torch


def cosine_cutoff(r, r_cut):
    """Smooth envelope, 1 at r=0, 0 at r=r_cut, zero derivative at both ends."""
    x = (r / r_cut).clamp(max=1.0)
    return 0.5 * (torch.cos(math.pi * x) + 1.0) * (r < r_cut)


def bessel_rbf(r, r_cut, n_basis):
    """DimeNet-style radial Bessel basis: sqrt(2/r_cut) * sin(n*pi*r/r_cut) / r.

    Safe at r -> 0 (limit is n*pi/r_cut). `r_cut` may be a python float (a
    single fixed cutoff, e.g. the local encoder's neighbor-list radius) or a
    tensor broadcastable against `r.unsqueeze(-1)` (a per-row length scale,
    e.g. `fmm.operators`' per-box `box_scale`) -- `** 0.5` is used instead of
    `math.sqrt` so both cases work without a separate code path.
    """
    n = torch.arange(1, n_basis + 1, device=r.device, dtype=r.dtype)
    r = r.unsqueeze(-1)  # (..., 1)
    arg = n * math.pi * r / r_cut
    eps = 1e-8
    safe_r = torch.where(r < eps, torch.full_like(r, eps), r)
    norm = (2.0 / r_cut) ** 0.5
    out = norm * torch.sin(arg) / safe_r
    # exact r -> 0 limit for the atoms that got clamped
    limit = norm * (n * math.pi / r_cut)
    out = torch.where(r < eps, limit.expand_as(out), out)
    return out


def distance_rbf_envelope(delta, r_cut, n_basis):
    """RBF + cosine-envelope featurization of a relative vector's norm --
    the same treatment `local.encoder.PaiNNMessage` applies to interatomic
    distances (see `bessel_rbf`/`cosine_cutoff` above), giving an MLP that
    only otherwise sees the raw Cartesian components a proper
    multi-frequency basis for the distance itself instead of relying on a
    couple of raw linear inputs to resolve it (plain MLPs underfit
    high-frequency functions of their raw inputs, which matters more for a
    derivative -- a force -- than for the value itself).

    delta: (..., 3). r_cut: (...,) matching delta's leading dims, or a
    python float -- the length scale at which the envelope reaches zero.
    Returns (..., n_basis).
    """
    dist = delta.norm(dim=-1)
    r_cut_col = r_cut.unsqueeze(-1) if torch.is_tensor(r_cut) else r_cut
    rbf = bessel_rbf(dist, r_cut_col, n_basis)
    envelope = cosine_cutoff(dist, r_cut).unsqueeze(-1)
    return rbf * envelope
