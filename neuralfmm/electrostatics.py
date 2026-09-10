import math

import torch

COULOMB_CONSTANT = 14.399645351950548  # eV * Angstrom / e^2  (1/(4 pi eps0))


def _k_grid(cell, kmax, dtype, device):
    recip = 2 * math.pi * torch.linalg.inv(cell).T
    kx = torch.arange(-kmax, kmax + 1, device=device)
    k_grid = torch.cartesian_prod(kx, kx, kx).to(dtype)
    nonzero = ~((k_grid == 0).all(dim=-1))
    k_grid = k_grid[nonzero]
    return k_grid @ recip  # (K, 3)


def smoothed_kernel_matrix(positions, cell, alpha, kmax=6):
    """Per-channel, periodic, reciprocal-space-only long-range kernel: for
    each channel c, K_c is the (N, N) matrix such that 0.5 * q^T K_c q
    approximates (truncated at `kmax` reciprocal shells) the periodic
    lattice sum of erf(alpha_c * r_ij) / r_ij * q_i * q_j over i != j
    (COULOMB_CONSTANT folded in, self/i==i entries excluded).

    This is deliberately only the reciprocal-space half of a standard Ewald
    split. A *full* real+reciprocal Ewald sum reconstructs the exact 1/r
    Coulomb potential for any alpha -- there alpha is purely a numerical
    real/reciprocal convergence knob, not something that changes what is
    being summed. Dropping the real-space erfc correction and keeping only
    this smooth erf(alpha*r)/r part makes alpha a genuine, learnable
    "interaction range": alpha -> large recovers full long-range 1/r
    Coulomb, alpha -> small collapses the kernel toward zero range. That is
    what lets different latent channels, each with their own alpha, learn
    genuinely different effective ranges instead of every channel sharing
    the exact same 1/r decay -- see `neuralfmm.model.NeuralFMMLES` for how
    alpha is parameterized per channel and kept bounded so it stays
    consistent with `kmax`'s truncation accuracy.

    positions: (N, 3), cell: (3, 3), alpha: (C,) one range parameter per
    channel (> 0). Returns (C, N, N).
    """
    device, dtype = positions.device, positions.dtype
    n = positions.shape[0]
    volume = torch.abs(torch.linalg.det(cell))
    alpha = alpha.to(device=device, dtype=dtype)  # (C,)

    k_vecs = _k_grid(cell, kmax, dtype, device)  # (K, 3)
    k2 = (k_vecs**2).sum(dim=-1)  # (K,), k=0 already excluded by _k_grid

    kr = torch.einsum("ka,na->kn", k_vecs, positions)  # (K, N)
    dphase = kr.unsqueeze(2) - kr.unsqueeze(1)  # (K, N, N)
    cos_dphase = torch.cos(dphase)

    prefactor = (4 * math.pi / k2).unsqueeze(0) * torch.exp(-k2.unsqueeze(0) / (4 * alpha.unsqueeze(1) ** 2)) / volume
    k_recip = torch.einsum("ck,kij->cij", prefactor, cos_dphase)  # (C, N, N)

    self_diag = 2 * alpha / math.sqrt(math.pi)  # (C,)
    eye = torch.eye(n, device=device, dtype=dtype)
    k_total = k_recip - self_diag.view(-1, 1, 1) * eye

    return COULOMB_CONSTANT * k_total


def smoothed_kernel_matrix_batched(positions, cell, alpha, kmax=6):
    """Batched version of `smoothed_kernel_matrix`: positions (B, N, 3),
    cell (B, 3, 3), alpha (C,) -> (B, C, N, N). Requires the same atom count
    N across the batch (true for this dataset). The reciprocal-space k-shell
    is sized once for the whole batch (a fixed k-grid from `kmax`, shared
    across every structure), matching the unbatched result exactly.
    """
    device, dtype = positions.device, positions.dtype
    B, n, _ = positions.shape
    volume = torch.abs(torch.linalg.det(cell))  # (B,)
    alpha = alpha.to(device=device, dtype=dtype)  # (C,)

    recip = 2 * math.pi * torch.linalg.inv(cell).transpose(-1, -2)  # (B, 3, 3)
    kx = torch.arange(-kmax, kmax + 1, device=device)
    k_grid = torch.cartesian_prod(kx, kx, kx).to(dtype)
    nonzero = ~((k_grid == 0).all(dim=-1))
    k_grid = k_grid[nonzero]  # (K, 3), shared across the batch
    k_vecs = torch.einsum("kc,bcd->bkd", k_grid, recip)  # (B, K, 3)
    k2 = (k_vecs**2).sum(dim=-1)  # (B, K)

    kr = torch.einsum("bka,bna->bkn", k_vecs, positions)  # (B, K, N)
    dphase = kr.unsqueeze(3) - kr.unsqueeze(2)  # (B, K, N, N)
    cos_dphase = torch.cos(dphase)

    prefactor = (4 * math.pi / k2).unsqueeze(1) * torch.exp(
        -k2.unsqueeze(1) / (4 * alpha.view(1, -1, 1) ** 2)
    ) / volume.view(B, 1, 1)  # (B, C, K)
    k_recip = torch.einsum("bck,bkij->bcij", prefactor, cos_dphase)  # (B, C, N, N)

    self_diag = 2 * alpha / math.sqrt(math.pi)  # (C,)
    eye = torch.eye(n, device=device, dtype=dtype)
    k_total = k_recip - self_diag.view(1, -1, 1, 1) * eye.view(1, 1, n, n)

    return COULOMB_CONSTANT * k_total


def ewald_energy(latent_charges, kernel_matrix):
    """LES style long range energy: latent_charges (N, C) treated as C
    independent generalized "charge" channels, each summed through its OWN
    kernel slice `kernel_matrix[c]` (C, N, N) -- see
    `smoothed_kernel_matrix` -- with no cross-channel coupling (each
    channel is meant to specialize toward a different effective decay, so
    mixing them has no physical meaning) and no equilibration -- the
    latents are used directly, unlike QEq's solved charges. Returns a
    scalar: 0.5 * sum_c q_c^T K_c q_c.
    """
    return 0.5 * torch.einsum("nc,cnm,mc->", latent_charges, kernel_matrix, latent_charges)


def ewald_energy_batched(latent_charges, kernel_matrix):
    """Batched version of `ewald_energy`: latent_charges (B, N, C),
    kernel_matrix (B, C, N, N) -> per-structure energy (B,)."""
    return 0.5 * torch.einsum("bnc,bcnm,bmc->b", latent_charges, kernel_matrix, latent_charges)
