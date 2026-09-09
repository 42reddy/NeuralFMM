import math

import torch

COULOMB_CONSTANT = 14.399645351950548  # eV * Angstrom / e^2  (1/(4 pi eps0))


def _reciprocal_cell(cell):
    return 2 * math.pi * torch.linalg.inv(cell).T


def ewald_matrix(positions, cell, alpha, r_cutoff, kmax=6, k_cutoff=None):
    """Dense symmetric (N, N) matrix K such that the periodic point-charge
    Coulomb energy is 0.5 * q^T K q (COULOMB_CONSTANT already folded in).

    K = K_real (erfc, direct sum over periodic images within r_cutoff)
      + K_recip (Ewald reciprocal-space sum, k-vectors within k_cutoff)
      + diag(self-energy correction, -2*alpha/sqrt(pi))
      + uniform background correction (charged-cell term, -pi/(V alpha^2))

    O(N^2) dense construction -- adequate for the toy/first-draft system
    sizes this codebase targets; swap for a neighbor-list + PME-style FFT
    reciprocal sum before scaling up.
    """
    device = positions.device
    dtype = positions.dtype
    n = positions.shape[0]
    volume = torch.abs(torch.det(cell))

    # --- real space, erfc, over periodic images ---
    a0, a1, a2 = cell[0], cell[1], cell[2]
    vol = torch.abs(torch.dot(a0, torch.cross(a1, a2, dim=-1)))
    widths = [
        vol / torch.norm(torch.cross(a1, a2, dim=-1)).clamp_min(1e-12),
        vol / torch.norm(torch.cross(a2, a0, dim=-1)).clamp_min(1e-12),
        vol / torch.norm(torch.cross(a0, a1, dim=-1)).clamp_min(1e-12),
    ]
    n_rep = [int(torch.ceil(r_cutoff / w).item()) for w in widths]
    sx = torch.arange(-n_rep[0], n_rep[0] + 1, device=device)
    sy = torch.arange(-n_rep[1], n_rep[1] + 1, device=device)
    sz = torch.arange(-n_rep[2], n_rep[2] + 1, device=device)
    shift_grid = torch.cartesian_prod(sx, sy, sz).to(dtype)  # (S, 3)
    shift_cart = shift_grid @ cell  # (S, 3)
    zero_shift = int(((shift_grid == 0).all(dim=-1)).nonzero(as_tuple=True)[0].item())

    diff = positions.unsqueeze(1) - positions.unsqueeze(0)  # (N, N, 3)
    r = diff.unsqueeze(0) + shift_cart.view(-1, 1, 1, 3)  # (S, N, N, 3)
    # torch.norm's backward Jacobian (r/|r|) is NaN exactly at r=0 (the
    # excluded self-term); even though torch.where masks it out below, the
    # NaN survives multiplication by the zero mask in the backward pass. A
    # clamped sqrt keeps the local derivative finite everywhere.
    dist = torch.sqrt((r**2).sum(dim=-1).clamp_min(1e-24))  # (S, N, N)

    eye = torch.eye(n, device=device, dtype=torch.bool)
    exclude = torch.zeros_like(dist, dtype=torch.bool)
    exclude[zero_shift] = eye
    within_cutoff = (dist < r_cutoff) & (~exclude)

    safe_dist = torch.where(within_cutoff, dist, torch.ones_like(dist))
    contrib = torch.erfc(alpha * safe_dist) / safe_dist
    contrib = torch.where(within_cutoff, contrib, torch.zeros_like(contrib))
    k_real = contrib.sum(dim=0)  # (N, N)

    # --- reciprocal space ---
    recip = _reciprocal_cell(cell)
    b0, b1, b2 = recip[0], recip[1], recip[2]
    kx = torch.arange(-kmax, kmax + 1, device=device)
    k_grid = torch.cartesian_prod(kx, kx, kx).to(dtype)
    nonzero = ~((k_grid == 0).all(dim=-1))
    k_grid = k_grid[nonzero]
    k_vecs = k_grid @ recip  # (K, 3)
    k2 = (k_vecs**2).sum(dim=-1)
    if k_cutoff is None:
        k_cutoff = 2 * alpha * kmax  # generous default tied to the requested shell
    keep = k2 < k_cutoff**2
    k_vecs = k_vecs[keep]
    k2 = k2[keep]

    kr = torch.einsum("ka,na->kn", k_vecs, positions)  # (K, N)
    prefactor = (4 * math.pi / k2) * torch.exp(-k2 / (4 * alpha**2)) / volume  # (K,)
    dphase = kr.unsqueeze(2) - kr.unsqueeze(1)  # (K, N, N)
    k_recip = torch.einsum("k,kij->ij", prefactor, torch.cos(dphase))

    self_diag = torch.full((n,), -2 * alpha / math.sqrt(math.pi), device=device, dtype=dtype)
    background = -math.pi / (volume * alpha**2)

    k_total = k_real + k_recip + torch.diag(self_diag) + background
    return COULOMB_CONSTANT * k_total


def ewald_matrix_batched(positions, cell, alpha, r_cutoff, kmax=6, k_cutoff=None):
    """Batched version of `ewald_matrix`: positions (B, N, 3), cell (B, 3, 3)
    -> (B, N, N). Requires the same atom count N across the batch (true for
    this dataset -- every structure has 192 atoms).

    Real-space image count and the reciprocal-space k-shell are each sized
    once for the whole batch (the max repeat count / a fixed k-grid) rather
    than per structure, so every structure's (S, 3) shift grid and (K, 3)
    k-grid have the same shape and can be stacked; a structure that needed
    fewer images/k-vectors than the batch max just gets extra terms that
    `within_cutoff`/`keep` zero out, so results are identical to calling the
    unbatched `ewald_matrix` on each structure -- see
    tests/test_model.py for the equivalence check.
    """
    device = positions.device
    dtype = positions.dtype
    B, n, _ = positions.shape
    volume = torch.abs(torch.linalg.det(cell))  # (B,)

    # --- real space, erfc, over periodic images ---
    a0, a1, a2 = cell[:, 0], cell[:, 1], cell[:, 2]  # (B, 3) each
    vol = torch.abs((a0 * torch.cross(a1, a2, dim=-1)).sum(dim=-1))  # (B,)
    widths = torch.stack(
        [
            vol / torch.norm(torch.cross(a1, a2, dim=-1), dim=-1).clamp_min(1e-12),
            vol / torch.norm(torch.cross(a2, a0, dim=-1), dim=-1).clamp_min(1e-12),
            vol / torch.norm(torch.cross(a0, a1, dim=-1), dim=-1).clamp_min(1e-12),
        ],
        dim=-1,
    )  # (B, 3)
    # shared, conservative repeat count: the max any structure in the batch needs
    n_rep = torch.ceil(r_cutoff / widths.clamp_min(1e-12)).long().amax(dim=0)  # (3,)
    sx = torch.arange(-n_rep[0].item(), n_rep[0].item() + 1, device=device)
    sy = torch.arange(-n_rep[1].item(), n_rep[1].item() + 1, device=device)
    sz = torch.arange(-n_rep[2].item(), n_rep[2].item() + 1, device=device)
    shift_grid = torch.cartesian_prod(sx, sy, sz).to(dtype)  # (S, 3)
    zero_shift = int(((shift_grid == 0).all(dim=-1)).nonzero(as_tuple=True)[0].item())

    shift_cart = torch.einsum("sc,bcd->bsd", shift_grid, cell)  # (B, S, 3)

    diff = positions.unsqueeze(2) - positions.unsqueeze(1)  # (B, N, N, 3)
    r = diff.unsqueeze(1) + shift_cart.view(B, -1, 1, 1, 3)  # (B, S, N, N, 3)
    dist = torch.sqrt((r**2).sum(dim=-1).clamp_min(1e-24))  # (B, S, N, N)

    eye = torch.eye(n, device=device, dtype=torch.bool)
    exclude = torch.zeros_like(dist, dtype=torch.bool)
    exclude[:, zero_shift] = eye
    within_cutoff = (dist < r_cutoff) & (~exclude)

    safe_dist = torch.where(within_cutoff, dist, torch.ones_like(dist))
    contrib = torch.erfc(alpha * safe_dist) / safe_dist
    contrib = torch.where(within_cutoff, contrib, torch.zeros_like(contrib))
    k_real = contrib.sum(dim=1)  # (B, N, N)

    # --- reciprocal space ---
    recip = 2 * math.pi * torch.linalg.inv(cell).transpose(-1, -2)  # (B, 3, 3)
    kx = torch.arange(-kmax, kmax + 1, device=device)
    k_grid = torch.cartesian_prod(kx, kx, kx).to(dtype)
    nonzero = ~((k_grid == 0).all(dim=-1))
    k_grid = k_grid[nonzero]  # (K, 3), shared across the batch
    k_vecs = torch.einsum("kc,bcd->bkd", k_grid, recip)  # (B, K, 3)
    k2 = (k_vecs**2).sum(dim=-1)  # (B, K)
    if k_cutoff is None:
        k_cutoff = 2 * alpha * kmax
    keep = (k2 < k_cutoff**2).to(dtype)  # (B, K) multiplicative mask -- keeps K fixed across the batch

    kr = torch.einsum("bka,bna->bkn", k_vecs, positions)  # (B, K, N)
    prefactor = (4 * math.pi / k2) * torch.exp(-k2 / (4 * alpha**2)) / volume.unsqueeze(-1)  # (B, K)
    prefactor = prefactor * keep
    dphase = kr.unsqueeze(3) - kr.unsqueeze(2)  # (B, K, N, N)
    k_recip = torch.einsum("bk,bkij->bij", prefactor, torch.cos(dphase))

    self_diag = torch.full((B, n), -2 * alpha / math.sqrt(math.pi), device=device, dtype=dtype)
    background = -math.pi / (volume * alpha**2)  # (B,)

    k_total = k_real + k_recip + torch.diag_embed(self_diag) + background.view(B, 1, 1)
    return COULOMB_CONSTANT * k_total
