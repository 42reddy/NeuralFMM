import torch


def _n_repeats_for_cutoff(cell, cutoff):
    """Minimum number of periodic images needed along each lattice vector so
    that no atom pair within `cutoff` is missed, via the perpendicular-width
    of the cell along each axis (V / |a_j x a_k|)."""
    a0, a1, a2 = cell[0], cell[1], cell[2]
    vol = torch.abs(torch.dot(a0, torch.cross(a1, a2, dim=-1)))
    widths = [
        vol / torch.norm(torch.cross(a1, a2, dim=-1)).clamp_min(1e-12),
        vol / torch.norm(torch.cross(a2, a0, dim=-1)).clamp_min(1e-12),
        vol / torch.norm(torch.cross(a0, a1, dim=-1)).clamp_min(1e-12),
    ]
    return tuple(int(torch.ceil(cutoff / w).item()) for w in widths)


def periodic_neighbor_list(positions, cell, cutoff):
    """Brute-force periodic neighbor list (fine for toy/first-draft system sizes).

    Returns:
        edge_index: (2, E) long, [src, dst]  (dst receives the message from src)
        shifts: (E, 3) integer lattice shifts applied to the source atom
        vectors: (E, 3) r_ij = (positions[dst] + shifts @ cell) - positions[src]
    """
    device = positions.device
    n = positions.shape[0]
    nx, ny, nz = _n_repeats_for_cutoff(cell, cutoff)
    sx = torch.arange(-nx, nx + 1, device=device)
    sy = torch.arange(-ny, ny + 1, device=device)
    sz = torch.arange(-nz, nz + 1, device=device)
    shift_grid = torch.cartesian_prod(sx, sy, sz).to(positions.dtype)  # (S, 3)
    n_shifts = shift_grid.shape[0]

    shift_cart = shift_grid @ cell  # (S, 3)

    src = torch.arange(n, device=device).repeat_interleave(n * n_shifts)
    dst = torch.arange(n, device=device).repeat(n).repeat_interleave(n_shifts)
    shift_idx = torch.arange(n_shifts, device=device).repeat(n * n)

    src_pos = positions[src]
    dst_pos = positions[dst] + shift_cart[shift_idx]
    vec = dst_pos - src_pos
    dist = torch.norm(vec, dim=-1)

    zero_shift_id = int(((shift_grid == 0).all(dim=-1)).nonzero(as_tuple=True)[0].item())
    is_self_no_shift = (src == dst) & (shift_idx == zero_shift_id)
    mask = (dist < cutoff) & (dist > 1e-8) & (~is_self_no_shift)

    edge_index = torch.stack([src[mask], dst[mask]], dim=0)
    shifts = shift_grid[shift_idx[mask]].to(torch.long)
    vectors = vec[mask]
    return edge_index, shifts, vectors


def edge_vectors(positions, cell, edge_index, shifts):
    """Recompute r_ij for a fixed (cached) edge_index/shifts topology from
    the current `positions` -- cheap (gather + matmul) and, unlike
    `periodic_neighbor_list`, does no data-dependent boolean-mask indexing,
    so it never forces a GPU synchronize. Differentiable in `positions`, so
    forces still flow correctly; only the *topology* (which pairs count as
    neighbors) is treated as fixed, exactly like `fmm.octree`'s cached box
    assignment -- both are discontinuous functions of position that get
    built once from detached values, while every continuous quantity
    (features, distances, energies) built on top keeps flowing gradients.
    """
    src, dst = edge_index
    shift_cart = shifts.to(positions.dtype) @ cell
    return (positions[dst] + shift_cart) - positions[src]
