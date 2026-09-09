from dataclasses import dataclass, field
import numpy as np
import torch


def _interleave_bits(ix: np.ndarray, iy: np.ndarray, iz: np.ndarray, bits: int) -> np.ndarray:
    """Bit-interleaved (Morton) code from three `bits`-wide integer indices.
    Gives a natural 1D space-filling order used later as the RoPE position,
    and has the convenient property that the level-(l-1) parent of a
    level-l box is simply `code >> 3`.
    """
    code = np.zeros_like(ix, dtype=np.int64)
    for b in range(bits):
        code |= ((ix >> b) & 1).astype(np.int64) << (3 * b + 2)
        code |= ((iy >> b) & 1).astype(np.int64) << (3 * b + 1)
        code |= ((iz >> b) & 1).astype(np.int64) << (3 * b)
    return code


@dataclass
class LevelInfo:
    codes: np.ndarray  # (n_boxes,) sorted ascending Morton codes, occupied boxes only
    code_to_row: dict
    positions: torch.Tensor  # (n_boxes,) RoPE position = Morton code itself
    parent_row: torch.Tensor | None = None  # (n_boxes,) row into level l-1, absent at root
    u_target_row: torch.Tensor | None = None  # (K,) M2L target rows at this level
    u_source_row: torch.Tensor | None = None  # (K,) M2L source rows at this level
    leaf_atom_row: torch.Tensor | None = None  # (N,) only set at the leaf level


class Octree:
    depth: int
    levels: list = field(default_factory=list)  # LevelInfo per level, index 0 = root

    def leaf(self) -> LevelInfo:
        return self.levels[self.depth]


def _grid_indices(positions: torch.Tensor, cell: torch.Tensor, grid_size: int) -> np.ndarray:
    inv_cell = torch.linalg.inv(cell)
    frac = positions.detach() @ inv_cell
    frac = frac - torch.floor(frac)
    idx = torch.floor(frac * grid_size).long().clamp_(0, grid_size - 1)
    return idx.cpu().numpy()


def _near_neighbor_codes(ix: int, iy: int, iz: int, grid_size: int, bits: int) -> list[int]:
    out = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                nx = (ix + dx) % grid_size
                ny = (iy + dy) % grid_size
                nz = (iz + dz) % grid_size
                out.append(int(_interleave_bits(np.array([nx]), np.array([ny]), np.array([nz]), bits)[0]))
    return out


def build_octree(positions: torch.Tensor, cell: torch.Tensor, depth: int) -> Octree:
    """Build a periodic (torus) sparse octree over the fractional coordinates
    of `positions` inside `cell`, down to `depth` levels (root = level 0,
    leaves = level `depth`). Only occupied boxes are stored at every level.

    Box-tree bookkeeping (which atom sits in which box, which boxes are
    "near"/"far") is inherently a discontinuous function of the atomic
    positions, so it is built with `positions.detach()` and is NOT part of
    the autograd graph -- forces still flow correctly through every tensor
    *value* carried on the tree (the local-GNN features that seed the leaves,
    and every learned M2M/M2L/L2L map), just not through which box an atom
    is assigned to. See ARCHITECTURE.md for the practical consequence
    (small force discontinuities as atoms cross box boundaries) and the
    smooth-partition follow-up noted there.
    """
    n_atoms = positions.shape[0]
    device = positions.device

    leaf_idx = _grid_indices(positions, cell, 2 ** depth)  # (N, 3)
    leaf_codes = _interleave_bits(leaf_idx[:, 0], leaf_idx[:, 1], leaf_idx[:, 2], depth)

    tree = Octree(depth=depth)
    child_row_of: dict[int, int] | None = None
    child_codes_prev: np.ndarray | None = None

    for l in range(depth, -1, -1):
        shift = 3 * (depth - l)
        codes_at_l = leaf_codes >> shift
        uniq_codes, inverse = np.unique(codes_at_l, return_inverse=True)
        code_to_row = {int(c): i for i, c in enumerate(uniq_codes)}

        info = LevelInfo(
            codes=uniq_codes,
            code_to_row=code_to_row,
            positions=torch.as_tensor(uniq_codes, device=device, dtype=torch.float32),
        )

        if l == depth:
            info.leaf_atom_row = torch.as_tensor(inverse, device=device, dtype=torch.long)

        if l < depth:
            # child_codes_prev are level (l+1) codes in *their own row order*;
            # parent row for each child = row of (child_code >> 3) at this level
            parent_codes = child_codes_prev >> 3
            parent_rows = np.array([code_to_row[int(c)] for c in parent_codes], dtype=np.int64)
            tree.levels[-1].parent_row = torch.as_tensor(parent_rows, device=device, dtype=torch.long)

        child_codes_prev = uniq_codes
        tree.levels.append(info)

        grid_size = 2 ** l
        if grid_size >= 4:  # far-field only well-defined once >=3 boxes/axis
            targets, sources = [], []
            for code in uniq_codes:
                ix = 0
                iy = 0
                iz = 0
                c = int(code)
                for b in range(l):
                    iz |= (c & 1) << b
                    c >>= 1
                    iy |= (c & 1) << b
                    c >>= 1
                    ix |= (c & 1) << b
                    c >>= 1
                near_self = set(_near_neighbor_codes(ix, iy, iz, grid_size, l)) & set(code_to_row)

                # parent's near-neighbors, then all occupied children of those
                # (this level's array doesn't know parents yet within this loop
                # since we haven't ascended -- recompute parent index directly)
                pix, piy, piz = ix >> 1, iy >> 1, iz >> 1
                parent_grid = grid_size // 2
                parent_near = _near_neighbor_codes(pix, piy, piz, parent_grid, l - 1)
                u_set = set()
                for pcode in parent_near:
                    child_base = pcode << 3
                    for child_offset in range(8):
                        cc = child_base | child_offset
                        if cc in code_to_row:
                            u_set.add(cc)
                u_set -= near_self
                u_set.discard(int(code))

                tgt_row = code_to_row[int(code)]
                for scode in u_set:
                    targets.append(tgt_row)
                    sources.append(code_to_row[scode])

            info.u_target_row = torch.as_tensor(targets, device=device, dtype=torch.long)
            info.u_source_row = torch.as_tensor(sources, device=device, dtype=torch.long)

    tree.levels.reverse()  # index 0 = root, index depth = leaf
    return tree
