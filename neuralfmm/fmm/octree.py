import numpy as np
import torch


def _interleave_bits(ix, iy, iz, bits):
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


class LevelInfo:
    """Everything the tree passes need about the occupied boxes at one level.

    codes: (n_boxes,) sorted ascending Morton codes, occupied boxes only
    code_to_row: dict, Morton code -> row index into this level's tensors
    positions: (n_boxes,) RoPE position = Morton code itself
    parent_row: (n_boxes,) row into level l-1, None at the root
    u_target_row / u_source_row: (K,) M2L target/source rows at this level
    leaf_atom_row: (N,) only set at the leaf level
    """

    def __init__(self, codes, code_to_row, positions, parent_row=None, u_target_row=None,
                 u_source_row=None, leaf_atom_row=None):
        self.codes = codes
        self.code_to_row = code_to_row
        self.positions = positions
        self.parent_row = parent_row
        self.u_target_row = u_target_row
        self.u_source_row = u_source_row
        self.leaf_atom_row = leaf_atom_row


class Octree:
    def __init__(self, depth, levels=None):
        self.depth = depth
        self.levels = levels if levels is not None else []  # LevelInfo per level, index 0 = root

    def leaf(self):
        return self.levels[self.depth]


def _grid_indices(positions, cell, grid_size):
    inv_cell = torch.linalg.inv(cell)
    frac = positions.detach() @ inv_cell
    frac = frac - torch.floor(frac)
    idx = torch.floor(frac * grid_size).long().clamp_(0, grid_size - 1)
    return idx.cpu().numpy()


def _near_neighbor_codes(ix, iy, iz, grid_size, bits):
    out = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                nx = (ix + dx) % grid_size
                ny = (iy + dy) % grid_size
                nz = (iz + dz) % grid_size
                out.append(int(_interleave_bits(np.array([nx]), np.array([ny]), np.array([nz]), bits)[0]))
    return out


def build_octree(positions, cell, depth):
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
    leaf_idx = _grid_indices(positions, cell, 2 ** depth)  # (N, 3)
    leaf_codes = _interleave_bits(leaf_idx[:, 0], leaf_idx[:, 1], leaf_idx[:, 2], depth)

    tree = Octree(depth=depth)
    child_codes_prev = None

    for l in range(depth, -1, -1):
        shift = 3 * (depth - l)
        codes_at_l = leaf_codes >> shift
        uniq_codes, inverse = np.unique(codes_at_l, return_inverse=True)
        code_to_row = {int(c): i for i, c in enumerate(uniq_codes)}

        info = LevelInfo(
            codes=uniq_codes,
            code_to_row=code_to_row,
            positions=torch.as_tensor(uniq_codes, device=positions.device, dtype=torch.float32),
        )

        if l == depth:
            info.leaf_atom_row = torch.as_tensor(inverse, device=positions.device, dtype=torch.long)

        if l < depth:
            # child_codes_prev are level (l+1) codes in *their own row order*;
            # parent row for each child = row of (child_code >> 3) at this level
            parent_codes = child_codes_prev >> 3
            parent_rows = np.array([code_to_row[int(c)] for c in parent_codes], dtype=np.int64)
            tree.levels[-1].parent_row = torch.as_tensor(parent_rows, device=positions.device, dtype=torch.long)

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

            info.u_target_row = torch.as_tensor(targets, device=positions.device, dtype=torch.long)
            info.u_source_row = torch.as_tensor(sources, device=positions.device, dtype=torch.long)

    tree.levels.reverse()  # index 0 = root, index depth = leaf
    return tree
