import numpy as np
import torch


def _interleave_bits(ix, iy, iz, bits):
    """Bit-interleaved (Morton) code from three `bits`-wide integer indices.
    Gives a natural 1D space-filling order for enumerating occupied boxes,
    and has the convenient property that the level-(l-1) parent of a
    level-l box is simply `code >> 3`.
    """
    code = np.zeros_like(ix, dtype=np.int64)
    for b in range(bits):
        code |= ((ix >> b) & 1).astype(np.int64) << (3 * b + 2)
        code |= ((iy >> b) & 1).astype(np.int64) << (3 * b + 1)
        code |= ((iz >> b) & 1).astype(np.int64) << (3 * b)
    return code


def _decode_grid_indices(codes, bits):
    """Inverse of `_interleave_bits`: Morton codes -> (ix, iy, iz) grid
    indices, vectorized over the whole `codes` array."""
    codes = codes.astype(np.int64)
    ix = np.zeros_like(codes)
    iy = np.zeros_like(codes)
    iz = np.zeros_like(codes)
    for b in range(bits):
        iz |= ((codes >> (3 * b)) & 1) << b
        iy |= ((codes >> (3 * b + 1)) & 1) << b
        ix |= ((codes >> (3 * b + 2)) & 1) << b
    return ix, iy, iz


class LevelInfo:
    """Everything the tree passes need about the occupied boxes at one level.

    codes: (n_boxes,) sorted ascending Morton codes, occupied boxes only
    code_to_row: dict, Morton code -> row index into this level's tensors
    centers: (n_boxes, 3) Cartesian box centers
    parent_row: (n_boxes,) row into level l-1, None at the root
    delta_to_parent: (n_boxes, 3) this box's center relative to its parent's
        center (Cartesian, no periodic wrap needed -- a box is always
        geometrically nested inside its parent), None at the root
    u_target_row / u_source_row: (K,) M2L target/source rows at this level
    delta_far: (K, 3) source box center relative to target box center for
        each (target, source) pair above, periodic-minimum-image wrapped
        (far-field pairs can straddle the periodic boundary)
    leaf_atom_row: (N,) only set at the leaf level
    """

    def __init__(self, codes, code_to_row, centers, parent_row=None, delta_to_parent=None,
                 u_target_row=None, u_source_row=None, delta_far=None, leaf_atom_row=None):
        self.codes = codes
        self.code_to_row = code_to_row
        self.centers = centers
        self.parent_row = parent_row
        self.delta_to_parent = delta_to_parent
        self.u_target_row = u_target_row
        self.u_source_row = u_source_row
        self.delta_far = delta_far
        self.leaf_atom_row = leaf_atom_row


class Octree:
    def __init__(self, depth, levels=None):
        self.depth = depth
        self.levels = levels if levels is not None else []  # LevelInfo per level, index 0 = root

    def leaf(self):
        return self.levels[self.depth]


def _move_level_to(level, device):
    def _t(t):
        return t.to(device) if t is not None else None

    return LevelInfo(
        codes=level.codes,
        code_to_row=level.code_to_row,
        centers=_t(level.centers),
        parent_row=_t(level.parent_row),
        delta_to_parent=_t(level.delta_to_parent),
        u_target_row=_t(level.u_target_row),
        u_source_row=_t(level.u_source_row),
        delta_far=_t(level.delta_far),
        leaf_atom_row=_t(level.leaf_atom_row),
    )


def move_tree_to(tree, device):
    """Copy an already-built Octree's tensors onto `device`, reusing the
    Python-side bookkeeping (codes, code_to_row) as-is. Cheap relative to
    `build_octree` -- no numpy rebuild, just a handful of small tensor
    transfers -- so callers should cache a tree once (e.g. per training
    sample, whose atomic positions never change across epochs) and move it
    with this instead of rebuilding from scratch on every forward pass."""
    return Octree(depth=tree.depth, levels=[_move_level_to(level, device) for level in tree.levels])


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


def atom_box_delta(positions, cell_per_atom, box_center_per_atom):
    """Differentiable, periodic-minimum-image r_i - center(box) for each atom
    i -- used by P2O/I2P (fmm/operators.py) to give those operators a real
    geometric relative vector instead of a bare position. Unlike the tree's
    box-to-box deltas (built once from detached positions at tree-build
    time), this one is recomputed from the LIVE (`requires_grad`) `positions`
    every forward pass, exactly like `local.neighbors.edge_vectors` -- the
    box assignment (which box owns atom i) is fixed/non-differentiable, but
    the *value* of r_i relative to that box's center still carries a
    gradient back to r_i.

    positions: (N, 3). cell_per_atom: (3, 3) or (N, 3, 3) (per-atom cell, for
    a batch of structures with different cells). box_center_per_atom: (N, 3)
    -- each atom's own leaf box center, already gathered via leaf_atom_row.
    """
    inv_cell = torch.linalg.inv(cell_per_atom)
    frac_atom = torch.matmul(positions.unsqueeze(-2), inv_cell).squeeze(-2)
    frac_center = torch.matmul(box_center_per_atom.unsqueeze(-2), inv_cell).squeeze(-2)
    delta_frac = frac_atom - frac_center
    delta_frac = delta_frac - torch.round(delta_frac)  # minimum image, wrap to [-0.5, 0.5)
    return torch.matmul(delta_frac.unsqueeze(-2), cell_per_atom).squeeze(-2)


def build_octree(positions, cell, depth):
    """Build a periodic (torus) sparse octree over the fractional coordinates
    of `positions` inside `cell`, down to `depth` levels (root = level 0,
    leaves = level `depth`). Only occupied boxes are stored at every level.

    This is the purely geometric half of the pipeline -- identical in spirit
    to a classical FMM/GROMACS-style tree build: LEAF ASSIGNMENT (which box
    each atom falls into, below), then, per level, NEAR/FAR CLASSIFICATION
    and BUILD FAR-FIELD INTERACTION LISTS (the U-list construction below,
    stored as `u_target_row`/`u_source_row`, alongside the periodic relative
    vector `delta_far` between each pair). None of this is learned -- only
    what flows *through* the resulting boxes (fmm/operators.py) is, and that
    flow is given real Cartesian relative vectors (`delta_to_parent`,
    `delta_far`, and `atom_box_delta` above) rather than an abstract
    positional index, so every learned operator can actually see how far
    apart two boxes (or an atom and its box) are.

    Box-tree bookkeeping (which atom sits in which box, which boxes are
    "near"/"far", and the box-to-box relative vectors below) is inherently a
    discontinuous function of the atomic positions, so it is built with
    `positions.detach()` and is NOT part of the autograd graph -- forces
    still flow correctly through every tensor *value* carried on the tree
    (the local-encoder features that seed the leaves, every learned
    P2O/O2O/O2I/I2I map, and the live atom-to-box deltas from
    `atom_box_delta`), just not through which box an atom is assigned to or
    the box centers themselves. See ARCHITECTURE.md for the practical
    consequence (small force discontinuities as atoms cross box boundaries)
    and the smooth-partition follow-up noted there.
    """
    # ---- LEAF ASSIGNMENT: which box each atom falls into ----
    leaf_idx = _grid_indices(positions, cell, 2 ** depth)  # (N, 3)
    leaf_codes = _interleave_bits(leaf_idx[:, 0], leaf_idx[:, 1], leaf_idx[:, 2], depth)

    tree = Octree(depth=depth)
    child_codes_prev = None

    for l in range(depth, -1, -1):
        shift = 3 * (depth - l)
        codes_at_l = leaf_codes >> shift
        uniq_codes, inverse = np.unique(codes_at_l, return_inverse=True)
        code_to_row = {int(c): i for i, c in enumerate(uniq_codes)}

        grid_size = 2 ** l
        gix, giy, giz = _decode_grid_indices(uniq_codes, l)
        frac_center_np = (np.stack([gix, giy, giz], axis=1) + 0.5) / grid_size  # (n_boxes, 3)
        frac_center = torch.as_tensor(frac_center_np, device=positions.device, dtype=positions.dtype)
        cart_center = frac_center @ cell

        info = LevelInfo(codes=uniq_codes, code_to_row=code_to_row, centers=cart_center)

        if l == depth:
            info.leaf_atom_row = torch.as_tensor(inverse, device=positions.device, dtype=torch.long)

        if l < depth:
            # child_codes_prev are level (l+1) codes in *their own row order*;
            # parent row for each child = row of (child_code >> 3) at this level
            parent_codes = child_codes_prev >> 3
            parent_rows = np.array([code_to_row[int(c)] for c in parent_codes], dtype=np.int64)
            parent_rows_t = torch.as_tensor(parent_rows, device=positions.device, dtype=torch.long)
            child_level = tree.levels[-1]
            child_level.parent_row = parent_rows_t
            # child is always geometrically nested inside its parent (no
            # periodic wraparound between a box and its own parent), so a
            # plain Cartesian difference is exact -- unlike delta_far below.
            child_level.delta_to_parent = child_level.centers - cart_center[parent_rows_t]

        child_codes_prev = uniq_codes
        tree.levels.append(info)

        # ---- NEAR/FAR CLASSIFICATION + BUILD FAR-FIELD INTERACTION LISTS ----
        if grid_size >= 4:  # far-field only well-defined once >=3 boxes/axis
            targets, sources = [], []
            for row, code in enumerate(uniq_codes):
                ix, iy, iz = int(gix[row]), int(giy[row]), int(giz[row])
                near_self = set(_near_neighbor_codes(ix, iy, iz, grid_size, l)) & set(code_to_row)

                # parent's near-neighbors, then all occupied children of those
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

            targets_np = np.array(targets, dtype=np.int64)
            sources_np = np.array(sources, dtype=np.int64)
            info.u_target_row = torch.as_tensor(targets_np, device=positions.device, dtype=torch.long)
            info.u_source_row = torch.as_tensor(sources_np, device=positions.device, dtype=torch.long)

            if targets_np.size > 0:
                # far-field pairs can straddle the periodic boundary (that's
                # exactly why they were found via the wrapped `% grid_size`
                # neighbor search above), so this needs a genuine minimum-image
                # wrap, unlike delta_to_parent.
                delta_frac = frac_center_np[sources_np] - frac_center_np[targets_np]
                delta_frac -= np.round(delta_frac)
                delta_far = torch.as_tensor(delta_frac, device=positions.device, dtype=positions.dtype) @ cell
            else:
                delta_far = torch.zeros(0, 3, device=positions.device, dtype=positions.dtype)
            info.delta_far = delta_far

    tree.levels.reverse()  # index 0 = root, index depth = leaf
    return tree


def merge_trees(trees):
    """Merge same-depth, per-structure Octrees into one block-diagonal
    Octree spanning a whole training batch.

    FMMBlock (fmm/operators.py) only ever touches a tree through flat
    per-level row indices (parent_row, u_target_row, u_source_row,
    leaf_atom_row) fed to index_add_/gather, and per-edge relative vectors
    (delta_to_parent, delta_far) that need no reindexing at all -- it never
    assumes those rows all come from a single structure. So concatenating N
    structures' level-l tensors and shifting structure i's row *indices* by
    the running box count from structures 0..i-1 gives one tree that
    FMMBlock.forward runs through completely unchanged, in a single batched
    pass, while every O2O/O2I/I2I gather-scatter still only ever mixes
    boxes/atoms belonging to the same original structure (no cross-structure
    edges are added).

    This is what lets NeuralFMMTree run once per training batch instead of
    once per structure.
    """
    depth = trees[0].depth
    n_levels = depth + 1

    merged_levels = []
    box_offset = [0] * n_levels

    per_level_centers = [[] for _ in range(n_levels)]
    per_level_codes = [[] for _ in range(n_levels)]
    per_level_parent_row = [[] for _ in range(n_levels)]
    per_level_delta_to_parent = [[] for _ in range(n_levels)]
    per_level_u_target = [[] for _ in range(n_levels)]
    per_level_u_source = [[] for _ in range(n_levels)]
    per_level_delta_far = [[] for _ in range(n_levels)]
    leaf_atom_rows = []

    for tree in trees:
        assert tree.depth == depth, "merge_trees requires every tree to share the same depth"
        # Snapshot the offsets from *previous* trees only: levels within
        # this tree are processed root-to-leaf below, so box_offset[l - 1]
        # would otherwise already include this same tree's own level-(l-1)
        # boxes (added a moment ago) by the time level l is reached.
        start_offset = list(box_offset)

        for l in range(n_levels):
            level = tree.levels[l]
            n_boxes = level.centers.shape[0]

            per_level_centers[l].append(level.centers)
            per_level_codes[l].append(level.codes)

            if level.parent_row is not None:
                per_level_parent_row[l].append(level.parent_row + start_offset[l - 1])
                per_level_delta_to_parent[l].append(level.delta_to_parent)  # relative vector, no offset
            if level.u_target_row is not None:
                per_level_u_target[l].append(level.u_target_row + start_offset[l])
                per_level_u_source[l].append(level.u_source_row + start_offset[l])
                per_level_delta_far[l].append(level.delta_far)  # relative vector, no offset
            if level.leaf_atom_row is not None:
                # leaf_atom_row's VALUES are box rows at this (leaf) level,
                # one entry per atom -- so offset by the leaf level's running
                # box count, not an atom count.
                leaf_atom_rows.append(level.leaf_atom_row + start_offset[l])

            box_offset[l] += n_boxes

    def _cat(chunks):
        return torch.cat(chunks, dim=0) if chunks else None

    for l in range(n_levels):
        merged_levels.append(
            LevelInfo(
                codes=np.concatenate(per_level_codes[l]),
                code_to_row=None,  # only needed while building, not by FMMBlock
                centers=_cat(per_level_centers[l]),
                parent_row=_cat(per_level_parent_row[l]),
                delta_to_parent=_cat(per_level_delta_to_parent[l]),
                u_target_row=_cat(per_level_u_target[l]),
                u_source_row=_cat(per_level_u_source[l]),
                delta_far=_cat(per_level_delta_far[l]),
                leaf_atom_row=_cat(leaf_atom_rows) if l == depth else None,
            )
        )

    return Octree(depth=depth, levels=merged_levels)
