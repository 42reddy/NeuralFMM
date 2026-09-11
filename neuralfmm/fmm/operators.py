import torch
import torch.nn as nn

from .rope import apply_rope


def _mlp(in_dim, out_dim, depth=2):
    layers = []
    d = in_dim
    for _ in range(depth - 1):
        layers += [nn.Linear(d, out_dim), nn.SiLU()]
        d = out_dim
    layers += [nn.Linear(d, out_dim)]
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# The five learned operators, named to match the classical-FMM correspondence
# table: every analytic translation/evaluation operator is replaced by a
# small MLP, one instance per tree level (P2O and I2P are shared across
# levels since they only ever act at the leaf level). RoPE over each box's
# Morton code (fmm/rope.py) gives the level-shared operators the spatial
# awareness a fixed geometric formula would otherwise carry implicitly.
#
#   classical FMM   | this module | role
#   ---------------- | ----------- | ----
#   P2M               | P2O         | leaf box features -> outgoing repr
#   M2M                 | O2O         | child outgoing -> parent outgoing (upward pass)
#   M2L                   | O2I         | source box outgoing -> target box incoming (far field)
#   L2L                     | I2I         | parent incoming -> child incoming (downward pass)
#   L2P                       | I2P         | leaf incoming repr -> per-atom LR feature
# ---------------------------------------------------------------------------


class P2O(nn.Module):
    """Leaf box features (atom features pooled per leaf box) -> leaf box
    outgoing representation. Learned analogue of P2M: classical P2M
    analytically forms a multipole expansion from the particles physically
    inside a box; here an MLP forms a learned "outgoing" representation from
    the same pooled features instead."""

    def __init__(self, hidden_dim, operator_depth=2):
        super().__init__()
        self.mlp = _mlp(hidden_dim, hidden_dim, operator_depth)

    def forward(self, box_feat, box_positions):
        return self.mlp(apply_rope(box_feat, box_positions))


class O2O(nn.Module):
    """Child box outgoing representation -> parent box outgoing
    representation. Learned analogue of M2M (multipole-to-multipole
    translation); one instance per tree level, applied level-by-level during
    the upward pass."""

    def __init__(self, hidden_dim, operator_depth=2):
        super().__init__()
        self.mlp = _mlp(hidden_dim, hidden_dim, operator_depth)

    def forward(self, child_outgoing, child_positions):
        return self.mlp(apply_rope(child_outgoing, child_positions))


class O2I(nn.Module):
    """A source box's outgoing representation -> its contribution to a
    target box's incoming representation, for one (source, target) pair in
    the target's far-field interaction list. Learned analogue of M2L
    (multipole-to-local translation); one instance per tree level."""

    def __init__(self, hidden_dim, operator_depth=2):
        super().__init__()
        self.mlp = _mlp(hidden_dim, hidden_dim, operator_depth)

    def forward(self, source_outgoing, source_positions):
        return self.mlp(apply_rope(source_outgoing, source_positions))


class I2I(nn.Module):
    """Parent box incoming representation -> child box incoming
    representation. Learned analogue of L2L (local-to-local translation);
    one instance per tree level, applied level-by-level during the downward
    pass."""

    def __init__(self, hidden_dim, operator_depth=2):
        super().__init__()
        self.mlp = _mlp(hidden_dim, hidden_dim, operator_depth)

    def forward(self, parent_incoming, child_positions):
        return self.mlp(apply_rope(parent_incoming, child_positions))


class I2P(nn.Module):
    """Leaf box incoming representation -> per-atom long-range feature.
    Learned analogue of L2P (evaluating a local expansion at a particle's
    position)."""

    def __init__(self, hidden_dim, operator_depth=2):
        super().__init__()
        self.mlp = _mlp(hidden_dim, hidden_dim, operator_depth)

    def forward(self, leaf_incoming, leaf_positions):
        return self.mlp(apply_rope(leaf_incoming, leaf_positions))


class FMMBlock(nn.Module):
    """One literal FMM traversal over the octree:

        P2O / LEAF -> UPWARD PASS (O2O) -> ROOT
                    -> [BUILD FAR-FIELD INTERACTION LISTS: done by octree.py]
                    -> O2I / M2L -> ACCUMULATE INCOMING
                    -> DOWNWARD PASS (I2I) -> LEAVES -> I2P
                    -> ATOM-LEVEL LR FEATURES

    mirroring Eqs. 4-7 of Fognini/Betcke/Cox (arXiv:2509.20591), with every
    analytic translation operator (P2M/M2M/M2L/L2L/L2P) replaced by a
    learned operator from operators.py above, one per tree level.
    """

    def __init__(self, hidden_dim, depth, operator_depth=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.depth = depth

        self.p2o = P2O(hidden_dim, operator_depth)
        self.o2o = nn.ModuleList([O2O(hidden_dim, operator_depth) for _ in range(depth)])
        self.i2i = nn.ModuleList([I2I(hidden_dim, operator_depth) for _ in range(depth + 1)])
        self.o2i = nn.ModuleList([O2I(hidden_dim, operator_depth) for _ in range(depth + 1)])
        self.i2p = I2P(hidden_dim, operator_depth)

    def forward(self, atom_features, tree):
        leaf = tree.leaf()
        n_leaf_boxes = leaf.codes.shape[0]
        device, dtype = atom_features.device, atom_features.dtype

        # ---- P2O / LEAF: pool atom features into their leaf box, then P2O ----
        box_feat = torch.zeros(n_leaf_boxes, self.hidden_dim, device=device, dtype=dtype)
        box_feat.index_add_(0, leaf.leaf_atom_row, atom_features)

        # ---- UPWARD PASS: P2O at the leaves, O2O level-by-level up to the ROOT ----
        outgoing = [None] * (self.depth + 1)
        outgoing[self.depth] = self.p2o(box_feat, leaf.positions)

        for l in range(self.depth - 1, -1, -1):
            child = tree.levels[l + 1]
            transformed = self.o2o[l](outgoing[l + 1], child.positions)
            n_boxes_l = tree.levels[l].codes.shape[0]
            outgoing_l = torch.zeros(n_boxes_l, self.hidden_dim, device=device, dtype=dtype)
            outgoing_l.index_add_(0, child.parent_row, transformed)
            outgoing[l] = outgoing_l
        # outgoing[0] is now the ROOT's outgoing representation.

        # ---- BUILD FAR-FIELD INTERACTION LISTS: purely geometric, already
        # built by build_octree (tree.levels[l].u_target_row/u_source_row) --
        # nothing learned here, just consumed below. ----

        # ---- DOWNWARD PASS: O2I (far field) + I2I (from parent), level by
        # level, down to the LEAVES ----
        incoming = [None] * (self.depth + 1)
        incoming[0] = torch.zeros(tree.levels[0].codes.shape[0], self.hidden_dim, device=device, dtype=dtype)
        if self.depth >= 1:
            incoming[1] = torch.zeros(tree.levels[1].codes.shape[0], self.hidden_dim, device=device, dtype=dtype)

        for l in range(2, self.depth + 1):
            level = tree.levels[l]
            n_boxes_l = level.codes.shape[0]

            # I2I: incoming contribution inherited from the parent box.
            parent_incoming_bcast = incoming[l - 1][level.parent_row]
            from_parent = self.i2i[l](parent_incoming_bcast, level.positions)

            # O2I / M2L: incoming contribution from every box in this box's
            # far-field interaction list, then ACCUMULATE INCOMING -- sum
            # every interaction-list source's contribution into its target.
            far = torch.zeros(n_boxes_l, self.hidden_dim, device=device, dtype=dtype)
            if level.u_source_row is not None and level.u_source_row.numel() > 0:
                src_outgoing = outgoing[l][level.u_source_row]
                src_pos = level.positions[level.u_source_row]
                transformed = self.o2i[l](src_outgoing, src_pos)
                far.index_add_(0, level.u_target_row, transformed)  # accumulate incoming

            incoming[l] = from_parent + far

        # ---- LEAVES -> I2P -> ATOM-LEVEL LR FEATURES ----
        incoming_leaf = incoming[self.depth]
        leaf_lr_feat = self.i2p(incoming_leaf, leaf.positions)
        return leaf_lr_feat[leaf.leaf_atom_row]


class NeuralFMMTree(nn.Module):
    """Stack of T `FMMBlock`s, each one full literal FMM traversal
    (P2O -> ... -> I2P). Model depth is achieved by literally repeating the
    whole hierarchical traversal T times, following the neural-operator
    layout of Eq. 1 in arXiv:2509.20591 (lifting P, blocks with local W_t +
    nonlocal K_t + nonlinearity, projection Q, with K_t = one `FMMBlock`
    traversal) -- the paper reports this multi-block depth measurably
    improves accuracy over a single traversal, which is why it is kept here
    even though each individual block is a literal, single FMM pass.
    """

    def __init__(self, in_dim, hidden_dim=64, depth=4, n_blocks=3, operator_depth=2):
        super().__init__()
        self.depth = depth
        self.lifting = nn.Linear(in_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [FMMBlock(hidden_dim, depth, operator_depth) for _ in range(n_blocks)]
        )
        self.local_mix = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(n_blocks)])
        self.act = nn.SiLU()
        self.projection = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, atom_features, tree):
        x = self.lifting(atom_features)
        for block, local in zip(self.blocks, self.local_mix):
            x = self.act(local(x) + block(x, tree))
        return self.projection(x)
