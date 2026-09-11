import torch
import torch.nn as nn

from ..utils.cutoffs import distance_rbf_envelope


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
# table. Every operator is a function of a (scalar feature, Cartesian
# relative vector) pair -- e.g. O2O(q_C, Delta r_CB) -- concatenated and fed
# through a small per-level MLP. That relative vector is what carries
# geometric information from one box/atom to another: no positional
# encoding trick (the previous RoPE-over-Morton-code scheme) stands in for
# real displacement, so e.g. O2I finally knows how far apart its source and
# target boxes actually are, not just where the source sits in an abstract
# box ordering. Alongside the raw relative vector, each operator also gets a
# `distance_rbf_envelope` (see `utils.cutoffs`) expansion of that vector's
# norm against the level's own box scale -- the same RBF+cutoff-envelope
# treatment `local.encoder.PaiNNMessage` gives interatomic distances, so
# these MLPs aren't left resolving distance-dependence from a couple of raw
# linear inputs alone (a known weak spot for forces specifically, since a
# force needs the *derivative* of this to be accurate, not just its value).
#
#   classical FMM   | this module | role
#   ---------------- | ----------- | ----
#   P2M               | P2O         | per-atom feature -> per-atom contribution to its box's outgoing repr
#   M2M                 | O2O         | child outgoing -> parent outgoing (upward pass)
#   M2L                   | O2I         | source box outgoing -> target box incoming contribution (far field)
#   L2L                     | I2I         | parent incoming -> child incoming (downward pass)
#   L2P                       | I2P         | box incoming + atom's own feature/position -> per-atom LR feature
# ---------------------------------------------------------------------------


class P2O(nn.Module):
    """h_i, r_i - center(box) -> this atom's contribution to its box's
    outgoing representation q_B^L = sum_{i in B} P2O(h_i, r_i - center_B).
    Learned analogue of P2M: classical P2M analytically sums each particle's
    contribution (its charge times a function of its position) into the
    box's multipole moment; here the per-particle contribution is an MLP of
    the particle's own feature and its position relative to the box center,
    summed the same way (the sum itself happens in FMMBlock.forward, via
    index_add_, mirroring the classical P2M sum exactly instead of pooling
    features first and transforming second).
    """

    def __init__(self, hidden_dim, operator_depth=2, n_rbf=16):
        super().__init__()
        self.n_rbf = n_rbf
        self.mlp = _mlp(hidden_dim + 3 + n_rbf, hidden_dim, operator_depth)

    def forward(self, atom_features, atom_delta, r_cut):
        feat = distance_rbf_envelope(atom_delta, r_cut, self.n_rbf)
        return self.mlp(torch.cat([atom_features, atom_delta, feat], dim=-1))


class O2O(nn.Module):
    """q_C, Delta r_CB -> this child's contribution to its parent's outgoing
    representation q_B = sum_{C in child(B)} O2O(q_C, Delta r_CB). Learned
    analogue of M2M (multipole-to-multipole translation); one instance per
    tree level, applied level-by-level during the upward pass."""

    def __init__(self, hidden_dim, operator_depth=2, n_rbf=16):
        super().__init__()
        self.n_rbf = n_rbf
        self.mlp = _mlp(hidden_dim + 3 + n_rbf, hidden_dim, operator_depth)

    def forward(self, child_outgoing, delta_to_parent, r_cut):
        feat = distance_rbf_envelope(delta_to_parent, r_cut, self.n_rbf)
        return self.mlp(torch.cat([child_outgoing, delta_to_parent, feat], dim=-1))


class O2I(nn.Module):
    """q_S, Delta r_SB -> source box S's contribution m_{S->B} to target box
    B's incoming representation, for one (source, target) pair in B's
    far-field interaction list. Learned analogue of M2L (multipole-to-local
    translation); one instance per tree level. Delta r_SB is the actual
    (periodic-minimum-image) displacement between the two box centers, so
    unlike the previous RoPE-based version, this operator can see how far
    apart -- and in which direction -- the source and target boxes are."""

    def __init__(self, hidden_dim, operator_depth=2, n_rbf=16):
        super().__init__()
        self.n_rbf = n_rbf
        self.mlp = _mlp(hidden_dim + 3 + n_rbf, hidden_dim, operator_depth)

    def forward(self, source_outgoing, delta_far, r_cut):
        feat = distance_rbf_envelope(delta_far, r_cut, self.n_rbf)
        return self.mlp(torch.cat([source_outgoing, delta_far, feat], dim=-1))


class I2I(nn.Module):
    """h_{parent(B)}, Delta r -> this box's incoming contribution inherited
    from its parent. Learned analogue of L2L (local-to-local translation);
    one instance per tree level, applied level-by-level during the downward
    pass. Uses the same Delta r (this box's center relative to its parent's)
    as O2O, just consumed in the opposite direction."""

    def __init__(self, hidden_dim, operator_depth=2, n_rbf=16):
        super().__init__()
        self.n_rbf = n_rbf
        self.mlp = _mlp(hidden_dim + 3 + n_rbf, hidden_dim, operator_depth)

    def forward(self, parent_incoming, delta_to_parent, r_cut):
        feat = distance_rbf_envelope(delta_to_parent, r_cut, self.n_rbf)
        return self.mlp(torch.cat([parent_incoming, delta_to_parent, feat], dim=-1))


class I2P(nn.Module):
    """h_{B(i)}, h_i, r_i - center(box) -> per-atom long-range feature
    z_i^LR. Learned analogue of L2P (evaluating a local expansion at a
    particle's exact position), and the fix for the previous version's main
    defect: it took only the (shared, per-box) incoming representation, so
    every atom in the same leaf box received an *identical* long-range
    feature. Concatenating the atom's own encoder feature h_i and its
    position relative to the box center restores atom-level resolution."""

    def __init__(self, hidden_dim, operator_depth=2, n_rbf=16):
        super().__init__()
        self.n_rbf = n_rbf
        self.mlp = _mlp(2 * hidden_dim + 3 + n_rbf, hidden_dim, operator_depth)

    def forward(self, box_incoming, atom_features, atom_delta, r_cut):
        feat = distance_rbf_envelope(atom_delta, r_cut, self.n_rbf)
        return self.mlp(torch.cat([box_incoming, atom_features, atom_delta, feat], dim=-1))


class FMMBlock(nn.Module):
    """One literal FMM traversal over the octree:

        P2O / LEAF -> UPWARD PASS (O2O) -> ROOT
                    -> [BUILD FAR-FIELD INTERACTION LISTS: done by octree.py]
                    -> O2I / M2L -> ACCUMULATE INCOMING
                    -> DOWNWARD PASS (I2I) -> LEAVES -> I2P
                    -> ATOM-LEVEL LR FEATURES

    with every analytic translation operator (P2M/M2M/M2L/L2L/L2P) replaced
    by a learned operator from above, each a function of a real Cartesian
    relative vector (`fmm.octree.LevelInfo.delta_to_parent`/`delta_far`, and
    the live `atom_delta` computed by `fmm.octree.atom_box_delta`) rather
    than a positional-encoding surrogate.
    """

    def __init__(self, hidden_dim, depth, operator_depth=2, n_rbf=16):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.depth = depth
        # far-field (U-list) box separations run out to ~parent's near-
        # neighbors' children, i.e. several box-widths -- wider than a
        # single box_scale -- so the O2I envelope uses a multiple of it
        # instead of clipping genuine far-field pairs to zero.
        self.far_scale_multiplier = 4.0

        self.p2o = P2O(hidden_dim, operator_depth, n_rbf)
        self.o2o = nn.ModuleList([O2O(hidden_dim, operator_depth, n_rbf) for _ in range(depth)])
        self.i2i = nn.ModuleList([I2I(hidden_dim, operator_depth, n_rbf) for _ in range(depth + 1)])
        self.o2i = nn.ModuleList([O2I(hidden_dim, operator_depth, n_rbf) for _ in range(depth + 1)])
        self.i2p = I2P(hidden_dim, operator_depth, n_rbf)

    def forward(self, atom_features, atom_delta, tree):
        leaf = tree.leaf()
        device, dtype = atom_features.device, atom_features.dtype
        atom_r_cut = leaf.box_scale[leaf.leaf_atom_row]  # (n_atoms,)

        # ---- P2O / LEAF: per-atom transform, pooled into leaf boxes ----
        # q_B^L = sum_{i in B} P2O(h_i, r_i - center_B) -- the leaf level's
        # outgoing representation *is* this sum, no further leaf-level
        # operator on top of it.
        p2o_out = self.p2o(atom_features, atom_delta, atom_r_cut)  # (n_atoms, hidden)
        n_leaf_boxes = leaf.centers.shape[0]
        outgoing_leaf = torch.zeros(n_leaf_boxes, self.hidden_dim, device=device, dtype=dtype)
        outgoing_leaf.index_add_(0, leaf.leaf_atom_row, p2o_out)

        # ---- UPWARD PASS: O2O level-by-level up to the ROOT ----
        outgoing = [None] * (self.depth + 1)
        outgoing[self.depth] = outgoing_leaf

        for l in range(self.depth - 1, -1, -1):
            child = tree.levels[l + 1]
            transformed = self.o2o[l](outgoing[l + 1], child.delta_to_parent, child.box_scale)
            n_boxes_l = tree.levels[l].centers.shape[0]
            outgoing_l = torch.zeros(n_boxes_l, self.hidden_dim, device=device, dtype=dtype)
            outgoing_l.index_add_(0, child.parent_row, transformed)
            outgoing[l] = outgoing_l
        # outgoing[0] is now the ROOT's outgoing representation.

        # ---- BUILD FAR-FIELD INTERACTION LISTS: purely geometric, already
        # built by build_octree (tree.levels[l].u_target_row/u_source_row,
        # delta_far) -- nothing learned here, just consumed below. ----

        # ---- DOWNWARD PASS: O2I (far field) + I2I (from parent), level by
        # level, down to the LEAVES ----
        incoming = [None] * (self.depth + 1)
        incoming[0] = torch.zeros(tree.levels[0].centers.shape[0], self.hidden_dim, device=device, dtype=dtype)
        if self.depth >= 1:
            incoming[1] = torch.zeros(tree.levels[1].centers.shape[0], self.hidden_dim, device=device, dtype=dtype)

        for l in range(2, self.depth + 1):
            level = tree.levels[l]
            n_boxes_l = level.centers.shape[0]

            # I2I: incoming contribution inherited from the parent box.
            parent_incoming_bcast = incoming[l - 1][level.parent_row]
            from_parent = self.i2i[l](parent_incoming_bcast, level.delta_to_parent, level.box_scale)

            # O2I / M2L: incoming contribution from every box in this box's
            # far-field interaction list, then ACCUMULATE INCOMING -- sum
            # every interaction-list source's contribution into its target.
            far = torch.zeros(n_boxes_l, self.hidden_dim, device=device, dtype=dtype)
            if level.u_source_row is not None and level.u_source_row.numel() > 0:
                src_outgoing = outgoing[l][level.u_source_row]
                far_r_cut = level.box_scale[level.u_target_row] * self.far_scale_multiplier
                transformed = self.o2i[l](src_outgoing, level.delta_far, far_r_cut)
                far.index_add_(0, level.u_target_row, transformed)  # accumulate incoming

            incoming[l] = from_parent + far

        # ---- LEAVES -> I2P -> ATOM-LEVEL LR FEATURES ----
        incoming_leaf_bcast = incoming[self.depth][leaf.leaf_atom_row]  # (n_atoms, hidden)
        return self.i2p(incoming_leaf_bcast, atom_features, atom_delta, atom_r_cut)  # already per-atom, no extra gather


class NeuralFMMTree(nn.Module):
    """Stack of T `FMMBlock`s, each one full literal FMM traversal
    (P2O -> ... -> I2P). Model depth is achieved by literally repeating the
    whole hierarchical traversal T times, following the neural-operator
    layout of Eq. 1 in arXiv:2509.20591 (lifting P, blocks with local W_t +
    nonlocal K_t + nonlinearity, projection Q, with K_t = one `FMMBlock`
    traversal) -- the paper reports this multi-block depth measurably
    improves accuracy over a single traversal, which is why it is kept here
    even though each individual block is a literal, single FMM pass.

    The atom-to-box relative vector `atom_delta` is geometry -- fixed for
    the whole stack -- while the atom *features* `x` evolve block to block
    via a residual + nonlinearity, exactly like `x` in a normal deep net.
    """

    def __init__(self, in_dim, hidden_dim=64, depth=4, n_blocks=3, operator_depth=2, n_rbf=16):
        super().__init__()
        self.depth = depth
        self.lifting = nn.Linear(in_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [FMMBlock(hidden_dim, depth, operator_depth, n_rbf) for _ in range(n_blocks)]
        )
        self.local_mix = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(n_blocks)])
        self.act = nn.SiLU()
        self.projection = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, atom_features, atom_delta, tree):
        x = self.lifting(atom_features)
        for block, local in zip(self.blocks, self.local_mix):
            x = self.act(local(x) + block(x, atom_delta, tree))
        return self.projection(x)
