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


class NeuralFMMBlock(nn.Module):
    """One Upward -> Downward -> Leaf pass over the octree (Eqs. 4-7 of
    Fognini/Betcke/Cox, 2509.20591), with every analytic translation operator
    (T_ofs, T_ofo, T_ifi, T_ifo, T_tfi) replaced by a learned MLP, one per
    tree level as in the paper. RoPE over each box's Morton code restores the
    spatial awareness the level-shared MLPs would otherwise lack.
    """

    def __init__(self, hidden_dim, depth, operator_depth=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.depth = depth

        self.t_ofs = _mlp(hidden_dim, hidden_dim, operator_depth)
        self.t_ofo = nn.ModuleList([_mlp(hidden_dim, hidden_dim, operator_depth) for _ in range(depth)])
        self.t_ifi = nn.ModuleList([_mlp(hidden_dim, hidden_dim, operator_depth) for _ in range(depth + 1)])
        self.t_ifo = nn.ModuleList([_mlp(hidden_dim, hidden_dim, operator_depth) for _ in range(depth + 1)])
        self.t_tfi = _mlp(hidden_dim, hidden_dim, operator_depth)

    def forward(self, atom_features, tree):
        leaf = tree.leaf()
        n_leaf_boxes = leaf.codes.shape[0]
        box_feat = torch.zeros(n_leaf_boxes, self.hidden_dim, device=atom_features.device, dtype=atom_features.dtype)
        box_feat.index_add_(0, leaf.leaf_atom_row, atom_features)

        q = [None] * (self.depth + 1)
        q[self.depth] = self.t_ofs(apply_rope(box_feat, leaf.positions))

        for l in range(self.depth - 1, -1, -1):
            child = tree.levels[l + 1]
            transformed = self.t_ofo[l](apply_rope(q[l + 1], child.positions))
            n_boxes_l = tree.levels[l].codes.shape[0]
            q_l = torch.zeros(n_boxes_l, self.hidden_dim, device=atom_features.device, dtype=atom_features.dtype)
            q_l.index_add_(0, child.parent_row, transformed)
            q[l] = q_l

        h = [None] * (self.depth + 1)
        h[0] = torch.zeros(tree.levels[0].codes.shape[0], self.hidden_dim, device=atom_features.device, dtype=atom_features.dtype)
        if self.depth >= 1:
            h[1] = torch.zeros(tree.levels[1].codes.shape[0], self.hidden_dim, device=atom_features.device, dtype=atom_features.dtype)

        for l in range(2, self.depth + 1):
            level = tree.levels[l]
            n_boxes_l = level.codes.shape[0]

            h_parent_bcast = h[l - 1][level.parent_row]
            from_parent = self.t_ifi[l](apply_rope(h_parent_bcast, level.positions))

            far = torch.zeros(n_boxes_l, self.hidden_dim, device=atom_features.device, dtype=atom_features.dtype)
            if level.u_source_row is not None and level.u_source_row.numel() > 0:
                src_q = q[l][level.u_source_row]
                src_pos = level.positions[level.u_source_row]
                transformed = self.t_ifo[l](apply_rope(src_q, src_pos))
                far.index_add_(0, level.u_target_row, transformed)

            h[l] = from_parent + far

        h_leaf = h[self.depth]
        v_leaf = self.t_tfi(apply_rope(h_leaf, leaf.positions))
        return v_leaf[leaf.leaf_atom_row]


class DeepNeuralFMM(nn.Module):
    """Stack of NeuralFMMBlocks, mirroring the Neural Operator layout
    (lifting P, blocks with local W_t + nonlocal K_t + nonlinearity, projection
    Q) from Eq. 1 of the paper, with K_t = NeuralFMMBlock.
    """

    def __init__(self, in_dim, hidden_dim=64, depth=4, n_blocks=3, operator_depth=2):
        super().__init__()
        self.depth = depth
        self.lifting = nn.Linear(in_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [NeuralFMMBlock(hidden_dim, depth, operator_depth) for _ in range(n_blocks)]
        )
        self.local_mix = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(n_blocks)])
        self.act = nn.SiLU()
        self.projection = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, atom_features, tree):
        x = self.lifting(atom_features)
        for block, local in zip(self.blocks, self.local_mix):
            x = self.act(local(x) + block(x, tree))
        return self.projection(x)
