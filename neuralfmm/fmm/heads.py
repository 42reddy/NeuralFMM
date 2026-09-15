import torch
import torch.nn as nn


class CoupledEnergyHead(nn.Module):
    """The joint "LOCAL + LR ENERGY HEAD": a single MLP reading
    [h_i, z_i^LR] -- the local encoder feature concatenated with the tree's
    per-atom long-range output -- down to one energy contribution per atom.

    This replaces the previous design of two independent heads,
    e_i = LocalEnergyHead(h_i) + LRFieldEnergyHead(z_i^LR), summed. That
    additive shape can only ever express a short-range term plus a
    long-range term that never interact -- which is also exactly the shape
    of `les.LESModel` (E = f(h_i) + 0.5 q^T K q), so it could never
    demonstrate anything a fixed physical kernel couldn't already do. Here
    the far-field context is concatenated *into* the same MLP that predicts
    the local energy, so it can modulate the local prediction nonlinearly
    (gate it, rescale it, whatever the data needs) instead of only adding to
    it -- the one thing a pairwise Ewald-style long-range term structurally
    cannot do.

    Zero-initialized on the z_i^LR half of the input weight so the model
    starts out exactly at the species-baseline + local-only prediction, and
    only learns to lean on far-field context where doing so helps.
    """

    def __init__(self, num_species, local_dim, farfield_dim, hidden_dim=64):
        super().__init__()
        self.e0 = nn.Embedding(num_species, 1)
        self.mlp = nn.Sequential(
            nn.Linear(local_dim + farfield_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, species, local_feat, farfield_feat):
        combined = torch.cat([local_feat, farfield_feat], dim=-1)
        return self.e0(species).squeeze(-1) + self.mlp(combined).squeeze(-1)
