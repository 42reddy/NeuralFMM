import torch.nn as nn


def _mlp(in_dim, out_dim, hidden_dim):
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, out_dim),
    )


class LocalEnergyHead(nn.Module):
    """The "LOCAL ENERGY HEAD" box: a short-range atomic energy contribution
    read directly off the shared equivariant encoder, exactly like a
    Behler-Parrinello / 2G-HDNN energy term. Used identically by both
    architectures -- this is also the near-field replacement in the
    classical-FMM comparison table ("Near-field: direct kernel -> local
    equivariant MLIP"), so it is not specific to either the LES kernel or
    the Neural FMM tree.
    """

    def __init__(self, num_species, feat_dim, hidden_dim=64):
        super().__init__()
        self.e0 = nn.Embedding(num_species, 1)
        self.mlp = _mlp(feat_dim, 1, hidden_dim)

    def forward(self, species, feat):
        return self.e0(species).squeeze(-1) + self.mlp(feat).squeeze(-1)
