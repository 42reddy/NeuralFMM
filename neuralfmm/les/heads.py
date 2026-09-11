import torch.nn as nn


def _mlp(in_dim, out_dim, hidden_dim):
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, out_dim),
    )


class LatentChargeHead(nn.Module):
    """LES style per atom latent "charges": an elemental baseline plus an
    environment-dependent correction from the shared equivariant encoder's
    descriptor, predicted directly with no equilibration/neutrality
    constraint (unlike QEq's electronegativity + solved charge). `n_latent`
    independent channels are predicted per atom so the Ewald-style long-range
    kernel has more than one "type" of generalized charge to work with --
    e.g. one channel free to specialize toward Coulomb-like physics, another
    toward whatever else the kernel can fit. This head is LES-specific: the
    Neural FMM architecture has no charge concept at all (see fmm/model.py).
    """

    def __init__(self, num_species, feat_dim, n_latent, hidden_dim=64):
        super().__init__()
        self.n_latent = n_latent
        self.latent0 = nn.Embedding(num_species, n_latent)
        self.latent_mlp = _mlp(feat_dim, n_latent, hidden_dim)
        nn.init.zeros_(self.latent_mlp[-1].weight)
        nn.init.zeros_(self.latent_mlp[-1].bias)

    def forward(self, species, feat):
        return self.latent0(species) + self.latent_mlp(feat)
