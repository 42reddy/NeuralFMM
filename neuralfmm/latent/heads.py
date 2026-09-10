import torch.nn as nn


def _mlp(in_dim, out_dim, hidden_dim):
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, out_dim),
    )


class LocalLatentChargeHead(nn.Module):
    """LES style per atom latent "charges": an elemental baseline plus an
    environment-dependent correction from the local equivariant descriptor,
    predicted directly with no equilibration/neutrality constraint (unlike
    QEq's electronegativity + solved charge). `n_latent` independent
    channels are predicted per atom so the downstream long-range kernel
    (Ewald or Neural FMM) has more than one "type" of generalized charge to
    work with -- e.g. one channel free to specialize toward Coulomb-like
    physics, another toward whatever else the long-range kernel can fit.
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


class LocalEnergyHead(nn.Module):
    """Short-range (covalent-like) atomic energy contribution, analogous to a
    Behler-Parrinello / 2G-HDNN energy term -- everything that is NOT the
    long-range physics handled by the Ewald/Neural-FMM kernel.
    """

    def __init__(self, num_species, feat_dim, hidden_dim=64):
        super().__init__()
        self.e0 = nn.Embedding(num_species, 1)
        self.mlp = _mlp(feat_dim, 1, hidden_dim)

    def forward(self, species, feat):
        return self.e0(species).squeeze(-1) + self.mlp(feat).squeeze(-1)


class FarFieldEnergyHead(nn.Module):
    """Reads the Deep Neural FMM's per-atom output down to a single
    directly-predicted long-range energy contribution per atom -- the
    Neural FMM's counterpart to `electrostatics.ewald_energy`, except the
    "kernel" here is the tree's implicit hierarchy of learned M2M/M2L/L2L
    operators instead of one analytic 1/r form, so it isn't restricted to
    Coulomb-like decay (dispersion, induction, etc. are fair game). Zero-
    initialized so the far-field path starts as a no-op and the model
    trains from the local energy baseline outward.
    """

    def __init__(self, farfield_dim):
        super().__init__()
        self.readout = nn.Linear(farfield_dim, 1)
        nn.init.zeros_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)

    def forward(self, farfield_feat):
        return self.readout(farfield_feat).squeeze(-1)
