import torch.nn as nn
import torch.nn.functional as F


def _mlp(in_dim, out_dim, hidden_dim):
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, out_dim),
    )


class LocalElectronegativityHead(nn.Module):
    """4G-HDNN-style local electronegativity/hardness: an elemental baseline
    plus an environment-dependent correction from the local equivariant
    descriptor. Purely local -- this is the whole story when
    `use_neural_fmm=False`.
    """

    def __init__(self, num_species, feat_dim, hidden_dim=64):
        super().__init__()
        self.chi0 = nn.Embedding(num_species, 1)
        self.hardness0 = nn.Embedding(num_species, 1)
        self.chi_mlp = _mlp(feat_dim, 1, hidden_dim)
        self.hardness_mlp = _mlp(feat_dim, 1, hidden_dim)
        nn.init.zeros_(self.chi_mlp[-1].weight)
        nn.init.zeros_(self.chi_mlp[-1].bias)
        nn.init.zeros_(self.hardness_mlp[-1].weight)
        nn.init.zeros_(self.hardness_mlp[-1].bias)

    def forward(self, species, feat):
        chi = self.chi0(species).squeeze(-1) + self.chi_mlp(feat).squeeze(-1)
        hardness = F.softplus(self.hardness0(species).squeeze(-1)) + F.softplus(
            self.hardness_mlp(feat).squeeze(-1)
        )
        return chi, hardness + 1e-3


class LocalEnergyHead(nn.Module):
    """Short-range (covalent-like) atomic energy contribution, analogous to a
    Behler-Parrinello / 2G-HDNN energy term -- everything that is NOT the
    long-range electrostatic/dispersion physics handled elsewhere.
    """

    def __init__(self, num_species, feat_dim, hidden_dim=64):
        super().__init__()
        self.e0 = nn.Embedding(num_species, 1)
        self.mlp = _mlp(feat_dim, 1, hidden_dim)

    def forward(self, species, feat):
        return self.e0(species).squeeze(-1) + self.mlp(feat).squeeze(-1)


class FarFieldCorrectionHead(nn.Module):
    """Splits the Deep Neural FMM's per-atom output into a global correction
    to the local electronegativity, Delta_chi, and a directly-predicted
    long-range energy channel, E_far, meant to absorb whatever long-range
    physics is NOT plain point-charge electrostatics (dispersion, induction,
    etc. -- i.e. the "multiple learned kernels" instead of one analytic 1/r
    Ewald kernel). Both are zero-initialized so the far-field path starts as
    a no-op and the model trains from the local/QEq baseline outward.
    """

    def __init__(self, farfield_dim):
        super().__init__()
        self.readout = nn.Linear(farfield_dim, 2)
        nn.init.zeros_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)

    def forward(self, farfield_feat):
        out = self.readout(farfield_feat)
        return out[..., 0], out[..., 1]
