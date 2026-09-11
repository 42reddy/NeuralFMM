import torch.nn as nn


class LRFieldEnergyHead(nn.Module):
    """The "LR ENERGY HEAD" box: reads the Neural FMM tree's per-atom
    long-range output down to a single directly-predicted energy
    contribution per atom. This is the learned counterpart of classical
    FMM's L2P + Green's-function evaluation, except the "kernel" here is the
    tree's implicit hierarchy of learned P2O/O2O/O2I/I2I operators instead
    of an explicitly known Green's function, so it isn't restricted to
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
