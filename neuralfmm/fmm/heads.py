import torch
import torch.nn as nn


class CoupledEnergyHead(nn.Module):
    """Single per-atom MLP over the CONCATENATION of the local encoder
    feature h_i and the FMM tree's far-field feature z_i^LR -- lets local
    and far-field information interact nonlinearly (gate, rescale, modulate
    one another) instead of only being summed as two independent terms.

    A strictly additive split, e_i = LocalEnergyHead(h_i) +
    LRFieldEnergyHead(z_i^LR), can only ever express a short-range
    contribution plus a long-range contribution that never interact -- which
    is also exactly the shape of `les.LESModel` (E = f(h_i) + 0.5 q^T K q),
    so it can't demonstrate anything the tree does that a fixed physical
    kernel couldn't. There is real physics that isn't cleanly local XOR
    far-field (e.g. how a nearby cluster's field reshapes what "local"
    bonding energy even means for a polarizable atom -- induction), and a
    concatenate-then-MLP head is the minimal way to let the model represent
    that, on top of the still-present, still-separate analytic Coulomb term
    from `LatentChargeHead` (see `fmm.model.NeuralFMM` -- that physical
    branch is kept as-is precisely so this model stays directly comparable
    to `les.LESModel`: LES's term, plus whatever this coupled head can
    additionally express).

    Zero-initialized ONLY on the z_i^LR columns of the FIRST layer's weight
    (not the whole head): at init the far-field feature contributes exactly
    nothing to the first layer's pre-activation, so the model starts out at
    a normally-initialized, immediately-useful LOCAL-only prediction rather
    than identically zero for every atom (zeroing the whole head, as an
    earlier version did, throws away that local baseline and forces the
    model to re-derive ordinary local energetics from scratch at the same
    time it's learning to use far-field context -- worse early-training
    signal for no benefit). Every later layer is left at its normal init, so
    once far-field starts contributing it can interact with the local
    feature through the full MLP immediately, not just additively.
    """

    def __init__(self, num_species, local_dim, farfield_dim, hidden_dim=64):
        super().__init__()
        self.e0 = nn.Embedding(num_species, 1)
        self.input_layer = nn.Linear(local_dim + farfield_dim, hidden_dim)
        with torch.no_grad():
            self.input_layer.weight[:, local_dim:].zero_()
        self.rest = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, 1))

    def forward(self, species, local_feat, farfield_feat):
        combined = torch.cat([local_feat, farfield_feat], dim=-1)
        h = self.input_layer(combined)
        return self.e0(species).squeeze(-1) + self.rest(h).squeeze(-1)
