import torch.nn as nn


def _mlp(in_dim, out_dim, hidden_dim, depth=2):
    """`depth` hidden+output layers total (depth=2 reproduces the original
    fixed Linear-SiLU-Linear shape) -- mirrors `fmm.operators._mlp`'s depth
    knob so `LatentChargeHead` can be widened *and* deepened, not just
    widened."""
    layers = []
    d = in_dim
    for _ in range(depth - 1):
        layers += [nn.Linear(d, hidden_dim), nn.SiLU()]
        d = hidden_dim
    layers += [nn.Linear(d, out_dim)]
    return nn.Sequential(*layers)


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

    This is also the *only* learned component feeding LES's long-range
    channel -- unlike `fmm.operators.NeuralFMMTree`, there is no stack of
    "blocks" to make deeper; the Ewald kernel itself is a fixed analytic
    form with zero learned parameters (only its per-channel `alpha` is
    learned, in `LESModel`). `hidden_dim`/`depth` here are the only levers
    that grow this model's parameter count independently of the shared
    encoder.
    """

    def __init__(self, num_species, feat_dim, n_latent, hidden_dim=64, depth=2):
        super().__init__()
        self.n_latent = n_latent
        self.latent0 = nn.Embedding(num_species, n_latent)
        self.latent_mlp = _mlp(feat_dim, n_latent, hidden_dim, depth)
        nn.init.zeros_(self.latent_mlp[-1].weight)
        nn.init.zeros_(self.latent_mlp[-1].bias)

    def forward(self, species, feat):
        return self.latent0(species) + self.latent_mlp(feat)
