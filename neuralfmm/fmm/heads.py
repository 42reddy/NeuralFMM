import torch
import torch.nn as nn


def _mlp(in_dim, out_dim, hidden_dim, depth=2):
    layers = []
    d = in_dim
    for _ in range(depth - 1):
        layers += [nn.Linear(d, hidden_dim), nn.SiLU()]
        d = hidden_dim
    layers += [nn.Linear(d, out_dim)]
    return nn.Sequential(*layers)


class ChargeResponseHead(nn.Module):
    """Predicts a correction Delta q_i to atom i's initial latent charge,
    from atom i's own local feature AND phi_i -- the ambient Coulomb
    potential atom i feels from every other atom's *initial* charge, near
    and far alike (see `les.kernel.ewald_potential`). This is the model's
    one piece of genuine whole-system awareness: `les.LESModel`'s charges
    are a function of local environment only, by construction, and can
    never respond to anything outside the encoder's cutoff; phi_i is a
    single scalar (per channel) that already sums the effect of the entire
    system, so Delta q_i = f(h_i, phi_i) lets an atom's charge depend on
    the field the rest of the system puts it in -- the same physical idea
    behind charge-equilibration / polarizable force fields, and the thing
    fixed-charge or one-shot latent-charge electrostatics structurally
    cannot do.

    Zero-initialized so Delta q_i = 0 at the start of training: with no
    correction, this architecture is byte-for-byte `les.LESModel`'s
    long-range term, so any improvement over LES is attributable to the
    response mechanism, not to a different starting point.
    """

    def __init__(self, feat_dim, n_latent, hidden_dim=64, depth=2):
        super().__init__()
        self.mlp = _mlp(feat_dim + n_latent, n_latent, hidden_dim, depth)
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, feat, potential):
        return self.mlp(torch.cat([feat, potential], dim=-1))
