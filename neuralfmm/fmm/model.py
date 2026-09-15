import torch

from ..base import BaseAtomisticModel
from ..les.heads import LatentChargeHead
from ..les.kernel import (
    ewald_energy,
    ewald_energy_batched,
    ewald_potential,
    ewald_potential_batched,
    smoothed_kernel_matrix,
    smoothed_kernel_matrix_batched,
)
from ..local.encoder import EquivariantEncoder
from ..local.heads import LocalEnergyHead
from .heads import ChargeResponseHead


class NeuralFMM(BaseAtomisticModel):
    """Self-consistent latent-charge model: `les.LESModel` plus one
    charge-response correction, built to test a specific, narrow hypothesis
    about where a hierarchical/whole-system-aware long-range term should
    beat a fixed pairwise Ewald kernel -- rather than asking a generic
    learned operator stack to somehow discover an advantage on its own (two
    earlier versions of this file tried that: a literal FMM octree of
    learned P2O/O2O/O2I/I2I/I2P operators, and a version of it fed the
    Coulomb charges too; both landed at parity with, or slightly worse
    than, plain LES -- see git history of this file).

    The problem with LES specifically (and with this architecture's own
    earlier versions): every atom's latent charge q_i is a function of its
    *local* encoder feature alone, computed once, with no way for it to
    depend on anything happening elsewhere in the system. That is the
    textbook limitation fixed-charge electrostatics has relative to
    polarizable / charge-equilibration force fields: real charge
    distributions respond to the field they sit in (induction), and water
    -- this project's dataset -- is exactly the system where that response
    is well documented to matter (it's why fixed-charge water models like
    TIP3P need ad-hoc reparameterization, and why the DFT reference data
    this project trains against already contains a real many-body
    polarization signal that a purely local, one-shot charge assignment
    cannot represent, no matter how many latent channels it has).

    So instead of a generic learned hierarchy, this version does the
    smallest thing that could show whether that specific effect is
    learnable at all:

      1. Predict an initial latent charge q_i^0 from local features only,
         exactly as `les.LESModel` does (`LatentChargeHead`, reused as-is).
      2. Evaluate the ambient Coulomb potential phi_i = sum_j K_ij q_j^0
         every atom feels from every OTHER atom's initial charge -- near
         and far alike, via the exact same smoothed Ewald kernel LES uses
         (`les.kernel.ewald_potential`). This is the one place whole-system
         information enters the model: phi_i is a genuine sum over the
         entire system, not a locally-windowed feature.
      3. Predict a correction Delta q_i = f(h_i, phi_i) (`ChargeResponseHead`,
         zero-initialized) and use q_i = q_i^0 + Delta q_i for the final
         Coulomb energy, computed with the same kernel.

    At initialization Delta q_i = 0 everywhere, so this model is exactly
    `les.LESModel` at step 0 -- any gap that opens up during training is
    attributable to the response mechanism, not to extra unconstrained
    capacity or a different starting point. This deliberately has no
    hierarchical octree, no per-level learned translation operators, and no
    O(N log N) scaling story: those are all orthogonal to the question this
    version is built to answer (does whole-system-aware charge response
    help at all?), and can be reintroduced later, once/if this shows merit,
    purely as a cheaper way to evaluate the SAME `phi_i` at scale (that is
    what a real FMM is *for* -- fast evaluation of exactly this kind of
    sum -- not an excuse to also make the operators themselves opaque).

    INPUT (Z, r, H)
      -> EquivariantEncoder                       (LOCAL E(3)-EQUIVARIANT ENCODER)
      -> s == h_i                                  (ATOMIC FEATURES)
      -> LocalEnergyHead(s)                        (SHORT-RANGE ENERGY, same as LES)
      -> LatentChargeHead(s)                       q_i^0  (same as LES)
      -> smoothed_kernel_matrix                    K
      -> ewald_potential(q^0, K)                   phi_i = (K q^0)_i  (WHOLE-SYSTEM SIGNAL)
      -> ChargeResponseHead(h_i, phi_i)             Delta q_i  (zero-init)
      -> q_i = q_i^0 + Delta q_i
      -> ewald_energy(q, K)                        E_coulomb = 0.5 q^T K q
      -> E = E_local + E_coulomb -> autodiff -> forces
    """

    def __init__(
        self,
        num_species,
        hidden_dim=64,
        local_layers=3,
        n_rbf=16,
        local_r_cut=5.0,
        n_latent=4,
        ewald_alpha=0.3,
        ewald_alpha_min_ratio=0.1,
        ewald_kmax=6,
        latent_hidden_dim=128,
        latent_depth=4,
        response_hidden_dim=64,
        response_depth=2,
    ):
        super().__init__()
        self.n_latent = n_latent
        self.ewald_alpha_max = ewald_alpha
        self.ewald_alpha_min = ewald_alpha * ewald_alpha_min_ratio
        self.ewald_kmax = ewald_kmax

        self.local = EquivariantEncoder(num_species, hidden_dim, local_layers, n_rbf, local_r_cut)
        self.energy_head = LocalEnergyHead(num_species, hidden_dim)
        self.latent_head = LatentChargeHead(num_species, hidden_dim, n_latent, latent_hidden_dim, latent_depth)
        self.response_head = ChargeResponseHead(hidden_dim, n_latent, response_hidden_dim, response_depth)

        # One learnable range parameter per latent channel -- see
        # `les.LESModel._ewald_alpha` for the bounding/initialization
        # rationale, mirrored here verbatim.
        self.ewald_alpha_raw = torch.nn.Parameter(torch.full((n_latent,), 2.0))

    def _ewald_alpha(self):
        span = self.ewald_alpha_max - self.ewald_alpha_min
        return self.ewald_alpha_min + span * torch.sigmoid(self.ewald_alpha_raw)

    def compute(self, positions, species, cell, tree=None):
        s, _v = self.local(positions, species, cell)
        e_local = self.energy_head(species, s)  # (N,)

        q0 = self.latent_head(species, s)  # (N, n_latent)
        kernel = smoothed_kernel_matrix(positions, cell, self._ewald_alpha(), self.ewald_kmax)
        phi0 = ewald_potential(q0, kernel)  # (N, n_latent) -- whole-system signal

        latent = q0 + self.response_head(s, phi0)  # (N, n_latent)
        e_coulomb = ewald_energy(latent, kernel)

        e_local_total = e_local.sum()
        energy = e_local_total + e_coulomb

        return {
            "energy": energy,
            "atomic_features": s,
            "latent_charges": latent,
            "latent_charges_initial": q0,
            "e_local": e_local_total,
            "e_coulomb": e_coulomb,
        }

    def compute_batched(self, positions_list, species_list, cell_list, tree=None, local_graphs=None):
        """Batched version of `compute`: one structure's worth of energy per
        entry, but the whole batch runs through the encoder/kernel as a
        single set of ops instead of once per structure. Requires a uniform
        atom count across the batch (true for this dataset; each
        `positions_list[i]` still keeps its own cell). `tree` is accepted
        (and ignored) only so `Trainer.run_epoch` -- which calls
        `energy_and_forces_batched` identically for every architecture --
        doesn't need to know this model has no octree.
        """
        n_atoms_list = [p.shape[0] for p in positions_list]
        b = len(positions_list)
        n = n_atoms_list[0]
        assert all(x == n for x in n_atoms_list), "compute_batched requires a uniform atom count across the batch"

        flat_positions = torch.cat(positions_list, dim=0)
        flat_species = torch.cat(species_list, dim=0)
        batch_idx = torch.repeat_interleave(
            torch.arange(b, device=flat_positions.device), torch.tensor(n_atoms_list, device=flat_positions.device)
        )

        s, _v = self.local.forward_batched(positions_list, flat_species, cell_list, graphs=local_graphs)
        e_local = self.energy_head(flat_species, s)  # (B*N,)

        q0 = self.latent_head(flat_species, s)  # (B*N, n_latent)
        cell_stacked = torch.stack(cell_list, dim=0)
        q0_stacked = q0.view(b, n, self.n_latent)
        positions_stacked = flat_positions.view(b, n, 3)
        kernel = smoothed_kernel_matrix_batched(positions_stacked, cell_stacked, self._ewald_alpha(), self.ewald_kmax)
        phi0_stacked = ewald_potential_batched(q0_stacked, kernel)  # (B, N, n_latent)

        latent = q0 + self.response_head(s, phi0_stacked.reshape(b * n, self.n_latent))  # (B*N, n_latent)
        latent_stacked = latent.view(b, n, self.n_latent)
        e_coulomb_total = ewald_energy_batched(latent_stacked, kernel)  # (B,)

        e_local_total = torch.zeros(b, device=flat_positions.device, dtype=flat_positions.dtype)
        e_local_total.index_add_(0, batch_idx, e_local)

        energy = e_local_total + e_coulomb_total  # (B,)

        return {
            "energy": energy,
            "atomic_features": s,
            "latent_charges": latent_stacked,
            "latent_charges_initial": q0_stacked,
            "e_local": e_local_total,
            "e_coulomb": e_coulomb_total,
        }
