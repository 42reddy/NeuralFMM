import torch

from ..base import BaseAtomisticModel
from ..local.encoder import EquivariantEncoder
from ..local.heads import LocalEnergyHead
from .heads import LatentChargeHead
from .kernel import ewald_energy, ewald_energy_batched, smoothed_kernel_matrix, smoothed_kernel_matrix_batched


class LESModel(BaseAtomisticModel):
    """LES-style architecture (cf. Latent Ewald Summation, arXiv:2408.15165):
    a local equivariant net predicts a short-range atomic energy plus
    per-atom latent "charges" (`n_latent` channels, no physical meaning, no
    QEq-style equilibration or charge-neutrality constraint), and a
    long-range energy term is computed directly from those latents by one
    analytic kernel family -- a smoothed, reciprocal-space-only Ewald kernel
    (`les.kernel.smoothed_kernel_matrix`) -- except each latent channel gets
    its OWN learnable range parameter alpha_c instead of every channel
    sharing one fixed alpha: E_long = 0.5 * sum_c q_c^T K(alpha_c) q_c. This
    is still one hand-coded functional *form* (only its range is learned),
    which is exactly what makes it the baseline `fmm.NeuralFMM` is compared
    against -- see `_ewald_alpha`.

    INPUT (Z, r, H) -> EquivariantEncoder -> LocalEnergyHead + LatentChargeHead
    -> smoothed Ewald kernel -> E_SR + E_LR -> autodiff -> forces.
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
    ):
        super().__init__()
        self.n_latent = n_latent
        self.ewald_alpha_max = ewald_alpha
        self.ewald_alpha_min = ewald_alpha * ewald_alpha_min_ratio
        self.ewald_kmax = ewald_kmax

        self.local = EquivariantEncoder(num_species, hidden_dim, local_layers, n_rbf, local_r_cut)
        self.latent_head = LatentChargeHead(num_species, hidden_dim, n_latent, latent_hidden_dim, latent_depth)
        self.energy_head = LocalEnergyHead(num_species, hidden_dim)

        # One learnable range parameter per latent channel, bounded to
        # [ewald_alpha_min, ewald_alpha_max] via a sigmoid so training can
        # never push alpha_c past the point `ewald_kmax` was chosen to
        # resolve accurately. Initialized near the top of that range (close
        # to `ewald_alpha`, i.e. close to full long-range Coulomb behavior
        # for every channel) so training starts near a single-alpha
        # baseline and only pulls individual channels toward shorter
        # effective ranges where that actually helps the fit.
        self.ewald_alpha_raw = torch.nn.Parameter(torch.full((n_latent,), 2.0))

    def _ewald_alpha(self):
        span = self.ewald_alpha_max - self.ewald_alpha_min
        return self.ewald_alpha_min + span * torch.sigmoid(self.ewald_alpha_raw)

    def compute(self, positions, species, cell):
        s, _v = self.local(positions, species, cell)
        latent = self.latent_head(species, s)  # (N, n_latent)
        e_local = self.energy_head(species, s)  # (N,)

        kernel = smoothed_kernel_matrix(positions, cell, self._ewald_alpha(), self.ewald_kmax)
        e_long = ewald_energy(latent, kernel)

        e_local_total = e_local.sum()
        energy = e_local_total + e_long

        return {
            "energy": energy,
            "latent_charges": latent,
            "e_local": e_local_total,
            "e_long_range": e_long,
        }

    def compute_batched(self, positions_list, species_list, cell_list, tree=None, local_graphs=None):
        """Batched version of `compute`: one structure's worth of energy per
        entry, but the whole batch runs through the encoder/Ewald kernel as
        a single set of ops instead of once per structure. Requires a
        uniform atom count across the batch (true for this dataset; each
        `positions_list[i]` still keeps its own cell).

        `tree` is accepted (and ignored) only so `Trainer.run_epoch` -- which
        calls `energy_and_forces_batched` identically for both `LESModel`
        and `fmm.NeuralFMM` -- doesn't need to know which architecture it's
        driving; LES has no octree.
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
        latent = self.latent_head(flat_species, s)  # (B*N, n_latent)
        e_local = self.energy_head(flat_species, s)  # (B*N,)

        cell_stacked = torch.stack(cell_list, dim=0)
        latent_stacked = latent.view(b, n, self.n_latent)
        positions_stacked = flat_positions.view(b, n, 3)

        kernel = smoothed_kernel_matrix_batched(positions_stacked, cell_stacked, self._ewald_alpha(), self.ewald_kmax)
        e_long_total = ewald_energy_batched(latent_stacked, kernel)  # (B,)

        e_local_total = torch.zeros(b, device=flat_positions.device, dtype=flat_positions.dtype)
        e_local_total.index_add_(0, batch_idx, e_local)

        energy = e_local_total + e_long_total  # (B,)

        return {
            "energy": energy,
            "latent_charges": latent_stacked,
            "e_local": e_local_total,
            "e_long_range": e_long_total,
        }
