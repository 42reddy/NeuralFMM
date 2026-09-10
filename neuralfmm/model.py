import torch
import torch.nn as nn

from .electrostatics import ewald_energy, ewald_energy_batched, smoothed_kernel_matrix, smoothed_kernel_matrix_batched
from .fmm.blocks import DeepNeuralFMM
from .fmm.octree import build_octree, merge_trees
from .latent.heads import FarFieldEnergyHead, LocalEnergyHead, LocalLatentChargeHead
from .local.painn import PaiNN


class NeuralFMMLES(nn.Module):
    """LES-style architecture (cf. Latent Ewald Summation, arXiv:2408.15165):
    a local equivariant net predicts a short-range atomic energy plus
    per-atom latent "charges" (`n_latent` channels, no physical meaning, no
    QEq-style equilibration or charge-neutrality constraint), and a
    long-range energy term is computed directly from those latents by one
    of two interchangeable kernels -- this replaces the earlier 4G-HDNN +
    QEq baseline (`chi`/electronegativity, a linear solve for physical
    charges) with LES's actual pipeline, so the comparison below is kernel
    vs. kernel rather than "global info or not" bolted onto a QEq model
    whose neural network only ever saw local environments to begin with.

    use_neural_fmm=False:
        the LES baseline -- one analytic kernel family (a smoothed,
        reciprocal-space-only Ewald kernel, `electrostatics.
        smoothed_kernel_matrix`), but each latent channel gets its OWN
        learnable range parameter alpha_c instead of every channel sharing
        one fixed alpha: E_long = 0.5 * sum_c q_c^T K(alpha_c) q_c. This is
        still one hand-coded functional *form* (only its range is learned),
        so the comparison against Neural FMM below stays honest -- see
        `_ewald_alpha`.
    use_neural_fmm=True:
        the same per-atom latents instead drive the Neural FMM tree
        (Sec. 3 of arXiv:2509.20591's M2M/M2L/L2L machinery, see fmm/) --
        every tree level contributes its own learned operator at its own
        length scale, so the effective kernel is an implicit hierarchy of
        learned kernels rather than one hand-coded functional form. This is
        where long-range physics a single 1/r kernel cannot represent
        (dispersion, induction, ...) is meant to be picked up from data.

    Everything else -- the PaiNN local backbone, the short-range energy
    head, and computing forces as -dE/dR by autograd -- is identical
    between the two settings.
    """

    def __init__(
        self,
        num_species,
        hidden_dim=64,
        local_layers=3,
        n_rbf=16,
        local_r_cut=5.0,
        n_latent=4,
        use_neural_fmm=True,
        tree_depth=4,
        fmm_hidden_dim=64,
        fmm_blocks=3,
        operator_depth=2,
        ewald_alpha=0.3,
        ewald_alpha_min_ratio=0.1,
        ewald_kmax=6,
    ):
        super().__init__()
        self.use_neural_fmm = use_neural_fmm
        self.tree_depth = tree_depth
        self.n_latent = n_latent
        self.ewald_alpha_max = ewald_alpha
        self.ewald_alpha_min = ewald_alpha * ewald_alpha_min_ratio
        self.ewald_kmax = ewald_kmax

        self.local = PaiNN(num_species, hidden_dim, local_layers, n_rbf, local_r_cut)
        self.latent_head = LocalLatentChargeHead(num_species, hidden_dim, n_latent)
        self.energy_head = LocalEnergyHead(num_species, hidden_dim)

        self.deep_fmm = DeepNeuralFMM(n_latent, fmm_hidden_dim, tree_depth, fmm_blocks, operator_depth)
        self.farfield_head = FarFieldEnergyHead(fmm_hidden_dim)

        # One learnable range parameter per latent channel, bounded to
        # [ewald_alpha_min, ewald_alpha_max] via a sigmoid so training can
        # never push alpha_c past the point `ewald_kmax` was chosen to
        # resolve accurately. Initialized near the top of that range (close
        # to `ewald_alpha`, i.e. close to full long-range Coulomb behavior
        # for every channel) so training starts near the old single-alpha
        # baseline and only pulls individual channels toward shorter
        # effective ranges where that actually helps the fit.
        self.ewald_alpha_raw = nn.Parameter(torch.full((n_latent,), 2.0))

    def _ewald_alpha(self):
        span = self.ewald_alpha_max - self.ewald_alpha_min
        return self.ewald_alpha_min + span * torch.sigmoid(self.ewald_alpha_raw)

    def compute(self, positions, species, cell, tree=None):
        s, _v = self.local(positions, species, cell)
        latent = self.latent_head(species, s)  # (N, n_latent)
        e_local = self.energy_head(species, s)  # (N,)

        if self.use_neural_fmm:
            if tree is None:
                tree = build_octree(positions, cell, self.tree_depth)
            farfield_latent = self.deep_fmm(latent, tree)
            e_long = self.farfield_head(farfield_latent).sum()
        else:
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

    def energy_and_forces(self, positions, species, cell, tree=None):
        positions = positions.detach().clone().requires_grad_(True)
        out = self.compute(positions, species, cell, tree=tree)
        (forces,) = torch.autograd.grad(out["energy"], positions, create_graph=self.training)
        out["forces"] = -forces
        out["positions"] = positions
        return out

    def compute_batched(self, positions_list, species_list, cell_list, tree=None, local_graphs=None):
        """Batched version of `compute`: one structure's worth of energy per
        entry, but the whole batch runs through PaiNN/Ewald/the FMM tree as
        a single set of ops instead of once per structure -- see
        `PaiNN.forward_batched`, `smoothed_kernel_matrix_batched`, and
        `fmm.octree.merge_trees` for how each stage stays batched. Requires
        a uniform atom count across the batch (true for this dataset; each
        `positions_list[i]` still keeps its own cell, so structures need not
        be otherwise identical).
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

        if self.use_neural_fmm:
            if tree is None:
                tree = merge_trees([build_octree(p, c, self.tree_depth) for p, c in zip(positions_list, cell_list)])
            farfield_latent = self.deep_fmm(latent, tree)
            e_far = self.farfield_head(farfield_latent)  # (B*N,)
            e_long_total = torch.zeros(b, device=flat_positions.device, dtype=flat_positions.dtype)
            e_long_total.index_add_(0, batch_idx, e_far)
        else:
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

    def energy_and_forces_batched(self, positions_list, species_list, cell_list, tree=None, local_graphs=None):
        """Batched version of `energy_and_forces`: one autograd.grad call
        over the whole batch's summed energy produces every structure's
        forces in a single backward pass (structures never share edges, so
        d(sum of energies)/d(structure i's positions) is exactly that
        structure's own force -- no cross-structure leakage)."""
        positions_list = [p.detach().clone().requires_grad_(True) for p in positions_list]
        out = self.compute_batched(positions_list, species_list, cell_list, tree=tree, local_graphs=local_graphs)
        grads = torch.autograd.grad(out["energy"].sum(), positions_list, create_graph=self.training)
        out["forces"] = [-g for g in grads]
        out["positions"] = positions_list
        return out
