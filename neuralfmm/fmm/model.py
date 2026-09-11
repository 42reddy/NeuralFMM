import torch

from ..base import BaseAtomisticModel
from ..les.heads import LatentChargeHead
from ..les.kernel import ewald_energy, ewald_energy_batched, smoothed_kernel_matrix, smoothed_kernel_matrix_batched
from ..local.encoder import EquivariantEncoder
from ..local.heads import LocalEnergyHead
from .heads import LRFieldEnergyHead
from .octree import atom_box_delta, build_octree, merge_trees
from .operators import NeuralFMMTree


class NeuralFMM(BaseAtomisticModel):
    """Neural FMM: a local equivariant encoder feeds a short-range energy
    head, a `les`-style analytic Coulomb branch, AND a hierarchical octree of
    learned FMM-style operators (fmm/operators.py) -- structured to mirror
    classical FMM (and its GROMACS-style implementation) stage for stage,
    with every analytically-known operator replaced by a learned one, EXCEPT
    the Coulomb kernel itself, which is now given explicitly rather than
    left for the learned operators to rediscover from data:

        classical FMM        | here                          | learned?
        ----------------------|-------------------------------|----------
        spatial tree            | octree.build_octree            | no (geometric)
        leaf assignment            | octree.build_octree            | no (geometric)
        near/far classification       | octree.build_octree            | no (geometric)
        particle representation          | atomic features h_i (encoder)  | yes
        P2M                                 | P2O                            | yes
        M2M                                    | O2O                            | yes
        M2L                                       | O2I                            | yes
        L2L                                          | I2I                            | yes
        L2P                                             | I2P                            | yes
        near-field                                         | LocalEnergyHead (local MLIP)   | yes
        Green's function / Coulomb kernel                     | LatentChargeHead + smoothed    | charges: yes
                                                                | Ewald kernel (les.kernel)      | kernel form: no
        charge prediction                                        | LatentChargeHead (les-style)    | yes
        energy                                                       | E_local + E_coulomb + E_LR    | yes
        force                                                           | autodiff of energy             | n/a

    INPUT (Z, r, H)
      -> EquivariantEncoder                     (LOCAL E(3)-EQUIVARIANT ENCODER)
      -> LocalEnergyHead(s)                      (LOCAL ENERGY HEAD)
      -> s == h_i                                (ATOMIC FEATURES)
      -> LatentChargeHead(s)                     (PHYSICAL BRANCH: per-atom latent
                                                     charges q_i, same n_latent
                                                     channels and per-channel
                                                     learnable range alpha_c as
                                                     `les.LESModel`)
      -> smoothed_kernel_matrix / ewald_energy     E_coulomb = 0.5 sum_c q_c^T K(alpha_c) q_c
      -> build_octree(r, H)                      (BUILD OCTREE / LEAF FEATURES via P2O)
      -> NeuralFMMTree([h_i, q_i], tree)           (IMPLICIT BRANCH -- same UPWARD/O2O ->
                                                     ROOT -> O2I/M2L -> accumulate incoming ->
                                                     DOWNWARD/I2I -> LEAVES -> I2P as before,
                                                     now charge-aware, free to learn whatever
                                                     Coulomb-only Ewald can't -- dispersion,
                                                     induction, correlation, many-body terms)
      -> LRFieldEnergyHead                          (LR ENERGY HEAD)
      -> E = E_SR + E_coulomb + E_LR -> autodiff -> forces

    The physical branch is an exact copy of `les.LESModel`'s long-range
    machinery (same head, same kernel, same per-channel alpha
    parameterization) so the two architectures are directly comparable: this
    model is LES's physics-informed Coulomb term plus an additional,
    hierarchical, purely-learned correction on top -- rather than asking the
    learned operators to rediscover 1/r decay from scratch, as the previous
    version did.
    """

    def __init__(
        self,
        num_species,
        hidden_dim=64,
        local_layers=3,
        n_rbf=16,
        local_r_cut=5.0,
        tree_depth=4,
        fmm_hidden_dim=64,
        fmm_blocks=3,
        operator_depth=2,
        fmm_n_rbf=16,
        n_latent=4,
        ewald_alpha=0.3,
        ewald_alpha_min_ratio=0.1,
        ewald_kmax=6,
        latent_hidden_dim=128,
        latent_depth=4,
    ):
        super().__init__()
        self.needs_tree = True
        self.tree_depth = tree_depth
        self.n_latent = n_latent
        self.ewald_alpha_max = ewald_alpha
        self.ewald_alpha_min = ewald_alpha * ewald_alpha_min_ratio
        self.ewald_kmax = ewald_kmax

        self.local = EquivariantEncoder(num_species, hidden_dim, local_layers, n_rbf, local_r_cut)
        self.energy_head = LocalEnergyHead(num_species, hidden_dim)

        self.latent_head = LatentChargeHead(num_species, hidden_dim, n_latent, latent_hidden_dim, latent_depth)
        # One learnable range parameter per latent channel -- see
        # `les.LESModel._ewald_alpha` for the bounding/initialization
        # rationale, mirrored here verbatim.
        self.ewald_alpha_raw = torch.nn.Parameter(torch.full((n_latent,), 2.0))

        self.tree_module = NeuralFMMTree(
            hidden_dim + n_latent, fmm_hidden_dim, tree_depth, fmm_blocks, operator_depth, fmm_n_rbf
        )
        self.farfield_head = LRFieldEnergyHead(fmm_hidden_dim)

    def _ewald_alpha(self):
        span = self.ewald_alpha_max - self.ewald_alpha_min
        return self.ewald_alpha_min + span * torch.sigmoid(self.ewald_alpha_raw)

    def compute(self, positions, species, cell, tree=None):
        s, _v = self.local(positions, species, cell)
        e_local = self.energy_head(species, s)  # (N,)

        latent = self.latent_head(species, s)  # (N, n_latent)
        kernel = smoothed_kernel_matrix(positions, cell, self._ewald_alpha(), self.ewald_kmax)
        e_coulomb = ewald_energy(latent, kernel)

        h_i = torch.cat([s, latent], dim=-1)  # atomic features handed to the tree, charge-aware

        if tree is None:
            tree = build_octree(positions, cell, self.tree_depth)
        leaf = tree.leaf()
        atom_delta = atom_box_delta(positions, cell, leaf.centers[leaf.leaf_atom_row])
        farfield_feat = self.tree_module(h_i, atom_delta, tree)
        e_long = self.farfield_head(farfield_feat).sum()

        e_local_total = e_local.sum()
        energy = e_local_total + e_coulomb + e_long

        return {
            "energy": energy,
            "atomic_features": h_i,
            "latent_charges": latent,
            "e_local": e_local_total,
            "e_coulomb": e_coulomb,
            "e_long_range": e_long,
        }

    def compute_batched(self, positions_list, species_list, cell_list, tree=None, local_graphs=None):
        """Batched version of `compute`: one structure's worth of energy per
        entry, but the whole batch runs through the encoder/tree as a single
        set of ops instead of once per structure -- see
        `EquivariantEncoder.forward_batched` and `fmm.octree.merge_trees`.
        Requires a uniform atom count across the batch (true for this
        dataset; each `positions_list[i]` still keeps its own cell).
        """
        n_atoms_list = [p.shape[0] for p in positions_list]
        b = len(positions_list)
        assert all(x == n_atoms_list[0] for x in n_atoms_list), (
            "compute_batched requires a uniform atom count across the batch"
        )

        flat_positions = torch.cat(positions_list, dim=0)
        flat_species = torch.cat(species_list, dim=0)
        batch_idx = torch.repeat_interleave(
            torch.arange(b, device=flat_positions.device), torch.tensor(n_atoms_list, device=flat_positions.device)
        )

        s, _v = self.local.forward_batched(positions_list, flat_species, cell_list, graphs=local_graphs)
        e_local = self.energy_head(flat_species, s)  # (B*N,)

        latent = self.latent_head(flat_species, s)  # (B*N, n_latent)
        cell_stacked = torch.stack(cell_list, dim=0)
        latent_stacked = latent.view(b, n_atoms_list[0], self.n_latent)
        positions_stacked = flat_positions.view(b, n_atoms_list[0], 3)
        kernel = smoothed_kernel_matrix_batched(positions_stacked, cell_stacked, self._ewald_alpha(), self.ewald_kmax)
        e_coulomb_total = ewald_energy_batched(latent_stacked, kernel)  # (B,)

        h_i = torch.cat([s, latent], dim=-1)

        if tree is None:
            tree = merge_trees([build_octree(p, c, self.tree_depth) for p, c in zip(positions_list, cell_list)])
        leaf = tree.leaf()
        cell_per_atom = cell_stacked[batch_idx]  # (B*N, 3, 3) -- each atom's own structure's cell
        atom_delta = atom_box_delta(flat_positions, cell_per_atom, leaf.centers[leaf.leaf_atom_row])
        farfield_feat = self.tree_module(h_i, atom_delta, tree)
        e_far = self.farfield_head(farfield_feat)  # (B*N,)
        e_long_total = torch.zeros(b, device=flat_positions.device, dtype=flat_positions.dtype)
        e_long_total.index_add_(0, batch_idx, e_far)

        e_local_total = torch.zeros(b, device=flat_positions.device, dtype=flat_positions.dtype)
        e_local_total.index_add_(0, batch_idx, e_local)

        energy = e_local_total + e_coulomb_total + e_long_total  # (B,)

        return {
            "energy": energy,
            "atomic_features": h_i,
            "latent_charges": latent_stacked,
            "e_local": e_local_total,
            "e_coulomb": e_coulomb_total,
            "e_long_range": e_long_total,
        }
