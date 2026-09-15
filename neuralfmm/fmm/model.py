import torch

from ..base import BaseAtomisticModel
from ..local.encoder import EquivariantEncoder
from .heads import CoupledEnergyHead
from .octree import atom_box_delta, build_octree, merge_trees
from .operators import NeuralFMMTree


class NeuralFMM(BaseAtomisticModel):
    """Neural FMM: a local equivariant encoder feeds a hierarchical octree of
    learned FMM-style operators (fmm/operators.py) -- mirroring classical FMM
    stage for stage, with every analytically-known operator replaced by a
    learned one -- and the resulting per-atom long-range feature is combined
    with the local feature through a single joint energy head:

        classical FMM           | NeuralFMM                      | learned?
        ------------------------|--------------------------------|----------
        spatial tree            | octree.build_octree            | no (geometric)
        leaf assignment         | octree.build_octree            | no (geometric)
        near/far classification | octree.build_octree            | no (geometric)
        particle representation | atomic features h_i (encoder)  | yes
        P2M                     | P2O                            | yes
        M2M                     | O2O                            | yes
        M2L                     | O2I                            | yes
        L2L                     | I2I                            | yes
        L2P                     | I2P                            | yes
        near + far combination  | CoupledEnergyHead([h_i, z_i^LR])| yes
        force                   | autodiff of energy             | n/a

    INPUT (Z, r, H)
      -> EquivariantEncoder                     (LOCAL E(3)-EQUIVARIANT ENCODER)
      -> s == h_i                                (ATOMIC FEATURES)
      -> build_octree(r, H)                      (BUILD OCTREE / LEAF FEATURES via P2O)
      -> NeuralFMMTree(h_i, tree)                  (UPWARD/O2O -> ROOT -> O2I/M2L ->
                                                     accumulate incoming -> DOWNWARD/I2I ->
                                                     LEAVES -> I2P -> per-atom z_i^LR)
      -> CoupledEnergyHead(h_i, z_i^LR)             E_i = e0(species) + MLP([h_i, z_i^LR])
      -> E = sum_i E_i -> autodiff -> forces

    Why one joint head instead of two summed ones (`LocalEnergyHead(h_i) +
    LRFieldEnergyHead(z_i^LR)`, the previous design, and also exactly the
    shape of `les.LESModel`'s `E = f(h_i) + 0.5 q^T K q`): a sum of two
    independent functions can only ever express a short-range term plus a
    long-range term that never interact. That shape cannot demonstrate any
    advantage a hierarchical, whole-system-aware tree might have over a
    fixed pairwise physical kernel, because both architectures reduce to the
    same "local + independent long-range scalar" topology -- confirmed by
    NeuralFMM (even without a Coulomb branch at all) landing at parity with
    `les.LESModel` in practice, not because the tree lacks capacity but
    because nothing in that topology lets long-range context change what
    "local" means for a given atom. Concatenating h_i and z_i^LR into one
    MLP lets the far-field feature modulate (not just add to) the local
    energy prediction -- the one thing a pairwise long-range sum structurally
    cannot do, regardless of kernel or channel count. There is deliberately
    no separate Coulomb/latent-charge branch here: this architecture is the
    minimal test of "does whole-system context, fed into the same head as
    local features, add anything" -- see NeuralFMM class docstring history
    for the abandoned charge-head and charge-coupling variants.
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
        head_hidden_dim=64,
    ):
        super().__init__()
        self.needs_tree = True
        self.tree_depth = tree_depth

        self.local = EquivariantEncoder(num_species, hidden_dim, local_layers, n_rbf, local_r_cut)
        self.tree_module = NeuralFMMTree(
            hidden_dim, fmm_hidden_dim, tree_depth, fmm_blocks, operator_depth, fmm_n_rbf
        )
        self.energy_head = CoupledEnergyHead(num_species, hidden_dim, fmm_hidden_dim, head_hidden_dim)

    def compute(self, positions, species, cell, tree=None):
        s, _v = self.local(positions, species, cell)
        h_i = s

        if tree is None:
            tree = build_octree(positions, cell, self.tree_depth)
        leaf = tree.leaf()
        atom_delta = atom_box_delta(positions, cell, leaf.centers[leaf.leaf_atom_row])
        farfield_feat = self.tree_module(h_i, atom_delta, tree)

        e_atom = self.energy_head(species, h_i, farfield_feat)  # (N,)
        energy = e_atom.sum()

        return {
            "energy": energy,
            "atomic_features": h_i,
            "farfield_features": farfield_feat,
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
        h_i = s

        if tree is None:
            tree = merge_trees([build_octree(p, c, self.tree_depth) for p, c in zip(positions_list, cell_list)])
        leaf = tree.leaf()
        cell_stacked = torch.stack(cell_list, dim=0)
        cell_per_atom = cell_stacked[batch_idx]  # (B*N, 3, 3) -- each atom's own structure's cell
        atom_delta = atom_box_delta(flat_positions, cell_per_atom, leaf.centers[leaf.leaf_atom_row])
        farfield_feat = self.tree_module(h_i, atom_delta, tree)

        e_atom = self.energy_head(flat_species, h_i, farfield_feat)  # (B*N,)
        energy = torch.zeros(b, device=flat_positions.device, dtype=flat_positions.dtype)
        energy.index_add_(0, batch_idx, e_atom)

        return {
            "energy": energy,
            "atomic_features": h_i,
            "farfield_features": farfield_feat,
        }
