import torch

from ..base import BaseAtomisticModel
from ..local.encoder import EquivariantEncoder
from ..local.heads import LocalEnergyHead
from .heads import LRFieldEnergyHead
from .octree import atom_box_delta, build_octree, merge_trees
from .operators import NeuralFMMTree


class NeuralFMM(BaseAtomisticModel):
    """Neural FMM: a local equivariant encoder feeds BOTH a short-range
    energy head and a hierarchical octree of learned FMM-style operators
    (fmm/operators.py) that produce a long-range energy contribution --
    structured to mirror classical FMM (and its GROMACS-style
    implementation) stage for stage, with every analytically-known operator
    replaced by a learned one:

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
        Green's function / Coulomb kernel                     | not prescribed -- learned      | n/a
        charge prediction / QEq                                   | none -- no charge concept      | n/a
        energy                                                       | learned (LR energy head)       | yes
        force                                                           | autodiff of energy             | n/a

    INPUT (Z, r, H)
      -> EquivariantEncoder                     (LOCAL E(3)-EQUIVARIANT ENCODER)
      -> LocalEnergyHead(s)                      (LOCAL ENERGY HEAD)
      -> s == h_i                                (ATOMIC FEATURES)
      -> build_octree(r, H)                      (BUILD OCTREE / LEAF FEATURES via P2O)
      -> NeuralFMMTree(h_i, tree)                 (UPWARD/O2O -> ROOT -> interaction lists
                                                     -> O2I/M2L -> accumulate incoming ->
                                                     DOWNWARD/I2I -> LEAVES -> I2P)
      -> LRFieldEnergyHead                          (LR ENERGY HEAD)
      -> E = E_SR + E_LR -> autodiff -> forces

    Unlike `les.LESModel`, there is no per-atom "charge" at all: the
    encoder's invariant scalar descriptor `s` is fed directly into the tree
    as the atomic features h_i (see `compute` below), matching the table's
    "Charge prediction: none required".
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
    ):
        super().__init__()
        self.needs_tree = True
        self.tree_depth = tree_depth

        self.local = EquivariantEncoder(num_species, hidden_dim, local_layers, n_rbf, local_r_cut)
        self.energy_head = LocalEnergyHead(num_species, hidden_dim)

        self.tree_module = NeuralFMMTree(hidden_dim, fmm_hidden_dim, tree_depth, fmm_blocks, operator_depth)
        self.farfield_head = LRFieldEnergyHead(fmm_hidden_dim)

    def compute(self, positions, species, cell, tree=None):
        s, _v = self.local(positions, species, cell)
        e_local = self.energy_head(species, s)  # (N,)
        h_i = s  # atomic features handed to the tree -- no charge head

        if tree is None:
            tree = build_octree(positions, cell, self.tree_depth)
        leaf = tree.leaf()
        atom_delta = atom_box_delta(positions, cell, leaf.centers[leaf.leaf_atom_row])
        farfield_feat = self.tree_module(h_i, atom_delta, tree)
        e_long = self.farfield_head(farfield_feat).sum()

        e_local_total = e_local.sum()
        energy = e_local_total + e_long

        return {
            "energy": energy,
            "atomic_features": h_i,
            "e_local": e_local_total,
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
        h_i = s

        if tree is None:
            tree = merge_trees([build_octree(p, c, self.tree_depth) for p, c in zip(positions_list, cell_list)])
        leaf = tree.leaf()
        cell_stacked = torch.stack(cell_list, dim=0)
        cell_per_atom = cell_stacked[batch_idx]  # (B*N, 3, 3) -- each atom's own structure's cell
        atom_delta = atom_box_delta(flat_positions, cell_per_atom, leaf.centers[leaf.leaf_atom_row])
        farfield_feat = self.tree_module(h_i, atom_delta, tree)
        e_far = self.farfield_head(farfield_feat)  # (B*N,)
        e_long_total = torch.zeros(b, device=flat_positions.device, dtype=flat_positions.dtype)
        e_long_total.index_add_(0, batch_idx, e_far)

        e_local_total = torch.zeros(b, device=flat_positions.device, dtype=flat_positions.dtype)
        e_local_total.index_add_(0, batch_idx, e_local)

        energy = e_local_total + e_long_total  # (B,)

        return {
            "energy": energy,
            "atomic_features": h_i,
            "e_local": e_local_total,
            "e_long_range": e_long_total,
        }
