import torch
import torch.nn as nn

from .electronegativity.heads import FarFieldCorrectionHead, LocalElectronegativityHead, LocalEnergyHead
from .electrostatics import ewald_matrix
from .fmm.blocks import DeepNeuralFMM
from .fmm.octree import build_octree
from .local.painn import PaiNN
from .qeq import electrostatic_energy, solve_qeq


class NeuralFMM4GHDNN(nn.Module):
    """4G-HDNN-style local equivariant NN + QEq, with a boolean switch to add
    Neural-FMM-supplied global information to the electronegativity
    prediction (and an extra long-range, non-electrostatic energy channel).

    use_neural_fmm=False:
        pure 4G-HDNN baseline -- chi_i is a function of the local atomic
        environment only (PaiNN descriptor within `local_r_cut`); QEq still
        provides the nonlocal charge transfer 4G-HDNN is known for, and the
        electrostatic energy is a proper periodic Ewald sum.
    use_neural_fmm=True:
        chi_i additionally receives a correction from the Neural FMM tree
        (Sec. 3 of arXiv:2509.20591's M2M/M2L/L2L machinery, ported from a
        grid onto an atomic point cloud, see fmm/), and an extra per-atom
        long-range energy channel is predicted directly by the same tree --
        this is where dispersion-like or other non-Coulombic long-range
        physics the fixed analytic Ewald kernel cannot represent is meant to
        be learned from data, instead of hand-coding a second analytic
        kernel per phenomenon.
    """

    def __init__(
        self,
        num_species,
        hidden_dim=64,
        local_layers=3,
        n_rbf=16,
        local_r_cut=5.0,
        use_neural_fmm=True,
        tree_depth=4,
        fmm_hidden_dim=64,
        fmm_blocks=3,
        operator_depth=2,
        ewald_alpha=0.3,
        ewald_r_cutoff=8.0,
        ewald_kmax=6,
    ):
        super().__init__()
        self.use_neural_fmm = use_neural_fmm
        self.tree_depth = tree_depth
        self.ewald_alpha = ewald_alpha
        self.ewald_r_cutoff = ewald_r_cutoff
        self.ewald_kmax = ewald_kmax

        self.local = PaiNN(num_species, hidden_dim, local_layers, n_rbf, local_r_cut)
        self.chi_head = LocalElectronegativityHead(num_species, hidden_dim)
        self.energy_head = LocalEnergyHead(num_species, hidden_dim)

        self.deep_fmm = DeepNeuralFMM(hidden_dim, fmm_hidden_dim, tree_depth, fmm_blocks, operator_depth)
        self.farfield_head = FarFieldCorrectionHead(fmm_hidden_dim)

    def compute(self, positions, species, cell, total_charge=0.0):
        s, _v = self.local(positions, species, cell)
        chi, hardness = self.chi_head(species, s)
        e_local = self.energy_head(species, s)

        if self.use_neural_fmm:
            tree = build_octree(positions, cell, self.tree_depth)
            farfield_latent = self.deep_fmm(s, tree)
            delta_chi, e_far = self.farfield_head(farfield_latent)
            chi = chi + delta_chi
        else:
            e_far = torch.zeros_like(e_local)

        coulomb_mat = ewald_matrix(positions, cell, self.ewald_alpha, self.ewald_r_cutoff, self.ewald_kmax)
        q, lam = solve_qeq(chi, hardness, coulomb_mat, total_charge)
        e_es = electrostatic_energy(chi, q, coulomb_mat, hardness)

        e_local_total = e_local.sum()
        e_far_total = e_far.sum()
        energy = e_local_total + e_es + e_far_total

        return {
            "energy": energy,
            "charges": q,
            "lagrange_multiplier": lam,
            "chi": chi,
            "hardness": hardness,
            "e_local": e_local_total,
            "e_electrostatic": e_es,
            "e_farfield": e_far_total,
        }

    def energy_and_forces(self, positions, species, cell, total_charge=0.0):
        positions = positions.detach().clone().requires_grad_(True)
        out = self.compute(positions, species, cell, total_charge)
        (forces,) = torch.autograd.grad(out["energy"], positions, create_graph=self.training)
        out["forces"] = -forces
        out["positions"] = positions
        return out
