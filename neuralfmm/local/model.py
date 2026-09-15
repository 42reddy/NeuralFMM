import torch

from ..base import BaseAtomisticModel
from .encoder import EquivariantEncoder
from .heads import LocalEnergyHead


class LocalOnlyModel(BaseAtomisticModel):
    """The simplest possible baseline: a local equivariant encoder (PaiNN,
    same class both `les.LESModel` and `fmm.NeuralFMM` use) feeding a single
    per-atom energy head, summed to a total energy, forces from autodiff.
    No long-range term of any kind -- no Ewald kernel, no latent charges, no
    octree, nothing. Every atom's energy contribution depends only on its
    local neighborhood within `local_r_cut`.

    This exists purely as a sanity check on the shared plumbing (dataset,
    training loop, loss, evaluation) -- since every long-range variant tried
    so far has landed at or below this model's own local-only floor, the
    open question is whether that plumbing is even correct, not whether any
    particular long-range mechanism is clever enough. If this simple model
    trains to a sane energy/force MAE on the water dataset the same way
    `les.LESModel` does, the harness is trustworthy and the long-range
    architecture is the right place to keep iterating; if this also
    misbehaves, the bug is upstream of any long-range design at all.

    INPUT (Z, r, H) -> EquivariantEncoder -> s -> LocalEnergyHead(s) -> E_i
    -> E = sum_i E_i -> autodiff -> forces.
    """

    def __init__(self, num_species, hidden_dim=64, local_layers=3, n_rbf=16, local_r_cut=5.0, head_hidden_dim=64):
        super().__init__()
        self.local = EquivariantEncoder(num_species, hidden_dim, local_layers, n_rbf, local_r_cut)
        self.energy_head = LocalEnergyHead(num_species, hidden_dim, head_hidden_dim)

    def compute(self, positions, species, cell, tree=None):
        s, _v = self.local(positions, species, cell)
        e_atom = self.energy_head(species, s)  # (N,)
        energy = e_atom.sum()
        return {"energy": energy, "atomic_features": s}

    def compute_batched(self, positions_list, species_list, cell_list, tree=None, local_graphs=None):
        """Batched version of `compute` -- see
        `EquivariantEncoder.forward_batched`. `tree` is accepted (and
        ignored) only so `Trainer.run_epoch`, which calls
        `energy_and_forces_batched` identically for every architecture,
        doesn't need to know this model has no tree."""
        n_atoms_list = [p.shape[0] for p in positions_list]
        b = len(positions_list)

        flat_positions = torch.cat(positions_list, dim=0)
        flat_species = torch.cat(species_list, dim=0)
        batch_idx = torch.repeat_interleave(
            torch.arange(b, device=flat_positions.device), torch.tensor(n_atoms_list, device=flat_positions.device)
        )

        s, _v = self.local.forward_batched(positions_list, flat_species, cell_list, graphs=local_graphs)
        e_atom = self.energy_head(flat_species, s)  # (B*N,)

        energy = torch.zeros(b, device=flat_positions.device, dtype=flat_positions.dtype)
        energy.index_add_(0, batch_idx, e_atom)

        return {"energy": energy, "atomic_features": s}
