import torch
import torch.nn as nn


class BaseAtomisticModel(nn.Module):
    """Shared -dE/dR autodiff plumbing for both architectures (`les.LESModel`
    and `fmm.NeuralFMM`). A subclass need only implement `compute` (single
    structure) and `compute_batched` (a whole training batch); forces are
    then just autograd of the returned "energy" w.r.t. the positions passed
    in, positions detached and re-flagged `requires_grad_` first so the
    graph starts exactly at the input coordinates.
    """

    def energy_and_forces(self, positions, species, cell, **kwargs):
        positions = positions.detach().clone().requires_grad_(True)
        out = self.compute(positions, species, cell, **kwargs)
        (forces,) = torch.autograd.grad(out["energy"], positions, create_graph=self.training)
        out["forces"] = -forces
        out["positions"] = positions
        return out

    def energy_and_forces_batched(self, positions_list, species_list, cell_list, **kwargs):
        """Batched version: one `autograd.grad` call over the whole batch's
        summed energy yields every structure's forces in a single backward
        pass (structures never share edges, so d(sum of energies)/d(structure
        i's positions) is exactly that structure's own force)."""
        positions_list = [p.detach().clone().requires_grad_(True) for p in positions_list]
        out = self.compute_batched(positions_list, species_list, cell_list, **kwargs)
        grads = torch.autograd.grad(out["energy"].sum(), positions_list, create_graph=self.training)
        out["forces"] = [-g for g in grads]
        out["positions"] = positions_list
        return out
