from dataclasses import dataclass
import torch


class AtomicSystem:
    """A single periodic atomic configuration.

    positions: (N, 3) Cartesian coordinates, float
    species: (N,) long tensor, zero-indexed species id (index into an
        embedding table -- map atomic numbers to a contiguous [0, num_species)
        range before constructing this object)
    cell: (3, 3) lattice vectors as rows, a1 = cell[0], a2 = cell[1], a3 = cell[2]
    total_charge: net charge of the cell enforced as the QEq constraint
    """

    positions: torch.Tensor
    species: torch.Tensor
    cell: torch.Tensor
    total_charge: float = 0.0

    def to(self, *args, **kwargs) -> "AtomicSystem":
        return AtomicSystem(
            positions=self.positions.to(*args, **kwargs),
            species=self.species.to(*args, **kwargs),
            cell=self.cell.to(*args, **kwargs),
            total_charge=self.total_charge,
        )

    def num_atoms(self) -> int:
        return self.positions.shape[0]
