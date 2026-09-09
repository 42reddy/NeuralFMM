from dataclasses import dataclass
import torch
from torch.utils.data import Dataset
from .data import AtomicSystem


@dataclass
class Sample:
    system: AtomicSystem
    energy: torch.Tensor  # scalar
    forces: torch.Tensor  # (N, 3)


def build_species_map(symbols: list[str]) -> dict[str, int]:
    """Deterministic symbol -> zero-indexed id map, ordered by atomic number
    so it doesn't depend on the order species happen to first appear in a
    given file (important: this map must be identical between training and
    eval, so it's always saved into the checkpoint rather than recomputed)."""
    import ase.data

    unique = sorted(set(symbols), key=lambda s: ase.data.atomic_numbers[s])
    return {s: i for i, s in enumerate(unique)}


def load_extxyz(
    path: str,
    species_map: dict[str, int] | None = None,
    total_charge: float = 0.0,
    max_samples: int | None = None,
) -> tuple[list[Sample], dict[str, int]]:
    """Load an extxyz trajectory (ASE-readable) with per-frame `energy` and
    per-atom `forces` into a list of Samples. If `species_map` is None, one
    is built from every symbol seen in the file (pass the training set's map
    explicitly when loading a val/test file so indices line up)."""
    import ase.io

    frames = ase.io.read(path, index=":")
    if max_samples is not None:
        frames = frames[:max_samples]

    if species_map is None:
        all_symbols = [s for atoms in frames for s in atoms.get_chemical_symbols()]
        species_map = build_species_map(all_symbols)

    samples = []
    for atoms in frames:
        symbols = atoms.get_chemical_symbols()
        species = torch.tensor([species_map[s] for s in symbols], dtype=torch.long)
        positions = torch.tensor(atoms.get_positions(), dtype=torch.float32)
        cell = torch.tensor(atoms.cell.array, dtype=torch.float32)
        energy = torch.tensor(atoms.get_potential_energy(), dtype=torch.float32)
        forces = torch.tensor(atoms.get_forces(), dtype=torch.float32)

        system = AtomicSystem(positions=positions, species=species, cell=cell, total_charge=total_charge)
        samples.append(Sample(system=system, energy=energy, forces=forces))

    return samples, species_map


class AtomicDataset(Dataset):
    def __init__(self, samples: list[Sample]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Sample:
        return self.samples[idx]


def list_collate(batch: list[Sample]) -> list[Sample]:
    """No-op collate: the model consumes one structure at a time (see
    ARCHITECTURE.md's batching caveat), so a "batch" here is just the list of
    samples a training step loops over and averages the loss across."""
    return batch
