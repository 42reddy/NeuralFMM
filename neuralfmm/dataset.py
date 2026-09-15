import torch
from torch.utils.data import Dataset

from .data import (
    AtomicSystem,
    download_nacl_dataset,
    download_nacl_size_transfer_dataset,
    download_water_dataset,
)


class Sample:
    def __init__(self, system, energy, forces):
        self.system = system  # AtomicSystem
        self.energy = energy  # scalar tensor
        self.forces = forces  # (N, 3) tensor
        self._device_cache = {}  # device -> (energy, forces) already moved there

    def targets_on(self, device):
        """Like `(self.energy.to(device), self.forces.to(device))`, but
        memoized -- these labels are fixed for the life of the dataset, so
        the (synchronous, pageable-memory) H2D transfer only needs to
        happen once per sample ever. See AtomicSystem.to_cached for why
        repeating it every batch matters."""
        device = torch.device(device)
        if device not in self._device_cache:
            self._device_cache[device] = (self.energy.to(device), self.forces.to(device))
        return self._device_cache[device]


def build_species_map(symbols):
    """Deterministic symbol -> zero-indexed id map, ordered by atomic number
    so it doesn't depend on the order species happen to first appear in a
    given file (important: this map must be identical between training and
    eval, so it's always saved into the checkpoint rather than recomputed)."""
    import ase.data

    unique = sorted(set(symbols), key=lambda s: ase.data.atomic_numbers[s])
    return {s: i for i, s in enumerate(unique)}


def load_extxyz(path, species_map=None, max_samples=None):
    """Load an extxyz trajectory (ASE-readable) with per-frame `energy` and
    per-atom `forces` into a list of Samples. If `species_map` is None, one
    is built from every symbol seen in the file (pass the training set's map
    explicitly when loading a val/test file so indices line up)."""
    import ase.io

    frames = ase.io.read(str(path), index=":")
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
        # Prefer an attached calculator's results (older RPBE-D3 files), but
        # fall back to the raw extxyz info/array keys each dataset actually
        # uses instead of the calculator-backed "energy" / "forces" ASE
        # expects: "TotEnergy" / "force" for the revPBE0-D3 water dataset,
        # "REF_energy" / "REF_forces" for the NaCl(aq) dataset.
        try:
            energy = torch.tensor(atoms.get_potential_energy(), dtype=torch.float32)
            forces = torch.tensor(atoms.get_forces(), dtype=torch.float32)
        except RuntimeError:
            if "TotEnergy" in atoms.info:
                energy = torch.tensor(atoms.info["TotEnergy"], dtype=torch.float32)
                forces = torch.tensor(atoms.arrays["force"], dtype=torch.float32)
            else:
                energy = torch.tensor(atoms.info["REF_energy"], dtype=torch.float32)
                forces = torch.tensor(atoms.arrays["REF_forces"], dtype=torch.float32)

        system = AtomicSystem(positions=positions, species=species, cell=cell)
        samples.append(Sample(system=system, energy=energy, forces=forces))

    return samples, species_map


class AtomicDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def list_collate(batch):
    """No-op collate: the model consumes one structure at a time (see
    ARCHITECTURE.md's batching caveat), so a "batch" here is just the list of
    samples a training step loops over and averages the loss across."""
    return batch


def prepare_water_dataset(data_dir="data", max_train_samples=None, max_val_samples=None):
    """End-to-end: download the bundled bulk-water RPBE-D3 benchmark (if not
    already cached in `data_dir`), parse it, and return ready-to-train torch
    Datasets plus the species map used to build them."""
    train_path, test_path = download_water_dataset(data_dir)

    train_samples, species_map = load_extxyz(train_path, max_samples=max_train_samples)
    val_samples, _ = load_extxyz(test_path, species_map=species_map, max_samples=max_val_samples)

    return AtomicDataset(train_samples), AtomicDataset(val_samples), species_map


def prepare_nacl_dataset(data_dir="data", max_train_samples=None, max_val_samples=None):
    """End-to-end: download the bundled aqueous-NaCl revPBE0-D3 benchmark (if
    not already cached in `data_dir`), parse it, and return ready-to-train
    torch Datasets plus the species map used to build them -- mirrors
    `prepare_water_dataset`, for the long-range-electrostatics-heavy
    benchmark (see `data.NACL_DATASET_URL`)."""
    train_path, test_path = download_nacl_dataset(data_dir)

    train_samples, species_map = load_extxyz(train_path, max_samples=max_train_samples)
    val_samples, _ = load_extxyz(test_path, species_map=species_map, max_samples=max_val_samples)

    return AtomicDataset(train_samples), AtomicDataset(val_samples), species_map


def prepare_nacl_size_transfer_dataset(species_map, data_dir="data", max_samples=None):
    """Large-box (442/449-atom) aqueous-NaCl structures from the same
    release, held out entirely from `prepare_nacl_dataset`'s train/test
    split. Pass in the `species_map` a model was already trained with (from
    `prepare_nacl_dataset`) so indices line up -- this is meant purely as an
    evaluation set for the size-transfer comparison, never for training."""
    large_path = download_nacl_size_transfer_dataset(data_dir)
    samples, _ = load_extxyz(large_path, species_map=species_map, max_samples=max_samples)
    return AtomicDataset(samples)
