import urllib.request
from pathlib import Path

import torch

from .fmm.octree import build_octree, move_tree_to
from .local.neighbors import periodic_neighbor_list

# Bulk liquid water, revPBE0-D3 (hybrid-functional DFT), 192 atoms/frame
# (64 H2O), periodic, 1593 configurations. Source: the training-set/ folder
# of https://github.com/BingqingCheng/ab-initio-thermodynamics-of-water
# (companion data to Cheng, Engel, Behler, Dellago & Ceriotti, PNAS 116,
# 1110 (2019), arXiv:1811.08630). No train/test split is provided upstream,
# so `download_water_dataset` carves one out itself (see
# `_split_water_dataset`) and caches both halves alongside the raw download.
WATER_DATASET_URL = (
    "https://raw.githubusercontent.com/BingqingCheng/ab-initio-thermodynamics-of-water"
    "/master/training-set/dataset_1593.xyz"
)
# Every WATER_TEST_STRIDE-th configuration (in trajectory order) is held out
# as the test set; the rest is training data. Striding rather than a
# contiguous tail split spreads the held-out frames across the whole
# trajectory, avoiding a test set drawn from a single correlated MD window.
WATER_TEST_STRIDE = 10


class AtomicSystem:
    """A single periodic atomic configuration.

    positions: (N, 3) Cartesian coordinates, float
    species: (N,) long tensor, zero-indexed species id (index into an
        embedding table -- map atomic numbers to a contiguous [0, num_species)
        range before constructing this object)
    cell: (3, 3) lattice vectors as rows, a1 = cell[0], a2 = cell[1], a3 = cell[2]
    """

    def __init__(self, positions, species, cell):
        self.positions = positions
        self.species = species
        self.cell = cell
        self._octree_cache = {}  # depth -> Octree, plus (depth, device) -> Octree
        self._neighbor_cache = {}  # cutoff -> (edge_index, shifts), plus (cutoff, device) -> (edge_index, shifts)
        self._device_cache = {}  # device -> AtomicSystem (this system's tensors already moved there)

    def to(self, *args, **kwargs):
        new = AtomicSystem(
            positions=self.positions.to(*args, **kwargs),
            species=self.species.to(*args, **kwargs),
            cell=self.cell.to(*args, **kwargs),
        )
        new._octree_cache = self._octree_cache
        new._neighbor_cache = self._neighbor_cache
        return new

    def to_cached(self, device):
        """Like `.to(device)`, but memoized: positions/species/cell never
        change across training epochs for a fixed sample, so the H2D
        transfer only needs to happen once per sample ever, not once per
        batch/epoch. Plain `.to(device)` copies from ordinary (pageable)
        CPU memory, which is a *synchronous* copy -- the CPU blocks until
        it finishes -- so repeating it every batch, for every sample in the
        batch, serializes a chunk of CPU-blocking work between every pair of
        GPU-bound batches."""
        device = torch.device(device)
        if device not in self._device_cache:
            self._device_cache[device] = self.to(device)
        return self._device_cache[device]

    def num_atoms(self):
        return self.positions.shape[0]

    def get_octree(self, depth, device=None):
        """Octree topology (occupied boxes, parent/child + near/far
        neighbor rows) depends only on `positions`/`cell`, which never
        change across training epochs for a fixed sample -- so build it
        once (on CPU, cheaply) and cache it, then cache a per-device copy
        too, instead of rebuilding it from scratch on every forward pass
        (build_octree syncs the GPU, drops to numpy, and does per-box
        Python loops -- expensive to repeat every epoch)."""
        if depth not in self._octree_cache:
            self._octree_cache[depth] = build_octree(self.positions, self.cell, depth)
        tree = self._octree_cache[depth]

        if device is None:
            return tree
        device = torch.device(device)
        device_key = (depth, device)
        if device_key not in self._octree_cache:
            self._octree_cache[device_key] = move_tree_to(tree, device)
        return self._octree_cache[device_key]

    def get_neighbor_graph(self, cutoff, device=None):
        """Neighbor-list *topology* (which atom pairs are within `cutoff`,
        and by which periodic image) depends only on `positions`/`cell`,
        fixed across epochs for a training sample -- same reasoning as
        `get_octree`. Cached here as (edge_index, shifts); the caller
        recomputes the actual (differentiable) r_ij vectors from these each
        forward pass via `local.neighbors.edge_vectors`, so forces still
        flow correctly through positions -- only the discrete "who's a
        neighbor" decision is treated as fixed.

        `periodic_neighbor_list` determines this via boolean-mask indexing,
        which forces a GPU synchronize to learn the survivor count -- caching
        it avoids paying that sync on every single forward pass/epoch.
        """
        if cutoff not in self._neighbor_cache:
            edge_index, shifts, _ = periodic_neighbor_list(self.positions, self.cell, cutoff)
            self._neighbor_cache[cutoff] = (edge_index, shifts)
        edge_index, shifts = self._neighbor_cache[cutoff]

        if device is None:
            return edge_index, shifts
        device = torch.device(device)
        device_key = (cutoff, device)
        if device_key not in self._neighbor_cache:
            self._neighbor_cache[device_key] = (edge_index.to(device), shifts.to(device))
        return self._neighbor_cache[device_key]


def ensure_downloaded(url, dest):
    """Download `url` to `dest` if it isn't already there (a non-empty file
    at `dest` is treated as already-downloaded, so this is a no-op on repeat
    calls)."""
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    print(f"downloading {url} -> {dest}")

    def _progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100.0, downloaded * 100 / total_size)
            print(f"\r  {pct:5.1f}% ({downloaded / 1e6:.1f} / {total_size / 1e6:.1f} MB)", end="", flush=True)
        else:
            print(f"\r  {downloaded / 1e6:.1f} MB", end="", flush=True)

    urllib.request.urlretrieve(url, tmp, reporthook=_progress)
    print()
    tmp.rename(dest)
    return dest


def _split_dataset(raw_path, train_path, test_path, test_stride):
    """Split a single upstream trajectory file into cached train/test extxyz
    files, taking every `test_stride`-th frame (in file order) as test.
    Skipped if both outputs already exist. Generic over what's actually in
    the file -- used for both the bulk-water and NaCl-cluster datasets."""
    import ase.io

    if train_path.exists() and test_path.exists():
        return
    frames = ase.io.read(str(raw_path), index=":")
    test_frames = frames[::test_stride]
    test_ids = set(range(0, len(frames), test_stride))
    train_frames = [atoms for i, atoms in enumerate(frames) if i not in test_ids]
    ase.io.write(str(train_path), train_frames, format="extxyz")
    ase.io.write(str(test_path), test_frames, format="extxyz")


def download_water_dataset(data_dir="data"):
    """Fetch the bundled bulk-water revPBE0-D3 benchmark into `data_dir`,
    returning (train_path, test_path). Safe to call every run -- it only
    hits the network the first time, and the train/test split is cached
    alongside the raw download so it's only computed once too."""
    data_dir = Path(data_dir)
    raw_path = ensure_downloaded(WATER_DATASET_URL, data_dir / "dataset_1593.xyz")
    train_path = data_dir / "train-H2O_revPBE0-D3.xyz"
    test_path = data_dir / "test-H2O_revPBE0-D3.xyz"
    _split_dataset(raw_path, train_path, test_path, WATER_TEST_STRIDE)
    return train_path, test_path


# Small (16-17 atom) NaCl ionic clusters, non-periodic, 5000 configurations
# split exactly 50/50 between net-neutral Na8Cl8 (16 atoms) and deliberately
# net-CHARGED Na9Cl8 (17 atoms, a +1 cluster). Source: the fit-4hdnnp-NaCl/
# folder of https://github.com/BingqingCheng/cace-lr-fit (companion code/
# data to Cheng, "Latent Ewald summation for machine learning of long-range
# interactions," arXiv:2408.15165 -- the same LES this codebase's
# `les.LESModel` is modeled on), itself following the NaCl dataset from the
# 4th-generation HDNNP paper (Ko et al., Nat. Commun. 12, 398 (2021)).
#
# Unlike bulk water -- a neutral, dipolar liquid whose orientational
# disorder screens electrostatic correlations fast, which is why a trained
# `les.LESModel` on it can end up barely using its long-range channel at all
# (see the vacuum-padding results earlier) -- these are genuinely
# non-periodic, finite ionic clusters with real, unscreened 1/r Coulomb
# tails across the whole cluster. `download_nacl_dataset` is the
# data-genuinely-needs-long-range counterpart to `download_water_dataset`.
NACL_DATASET_URL = "https://raw.githubusercontent.com/BingqingCheng/cace-lr-fit/main/fit-4hdnnp-NaCl/NaCl.xyz"
# Both `les.LESModel.compute_batched` and `fmm.NeuralFMM.compute_batched`
# require every structure in a training batch to have the SAME atom count
# (true for bulk water, where every frame is 64 molecules), so the raw
# file's mix of 16- and 17-atom frames can't be used as-is. Keeping only the
# 16-atom, net-NEUTRAL Na8Cl8 half (rather than the 17-atom, net-charged
# half) trades away the "no periodic image to screen a net charge against"
# angle, but keeps the more standard, still-genuinely-long-range case and
# needs no change to the batching machinery. Supporting mixed cluster sizes
# (e.g. bucketing batches by atom count) is a real option if the net-charged
# half turns out to matter -- not done here to keep this change minimal.
NACL_N_ATOMS = 16
# Comfortably larger than any cluster's extent (~20 A max span) in every
# direction -- see `load_extxyz`'s `vacuum_box_length` and
# `cluster.pad_into_vacuum`: the file's own per-frame "Lattice" is an
# arbitrary, too-small artifact of how each cluster was cut out (pbc="F F F"
# throughout), not a real simulation cell, so it's discarded and replaced
# with this instead.
NACL_VACUUM_BOX = 60.0
NACL_TEST_STRIDE = 10


def _filter_by_atom_count(raw_path, filtered_path, n_atoms):
    """Keep only frames with exactly `n_atoms` atoms, caching the result at
    `filtered_path` (skipped if it already exists) -- see the comment above
    `NACL_N_ATOMS` for why this filtering is needed."""
    import ase.io

    if filtered_path.exists():
        return
    frames = ase.io.read(str(raw_path), index=":")
    kept = [atoms for atoms in frames if len(atoms) == n_atoms]
    ase.io.write(str(filtered_path), kept, format="extxyz")


def download_nacl_dataset(data_dir="data"):
    """Fetch the bundled NaCl ionic-cluster benchmark into `data_dir`,
    returning (train_path, test_path). Same caching behavior as
    `download_water_dataset`, plus the one-time uniform-atom-count filter
    (see `NACL_N_ATOMS`)."""
    data_dir = Path(data_dir)
    raw_path = ensure_downloaded(NACL_DATASET_URL, data_dir / "NaCl.xyz")
    filtered_path = data_dir / f"NaCl-{NACL_N_ATOMS}atom.xyz"
    _filter_by_atom_count(raw_path, filtered_path, NACL_N_ATOMS)
    train_path = data_dir / "train-NaCl.xyz"
    test_path = data_dir / "test-NaCl.xyz"
    _split_dataset(filtered_path, train_path, test_path, NACL_TEST_STRIDE)
    return train_path, test_path
