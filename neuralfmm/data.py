import urllib.request
import zipfile
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

# Aqueous NaCl, revPBE0-D3 (same DFT level as the bulk-water set above, via
# CP2K), periodic. Unlike bulk water, dissolved Na+/Cl- give the long-range
# Coulomb term a real, non-cancelling contribution to resolve -- see
# ARCHITECTURE.md / the FMM-vs-LES writeup for why bulk water alone (net
# Coulomb <1% of total energy there) can't distinguish the two architectures.
# Source: Hetzel & Stein, "Short-Range Machine-Learning Potentials for
# Aqueous Electrolyte Solutions", ChemPhysChem (2026), data released at
# Zenodo (CC BY 4.0): https://doi.org/10.5281/zenodo.18108882
NACL_DATASET_URL = "https://zenodo.org/api/records/18108882/files/Data%20sets.zip/content"
# The raw release bundles the actual NaCl(aq) boxes (194/196 atoms: 64 H2O +
# 1-2 ion pairs) together with a handful of 3-atom isolated-water-molecule
# normal-mode-scan frames used for a different purpose in the source paper --
# those aren't aqueous NaCl structures at all, so they're dropped here.
NACL_MIN_ATOMS = 50
# A second, independently-DFT-computed box size (442/449 atoms, same
# composition ratio, larger cell) bundled in the release's "augmented" file.
# Held out entirely from training -- see `prepare_nacl_size_transfer_dataset`
# -- to test whether a model trained at the 194/196-atom box size transfers
# to a larger box without retraining, the headline FMM-vs-LES comparison.
NACL_LARGE_BOX_ATOM_COUNTS = {442, 449}


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


def _split_water_dataset(raw_path, train_path, test_path, test_stride=WATER_TEST_STRIDE):
    """Split the single upstream trajectory file into cached train/test
    extxyz files, taking every `test_stride`-th frame as test. Skipped if
    both outputs already exist."""
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
    _split_water_dataset(raw_path, train_path, test_path)
    return train_path, test_path


def _extract_zip_member(zip_path, member, dest):
    """Pull a single member out of `zip_path` into `dest`, skipped if `dest`
    is already there. The Zenodo release bundles several unrelated datasets
    (HfO2, MAPbI3, ...) in one archive, so callers only ever ask for the one
    member (`Data sets/NaCl_*.xyz`) they actually need."""
    if dest.exists() and dest.stat().st_size > 0:
        return
    with zipfile.ZipFile(zip_path) as z, z.open(member) as src, open(dest, "wb") as out:
        out.write(src.read())


def _filter_nacl_frames(raw_path, filtered_path, select_atoms=None, min_atoms=NACL_MIN_ATOMS):
    """Drop frames that aren't aqueous-NaCl boxes at the size we want, and
    cache the result. `select_atoms`, when given, keeps only frames with
    exactly one of those atom counts (used to carve out just the large-box
    structures for the size-transfer set); otherwise keeps everything with
    at least `min_atoms` (drops the bundled isolated-water-molecule frames)."""
    import ase.io

    if filtered_path.exists():
        return
    frames = ase.io.read(str(raw_path), index=":")
    if select_atoms is not None:
        frames = [atoms for atoms in frames if len(atoms) in select_atoms]
    else:
        frames = [atoms for atoms in frames if len(atoms) >= min_atoms]
    ase.io.write(str(filtered_path), frames, format="extxyz")


def download_nacl_dataset(data_dir="data"):
    """Fetch the aqueous-NaCl revPBE0-D3 benchmark (194/196 atoms/frame) into
    `data_dir`, returning (train_path, test_path) -- mirrors
    `download_water_dataset`, except the upstream release already ships a
    train/test split (`NaCl_train.xyz`/`NaCl_test.xyz`), so there's no
    splitting to do here, only extraction from the zip and frame filtering."""
    data_dir = Path(data_dir)
    zip_path = ensure_downloaded(NACL_DATASET_URL, data_dir / "nacl_hetzel_stein.zip")

    train_path = data_dir / "train-NaCl_revPBE0-D3.xyz"
    test_path = data_dir / "test-NaCl_revPBE0-D3.xyz"
    if not train_path.exists():
        raw_train = data_dir / "_raw_NaCl_train.xyz"
        _extract_zip_member(zip_path, "Data sets/NaCl_train.xyz", raw_train)
        _filter_nacl_frames(raw_train, train_path)
    if not test_path.exists():
        raw_test = data_dir / "_raw_NaCl_test.xyz"
        _extract_zip_member(zip_path, "Data sets/NaCl_test.xyz", raw_test)
        _filter_nacl_frames(raw_test, test_path)
    return train_path, test_path


def download_nacl_size_transfer_dataset(data_dir="data"):
    """Fetch the large-box (442/449 atoms/frame) aqueous-NaCl structures from
    the same release's augmented file, returning a single path. These are
    never part of `download_nacl_dataset`'s train/test split -- they exist
    only to evaluate a model trained at the 194/196-atom box size on a
    larger, independently-DFT-computed box it has never seen."""
    data_dir = Path(data_dir)
    zip_path = ensure_downloaded(NACL_DATASET_URL, data_dir / "nacl_hetzel_stein.zip")

    large_path = data_dir / "size_transfer-NaCl_revPBE0-D3.xyz"
    if not large_path.exists():
        raw_aug = data_dir / "_raw_NaCl_train_aug.xyz"
        _extract_zip_member(zip_path, "Data sets/NaCl_train_aug.xyz", raw_aug)
        _filter_nacl_frames(raw_aug, large_path, select_atoms=NACL_LARGE_BOX_ATOM_COUNTS)
    return large_path
