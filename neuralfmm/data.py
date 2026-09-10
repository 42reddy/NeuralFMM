import urllib.request
from pathlib import Path

import torch

from .fmm.octree import build_octree, move_tree_to
from .local.neighbors import periodic_neighbor_list

# Bulk liquid water, RPBE-D3 (dispersion-corrected DFT), 192 atoms/frame,
# periodic. See data/README.md for provenance/citation. Source: the
# data-benchmark/ folder of https://github.com/ChengUCB/les_fit (companion
# data to Cheng, npj Comput. Mater. 11, 80 (2025), arXiv:2408.15165).
WATER_TRAIN_URL = (
    "https://raw.githubusercontent.com/ChengUCB/les_fit/main/data-benchmark/train-H2O_RPBE-D3.xyz"
)
WATER_TEST_URL = (
    "https://raw.githubusercontent.com/ChengUCB/les_fit/main/data-benchmark/test-H2O_RPBE-D3.xyz"
)


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


def download_water_dataset(data_dir="data"):
    """Fetch the bundled bulk-water RPBE-D3 benchmark into `data_dir`,
    returning (train_path, test_path). Safe to call every run -- it only
    hits the network the first time."""
    data_dir = Path(data_dir)
    train_path = ensure_downloaded(WATER_TRAIN_URL, data_dir / "train-H2O_RPBE-D3.xyz")
    test_path = ensure_downloaded(WATER_TEST_URL, data_dir / "test-H2O_RPBE-D3.xyz")
    return train_path, test_path
