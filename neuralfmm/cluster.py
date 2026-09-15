"""Carve a compact, non-periodic water cluster out of a periodic bulk
configuration, and re-embed it in cells of growing size -- the data-free way
to push a model trained only on one periodic bulk-water box toward the
free-space/non-periodic limit it never saw during training. See
`Evaluator.vacuum_padding_test` (evaluation.py) for how this is used to
compare `les.LESModel`'s Ewald kernel against `fmm.NeuralFMM`'s octree: no
new simulation, no new labels, just the existing dataset and a trained
checkpoint.
"""
import torch


def _minimum_image(delta, cell):
    """Cartesian displacement(s) `delta` wrapped to their periodic minimum
    image under `cell`. delta: (..., 3)."""
    inv_cell = torch.linalg.inv(cell)
    frac = delta @ inv_cell
    frac = frac - torch.round(frac)
    return frac @ cell


def assign_water_molecules(positions, species, cell, o_id, h_id):
    """Pair every H atom with its covalently bonded O via nearest
    periodic-minimum-image O -- robust to file ordering (the raw dataset
    does NOT list atoms molecule-by-molecule, see data.py), and needs no
    bond-length cutoff since every H belongs to exactly one O in a
    well-formed water configuration.

    Returns a list of (o_index, h_index_1, h_index_2) atom-index triples,
    one per molecule, in no particular order.
    """
    o_indices = (species == o_id).nonzero(as_tuple=True)[0]
    h_indices = (species == h_id).nonzero(as_tuple=True)[0]

    delta = positions[h_indices].unsqueeze(1) - positions[o_indices].unsqueeze(0)  # (n_h, n_o, 3)
    dist2 = (_minimum_image(delta, cell) ** 2).sum(-1)
    nearest_o = dist2.argmin(dim=1)  # (n_h,) row into o_indices

    molecules = {}
    for h_row, o_row in enumerate(nearest_o.tolist()):
        molecules.setdefault(o_row, []).append(h_indices[h_row].item())

    triples = []
    for o_row, h_list in molecules.items():
        assert len(h_list) == 2, f"O atom bonded to {len(h_list)} H atoms, expected 2"
        triples.append((o_indices[o_row].item(), h_list[0], h_list[1]))
    return triples


def extract_water_cluster(positions, species, cell, o_id, h_id, n_molecules, seed_index=None):
    """Carve `n_molecules` whole water molecules out of a periodic bulk
    configuration into one contiguous, UNWRAPPED (non-periodic) cluster.

    The seed molecule's O sits at the unwrap origin; every other selected
    molecule's O is placed via its periodic-minimum-image displacement from
    the seed O (exact as long as the cluster's physical extent stays well
    under half the original box length, true for any modest `n_molecules`
    seeded near the box center); each molecule's H atoms are placed via
    minimum-image displacement from their OWN (already-unwrapped) O, so
    every bond length is preserved exactly -- this just re-expresses the
    same geometry in a single non-periodic frame instead of wrapped into the
    original cell.

    Returns (cluster_positions, cluster_species): (3*n_molecules, 3) and
    (3*n_molecules,) long, atoms ordered molecule by molecule as (O, H, H).
    """
    triples = assign_water_molecules(positions, species, cell, o_id, h_id)
    o_positions = positions[[t[0] for t in triples]]

    if seed_index is None:
        center = cell.sum(dim=0) / 2
        seed_index = int((o_positions - center).norm(dim=-1).argmin())

    seed_o_pos = o_positions[seed_index]
    o_delta = _minimum_image(o_positions - seed_o_pos, cell)
    order = o_delta.norm(dim=-1).argsort()[:n_molecules].tolist()

    cluster_positions, cluster_species = [], []
    for row in order:
        o_idx, h1_idx, h2_idx = triples[row]
        o_unwrapped = seed_o_pos + o_delta[row]
        cluster_positions.append(o_unwrapped)
        cluster_species.append(species[o_idx])
        for h_idx in (h1_idx, h2_idx):
            h_delta = _minimum_image(positions[h_idx] - positions[o_idx], cell)
            cluster_positions.append(o_unwrapped + h_delta)
            cluster_species.append(species[h_idx])

    return torch.stack(cluster_positions), torch.stack(cluster_species)


def pad_into_vacuum(positions, box_length):
    """Re-center `positions` (already a compact, non-periodic cluster --
    see `extract_water_cluster`) inside a new cubic cell of side
    `box_length`, leaving the internal geometry bit-identical. Calling this
    with a growing `box_length` is what pushes a periodic-only architecture
    (LES's Ewald kernel, or the FMM octree's periodic-torus wrap) toward its
    free-space limit, using only re-labeled positions -- no new data.

    Returns (positions, cell).
    """
    centroid = positions.mean(dim=0)
    shift = (box_length / 2) - centroid
    cell = torch.eye(3, device=positions.device, dtype=positions.dtype) * box_length
    return positions + shift, cell
