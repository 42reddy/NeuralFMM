import torch

from neuralfmm.cluster import extract_water_cluster, pad_into_vacuum


def _make_wrapped_water_box(n_molecules=6, box=12.0, seed=0):
    """Molecules placed with a real O-H geometry, then wrapped into [0, box)
    and shuffled atom-by-atom -- so nothing about their storage order or
    coordinates hints at molecule membership, exercising the same
    reconstruction the real dataset needs (see data.py's raw file, where
    O/H/H row order does NOT correspond to bonding either)."""
    torch.manual_seed(seed)
    cell = torch.eye(3) * box

    # rejection-sample O positions with a realistic minimum O-O separation
    # (real liquid water never packs O atoms closer than ~2.5 A) -- random
    # placement without this can put two different molecules' O atoms close
    # enough that an H ends up nearer to the "wrong" O, which never happens
    # in an actual MD configuration.
    min_o_o = 3.0
    o_pos = torch.empty(0, 3)
    while o_pos.shape[0] < n_molecules:
        candidate = torch.rand(3) * box
        if o_pos.shape[0] == 0 or (o_pos - candidate).norm(dim=-1).min() > min_o_o:
            o_pos = torch.cat([o_pos, candidate.unsqueeze(0)], dim=0)

    # simple, fixed intramolecular O-H geometry (bond ~0.96 A, HOH ~104.5 deg)
    bond = 0.96
    half_angle = torch.deg2rad(torch.tensor(104.5 / 2))
    h_offsets = torch.stack(
        [
            torch.tensor([bond * torch.sin(half_angle), bond * torch.cos(half_angle), 0.0]),
            torch.tensor([-bond * torch.sin(half_angle), bond * torch.cos(half_angle), 0.0]),
        ]
    )

    positions, species = [], []
    for i in range(n_molecules):
        positions.append(o_pos[i])
        species.append(0)  # O = species id 0
        for off in h_offsets:
            h = o_pos[i] + off
            positions.append(h)
            species.append(1)  # H = species id 1
    positions = torch.stack(positions)
    species = torch.tensor(species, dtype=torch.long)

    # wrap into the cell, exactly like a real MD trajectory dump
    inv_cell = torch.linalg.inv(cell)
    frac = positions @ inv_cell
    frac = frac - torch.floor(frac)
    positions = frac @ cell

    # shuffle atom order so file order carries no molecule information
    perm = torch.randperm(positions.shape[0])
    return positions[perm], species[perm], cell


def test_extract_water_cluster_preserves_bond_lengths():
    positions, species, cell = _make_wrapped_water_box(n_molecules=8, box=12.0)
    cluster_pos, cluster_species = extract_water_cluster(positions, species, cell, o_id=0, h_id=1, n_molecules=8)

    assert cluster_species.shape[0] == 24
    # atoms ordered molecule-by-molecule as (O, H, H)
    for m in range(8):
        o = cluster_pos[3 * m]
        h1 = cluster_pos[3 * m + 1]
        h2 = cluster_pos[3 * m + 2]
        assert cluster_species[3 * m].item() == 0
        assert cluster_species[3 * m + 1].item() == 1
        assert cluster_species[3 * m + 2].item() == 1
        assert abs((o - h1).norm().item() - 0.96) < 1e-4
        assert abs((o - h2).norm().item() - 0.96) < 1e-4


def test_extract_water_cluster_is_compact_and_unwrapped():
    positions, species, cell = _make_wrapped_water_box(n_molecules=10, box=12.0)
    cluster_pos, _ = extract_water_cluster(positions, species, cell, o_id=0, h_id=1, n_molecules=5)

    # a genuinely unwrapped, contiguous cluster of 5 nearby molecules should
    # span well under the full box length in any direction
    span = (cluster_pos.max(dim=0).values - cluster_pos.min(dim=0).values)
    assert span.max().item() < 12.0


def test_pad_into_vacuum_keeps_internal_geometry():
    positions, species, cell = _make_wrapped_water_box(n_molecules=4, box=10.0)
    cluster_pos, _ = extract_water_cluster(positions, species, cell, o_id=0, h_id=1, n_molecules=4)

    padded_small, cell_small = pad_into_vacuum(cluster_pos, box_length=20.0)
    padded_large, cell_large = pad_into_vacuum(cluster_pos, box_length=50.0)

    assert cell_small[0, 0].item() == 20.0
    assert cell_large[0, 0].item() == 50.0

    # only a rigid translation between the two -- every pairwise distance
    # (and therefore every bond length/angle) must be identical
    d_small = torch.cdist(padded_small, padded_small)
    d_large = torch.cdist(padded_large, padded_large)
    assert torch.allclose(d_small, d_large, atol=1e-5)
