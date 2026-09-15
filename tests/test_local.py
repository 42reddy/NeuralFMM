import torch

from neuralfmm import LocalOnlyModel


def _make_system(n_atoms=8, num_species=2, seed=0, box=9.0):
    torch.manual_seed(seed)
    species = torch.randint(0, num_species, (n_atoms,))
    cell = torch.eye(3) * box
    positions = torch.rand(n_atoms, 3) * box
    return positions, species, cell


def _make_model(num_species=2):
    return LocalOnlyModel(
        num_species=num_species,
        hidden_dim=16,
        local_layers=2,
        n_rbf=8,
        local_r_cut=4.0,
    )


def test_forward_shapes():
    positions, species, cell = _make_system()
    model = _make_model()
    out = model.compute(positions, species, cell)
    assert out["energy"].shape == ()
    assert out["atomic_features"].shape == (positions.shape[0], model.local.hidden_dim)


def test_forces_match_finite_differences():
    torch.set_default_dtype(torch.float64)
    try:
        positions, species, cell = _make_system(n_atoms=6, seed=1)
        model = _make_model().double()

        out = model.energy_and_forces(positions, species, cell)
        analytic = out["forces"].detach()

        eps = 1e-5
        for i in range(positions.shape[0]):
            for d in range(3):
                pp, pm = positions.clone(), positions.clone()
                pp[i, d] += eps
                pm[i, d] -= eps
                ep = model.compute(pp, species, cell)["energy"].item()
                em = model.compute(pm, species, cell)["energy"].item()
                fd = -(ep - em) / (2 * eps)
                assert abs(fd - analytic[i, d].item()) < 1e-5
    finally:
        torch.set_default_dtype(torch.float32)


def test_translation_and_periodic_invariance():
    positions, species, cell = _make_system(n_atoms=10, seed=2)
    model = _make_model()
    e0 = model.compute(positions, species, cell)["energy"].item()

    shift = torch.rand(3) * cell[0, 0]
    e_shift = model.compute(positions + shift, species, cell)["energy"].item()
    assert abs(e0 - e_shift) < 1e-4

    e_lattice = model.compute(positions + cell[0], species, cell)["energy"].item()
    assert abs(e0 - e_lattice) < 1e-4


def test_batched_matches_single():
    torch.set_default_dtype(torch.float64)
    try:
        p1, s1, c1 = _make_system(n_atoms=6, seed=3)
        p2, s2, c2 = _make_system(n_atoms=6, seed=4)
        model = _make_model().double()

        out_batched = model.compute_batched([p1, p2], [s1, s2], [c1, c2])
        out1 = model.compute(p1, s1, c1)
        out2 = model.compute(p2, s2, c2)

        assert torch.allclose(out_batched["energy"][0], out1["energy"], atol=1e-8)
        assert torch.allclose(out_batched["energy"][1], out2["energy"], atol=1e-8)
    finally:
        torch.set_default_dtype(torch.float32)


def test_atom_outside_cutoff_has_no_effect():
    """The defining property of a local-only model: an atom placed far
    beyond local_r_cut cannot change any other atom's energy contribution
    at all (not even indirectly), unlike `fmm.NeuralFMM`'s charge-response
    term (see test_fmm.py) or `les.LESModel`'s Ewald sum, both of which are
    sums over the whole system by construction."""
    box = 20.0
    cell = torch.eye(3) * box
    positions = torch.tensor(
        [
            [1.0, 1.0, 1.0],
            [2.0, 1.2, 1.0],
            [1.3, 2.1, 1.0],
            [10.0, 1.0, 1.0],  # distant atom, index 3
        ]
    )
    species = torch.tensor([0, 1, 0, 1])
    model = _make_model()

    out0 = model.compute(positions, species, cell)

    moved = positions.clone()
    moved[3] = torch.tensor([13.5, 1.0, 1.0])  # still far outside local_r_cut=4.0
    out_moved = model.compute(moved, species, cell)

    assert abs(out0["energy"].item() - out_moved["energy"].item()) < 1e-6
