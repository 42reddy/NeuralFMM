import torch

from neuralfmm import NeuralFMM


def _make_system(n_atoms=8, num_species=2, seed=0, box=9.0):
    torch.manual_seed(seed)
    species = torch.randint(0, num_species, (n_atoms,))
    cell = torch.eye(3) * box
    positions = torch.rand(n_atoms, 3) * box
    return positions, species, cell


def _make_model(num_species=2):
    return NeuralFMM(
        num_species=num_species,
        hidden_dim=16,
        local_layers=2,
        n_rbf=8,
        local_r_cut=4.0,
        tree_depth=3,
        fmm_hidden_dim=16,
        fmm_blocks=2,
        operator_depth=2,
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
        with torch.no_grad():
            for p in model.farfield_head.parameters():
                p.add_(0.05 * torch.randn_like(p))

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


def test_farfield_head_zero_init_is_noop():
    """LRFieldEnergyHead is zero-initialized (see fmm/heads.py), so the
    long-range pathway starts as an exact no-op regardless of what the tree
    computes -- unlike LES's fixed analytic Ewald kernel, which contributes
    from the very first forward pass (see test_les.py)."""
    positions, species, cell = _make_system()
    model = _make_model()
    out = model.compute(positions, species, cell)
    assert out["e_long_range"].item() == 0.0
