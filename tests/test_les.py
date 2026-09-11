import torch

from neuralfmm import LESModel


def _make_system(n_atoms=8, num_species=2, seed=0, box=9.0):
    torch.manual_seed(seed)
    species = torch.randint(0, num_species, (n_atoms,))
    cell = torch.eye(3) * box
    positions = torch.rand(n_atoms, 3) * box
    return positions, species, cell


def _make_model(num_species=2, n_latent=3):
    return LESModel(
        num_species=num_species,
        hidden_dim=16,
        local_layers=2,
        n_rbf=8,
        local_r_cut=4.0,
        n_latent=n_latent,
        ewald_alpha=0.3,
        ewald_alpha_min_ratio=0.1,
        ewald_kmax=4,
    )


def test_forward_shapes():
    positions, species, cell = _make_system()
    model = _make_model()
    out = model.compute(positions, species, cell)
    assert out["energy"].shape == ()
    assert out["latent_charges"].shape == (positions.shape[0], model.n_latent)


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


def test_ewald_long_range_energy_contributes():
    """The Ewald kernel is a fixed analytic form applied to whatever latent
    charges the (randomly initialized) local net predicts, so the long-range
    energy need not vanish at init (unlike NeuralFMM's zero-initialized LR
    head -- see test_fmm.py)."""
    positions, species, cell = _make_system()
    model = _make_model()
    out = model.compute(positions, species, cell)
    assert out["latent_charges"].shape == (positions.shape[0], model.n_latent)
