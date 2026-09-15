import torch

from neuralfmm import NeuralFMM


def _make_system(n_atoms=8, num_species=2, seed=0, box=9.0):
    torch.manual_seed(seed)
    species = torch.randint(0, num_species, (n_atoms,))
    cell = torch.eye(3) * box
    positions = torch.rand(n_atoms, 3) * box
    return positions, species, cell


def _make_model(num_species=2, n_latent=3):
    return NeuralFMM(
        num_species=num_species,
        hidden_dim=16,
        local_layers=2,
        n_rbf=8,
        local_r_cut=4.0,
        n_latent=n_latent,
        ewald_alpha=0.3,
        ewald_alpha_min_ratio=0.1,
        ewald_kmax=4,
        response_hidden_dim=16,
        response_depth=2,
    )


def test_forward_shapes():
    positions, species, cell = _make_system()
    model = _make_model()
    out = model.compute(positions, species, cell)
    assert out["energy"].shape == ()
    assert out["latent_charges"].shape == (positions.shape[0], model.n_latent)
    assert out["latent_charges_initial"].shape == (positions.shape[0], model.n_latent)


def test_forces_match_finite_differences():
    torch.set_default_dtype(torch.float64)
    try:
        positions, species, cell = _make_system(n_atoms=6, seed=1)
        model = _make_model().double()
        with torch.no_grad():
            for p in model.response_head.parameters():
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


def test_zero_init_response_matches_les():
    """ChargeResponseHead is zero-initialized (see fmm/heads.py), so at
    init Delta q_i = 0 everywhere and NeuralFMM's final latent charges are
    exactly its own initial (LES-style) charges -- i.e. at step 0 this
    architecture *is* `les.LESModel`'s long-range term, byte for byte."""
    positions, species, cell = _make_system()
    model = _make_model()
    out = model.compute(positions, species, cell)
    assert torch.allclose(out["latent_charges"], out["latent_charges_initial"])


def test_response_reacts_to_atoms_outside_local_cutoff():
    """The structural property `les.LESModel` cannot have: its latent charge
    q_i is a pure function of atom i's *local* encoder feature, so moving an
    atom far outside local_r_cut can shift LES's Coulomb energy (real
    positions enter its Ewald sum directly) but can never change any other
    atom's own charge value. Here, moving a distant atom (well outside
    local_r_cut=4.0 both before and after, so atom 0's local feature s_0 is
    provably unchanged) still changes atom 0's *final* charge
    q_0 = q_0^0 + Delta q_0, because Delta q_0 depends on phi_0, a sum over
    the whole system.

    Positions are placed by hand (not random) so the "outside local_r_cut"
    claim is exact rather than probabilistic: atom 0 and its 3 neighbors sit
    within ~2 A of each other near one corner of a large box; the "distant"
    atom starts and ends far away (>= 9 A) from atom 0 either way.
    """
    box = 20.0
    cell = torch.eye(3) * box
    positions = torch.tensor(
        [
            [1.0, 1.0, 1.0],
            [2.0, 1.2, 1.0],
            [1.3, 2.1, 1.0],
            [1.0, 1.0, 2.2],
            [10.0, 1.0, 1.0],  # distant atom, index 4
        ]
    )
    species = torch.tensor([0, 1, 0, 1, 0])
    model = _make_model(num_species=2)
    with torch.no_grad():
        for p in model.response_head.parameters():
            p.add_(0.1 * torch.randn_like(p))

    out0 = model.compute(positions, species, cell)

    moved = positions.clone()
    moved[4] = torch.tensor([13.5, 1.0, 1.0])  # still >= 9 A from atom 0
    out_moved = model.compute(moved, species, cell)

    # atom 0's local feature (and hence its initial charge) is unaffected
    assert torch.allclose(out0["latent_charges_initial"][0], out_moved["latent_charges_initial"][0], atol=1e-6)
    # but its final, response-corrected charge is
    assert not torch.allclose(out0["latent_charges"][0], out_moved["latent_charges"][0], atol=1e-6)
