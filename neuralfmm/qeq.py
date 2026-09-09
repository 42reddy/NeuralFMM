import torch


def solve_qeq(chi, hardness, coulomb_matrix, total_charge=0.0):
    """Charge equilibration: minimize E_ES(q) = chi.q + 0.5 q^T A q subject to
    sum(q) = total_charge, with A = coulomb_matrix + diag(hardness).

    Solved as a bordered linear system with a Lagrange multiplier for charge
    neutrality; torch.linalg.solve is differentiable, so charges (and the
    downstream electrostatic energy/forces) backprop cleanly through chi,
    hardness, positions (via coulomb_matrix) and cell.

    Returns (q, lagrange_multiplier).
    """
    n = chi.shape[0]
    device, dtype = chi.device, chi.dtype

    a = coulomb_matrix + torch.diag(hardness)
    ones = torch.ones(n, device=device, dtype=dtype)

    lhs = torch.zeros(n + 1, n + 1, device=device, dtype=dtype)
    lhs[:n, :n] = a
    lhs[:n, n] = ones
    lhs[n, :n] = ones

    rhs = torch.zeros(n + 1, device=device, dtype=dtype)
    rhs[:n] = -chi
    rhs[n] = total_charge

    sol = torch.linalg.solve(lhs, rhs)
    q, lam = sol[:n], sol[n]
    return q, lam


def electrostatic_energy(chi, q, coulomb_matrix, hardness):
    a = coulomb_matrix + torch.diag(hardness)
    return (chi * q).sum() + 0.5 * (q @ a @ q)
