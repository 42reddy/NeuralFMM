# Neural FMM for MLIPs — first draft

## What this is, and a correction on the source paper

The request was to adapt "Neural FMM" (arXiv:2509.20591, Fognini/Betcke/Cox,
UCL) to MLIP long-range electrostatics as an alternative to Ewald message
passing. Worth flagging clearly: **that paper is about learning Green's
operators for the 2D Helmholtz equation (acoustic scattering)** — it has no
electronegativity, QEq, or MLIP content. An earlier automated web-summary of
it hallucinated an "Electronegativity/Charges" section that doesn't exist; I
caught this by pulling and reading the actual PDF text (`pdftotext`) before
writing any code. So the electronegativity/QEq/MLIP application here is your
own idea, not something the paper specifies — what's ported faithfully from
the paper is its **tree mechanism**: the Upward (M2M) / far-field translation
(M2L) / Downward (L2L) / Leaf pass structure of Eqs. 4–7 and Fig. 1 of
2509.20591, with every analytic translation operator replaced by a learned
MLP, applied per tree level, with RoPE position encoding over each box's
Morton code. That structure is implemented as-is on an atomic point cloud
instead of a PDE grid; see `neuralfmm/fmm/`.

## Pipeline

```
positions, species, cell (periodic)
        │
        ▼
  PaiNN local equivariant GNN  (neuralfmm/local/painn.py)
        │  per-atom invariant scalar descriptor s_i
        │
        ├─────────────────────────────┐
        ▼                              ▼ (only if use_neural_fmm)
 local chi_i, hardness J_i      Deep Neural FMM tree pass
 (4G-HDNN style, elemental      (neuralfmm/fmm/): periodic octree,
  baseline + local MLP)         M2M / M2L / L2L, all learned
        │                              │
        │                     delta_chi_i, E_far_i (2 channels,
        │                     zero-init -> no-op at init)
        └──────────► chi_i = chi_local_i (+ delta_chi_i) ◄────┘
                              │
                              ▼
                    QEq linear solve (neuralfmm/qeq.py)
                    A q = -chi,  A = Ewald_matrix + diag(J)
                    constrained to sum(q) = total_charge
                              │
                              ▼
        E_total = E_local(s_i) + E_electrostatic(q, chi, A) + E_far
                              │
                              ▼
                  forces = -dE_total/dR   (torch.autograd)
```

The boolean switch is `NeuralFMM4GHDNN.use_neural_fmm`:
- `False` → exactly a 4G-HDNN baseline: chi_i is a function of the local
  atomic environment only, QEq still gives nonlocal charge transfer, energy
  is local + proper periodic-Ewald electrostatics. No dispersion channel.
- `True` → chi_i additionally receives a correction from the far-field tree,
  and a second, directly-predicted long-range energy channel is added. This
  is deliberately *not* a second analytic kernel (e.g. a hand-coded C6/r^6
  term) — the far-field tree has no functional form imposed on it beyond the
  M2M/M2L/L2L information flow, so whatever long-range physics is present in
  training data (dispersion, induction, etc.) is what it has to learn.

## Why real Coulomb law stays analytic

The point of "let the NN learn multiple kernels instead of one analytic
kernel" is honored for *predicting electronegativities* (what determines how
much charge ends up where) and for the *extra* long-range energy channel
(whatever isn't plain 1/r electrostatics). It is **not** applied to the
actual charge-charge Coulomb interaction once charges exist — that's exact,
known physics (`neuralfmm/electrostatics.py` does a standard periodic Ewald
sum, real space erfc + reciprocal space + self + charged-cell corrections),
and there's no reason to make QEq itself replace that with something learned.
This mirrors how the paper's own Neural FMM keeps the *tree information
flow* fixed while replacing only the *coefficients* (T_ofs, T_ofo, T_ifi,
T_ifo, T_tfi) with MLPs.

## Local part (`neuralfmm/local/`)

PaiNN-style (Schütt et al., 2021): scalar features `s_i` (invariant) and
vector features `v_i` (equivariant, transform as vectors under O(3)) updated
via alternating message and update blocks, using only pairwise distances,
direction unit vectors, and linear/elementwise-gated combinations of
vectors — this is what keeps `v_i` a true equivariant vector while letting
`s_i` feed downstream heads as a rotation-invariant descriptor. Chosen over
e3nn/NequIP-style higher-order tensor products for this first draft since it
needs no extra heavy dependency and is simpler to get right; swapping in a
full e3nn backbone later only requires replacing `PaiNN` with something that
exposes the same `(positions, species, cell) -> per-atom invariant features`
interface.

Periodic neighbor list (`local/neighbors.py`) is a brute-force periodic-image
search (correct, `O(N^2 * n_images)`, fine at toy scale — swap for a cell
list before scaling up).

## Far-field tree (`neuralfmm/fmm/`)

- `octree.py`: a **periodic** (torus) sparse octree over the fractional
  coordinates of the atoms, built fresh each forward call from the current
  positions. Bit-interleaved (Morton) box codes give both a natural 1D
  ordering for RoPE and a free parent/child relation (`parent_code = code >>
  3`). Near/far separation and the FMM interaction list `U_τ` follow the
  classical construction (children of parent's near-neighbors, minus your
  own near-neighbors), applied only from level 2 down (levels 0–1 are too
  coarse to separate near/far, matching the paper's own note).
- `blocks.py`: `NeuralFMMBlock` is one Upward→Downward→Leaf pass (Eqs. 4–7);
  `DeepNeuralFMM` stacks several of these with a local linear residual +
  nonlinearity between them, mirroring the paper's Neural-Operator-style
  layout (lifting `P`, blocks with local `W_t` + nonlocal `K_t`, projection
  `Q`, Eq. 1).
- `rope.py`: standard rotary embedding, applied to every box's feature
  vector using its Morton code as the position, exactly as the paper's
  position-encoding scheme (Sec. 3.2).

**Design correspondence you should know about when picking hyperparameters:**
this codebase does *not* separately implement the classical FMM's explicit
near-field direct sum inside the tree. Instead, "near field" is the local
GNN's job (whatever the PaiNN cutoff sees), and "far field" is the tree's
job. For that split to make physical sense, choose `local_r_cut` and
`tree_depth` together so the leaf-box size (`cell_size / 2**tree_depth`) is
comparable to `local_r_cut` — otherwise you get either a gap (neither module
sees some interaction range) or heavy double-counting.

## QEq (`neuralfmm/qeq.py`)

Standard charge equilibration: minimizes `E_ES(q) = chi·q + 0.5 q^T A q`
subject to `sum(q) = total_charge`, solved as a bordered linear system with a
Lagrange multiplier via `torch.linalg.solve` — which is differentiable, so
gradients (forces) flow cleanly back through the charges into `chi`,
`hardness`, positions (via the Ewald matrix) and, when the switch is on,
through the whole far-field tree.

## Known limitations of this first draft (read before running real experiments)

1. **Box-assignment discontinuity.** Which leaf box an atom belongs to is a
   discrete function of its position, built with `positions.detach()`
   on purpose (see the docstring in `octree.py`). Forces computed via
   autograd are therefore *exact for the tree topology at the evaluated
   geometry*, but as an atom crosses a box boundary between MD steps, chi_i
   can jump discontinuously the way a hard neighbor-list cutoff would if it
   had no smooth envelope. In practice the local GNN's cosine envelope
   smooths its own contribution, but the far-field path currently has no
   analogous smoothing. Finite-difference force checks pass here (see
   `tests/test_model.py`) because the probe step doesn't cross a box
   boundary — this is a real caveat for longer MD trajectories, not a bug in
   what's implemented. A natural follow-up is a soft/overlapping partition
   of unity across box boundaries.
2. **No batching yet.** `NeuralFMM4GHDNN.compute` / `.energy_and_forces` take
   one structure at a time. Batch training by looping python-side and
   summing losses; a disjoint-graph batched tree is a reasonable follow-up
   once the single-structure version is validated.
3. **`O(N^2)` Ewald and brute-force tree/neighbor-list construction.** Fine
   for the toy systems this draft targets; swap in neighbor-list-based real
   space, PME/FFT reciprocal space, and a proper sparse tree before scaling
   to hundreds+ of atoms.
4. **General triclinic cells are supported by the neighbor list and Ewald
   sum, but the octree divides the *fractional* coordinate cube uniformly**,
   so near/far separation is only approximate (not perfectly isotropic) for
   strongly non-orthorhombic cells.
5. **No stress tensor / cell gradients** — only atomic forces. Needed if you
   want to train on/relax cell shape.
6. Ewald `alpha`/`r_cutoff`/`kmax` and tree `depth`/`operator_depth` are
   exposed as hyperparameters but not auto-tuned for a given system size or
   accuracy target.

## Validated so far

- Finite-difference forces match autograd forces to ~1e-8 (float64), for
  both `use_neural_fmm=False` and `=True`, including a second derivative
  (double-backward) fix needed for two `torch.norm` calls whose gradients
  are singular at/near zero (PaiNN's `Vv_norm`, the Ewald real-space
  self-distance) — see git history / code comments at those two spots.
- Charge neutrality constraint holds to ~1e-6.
- Translation invariance (arbitrary shift, and shift by exactly one lattice
  vector) holds to ~1e-6.
- `pytest tests/` — 6/6 passing (`tests/test_model.py`).

## Running it

```
pip install -r requirements.txt
pytest tests/
```

`neuralfmm.NeuralFMM4GHDNN(...)` is the entry point; see
`tests/test_model.py` for a minimal usage example on a random toy
configuration. Next step (not yet done): a real toy system generator (e.g. a
small ionic crystal or polar-molecule cluster where long-range electrostatics
and dispersion actually matter) and a training loop comparing
`use_neural_fmm=False` vs `True` against a reference method.
