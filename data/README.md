# Bulk liquid water, RPBE-D3 (bundled benchmark)

`train-H2O_RPBE-D3.xyz` / `test-H2O_RPBE-D3.xyz` — 604 / 50 periodic
configurations of 64 H2O molecules (192 atoms: 64 O + 128 H) in a fixed
12.429 Å cubic box, DFT energies + forces at the RPBE-D3 level (D3 =
dispersion-corrected DFT, so the reference data itself contains real
dispersion physics, not just bare electrostatics). extxyz format, readable
with `ase.io.read(path, index=":")`.

Source: `data-benchmark/` in the
[ChengUCB/les_fit](https://github.com/ChengUCB/les_fit) repository
(companion data to Cheng, "Latent Ewald summation for machine learning of
long-range interactions," npj Comput. Mater. 11, 80 (2025),
[arXiv:2408.15165](https://arxiv.org/abs/2408.15165)). That repo credits the
water dataset to David Limmer's group. Cite the paper above if you use this
data.

## Why this dataset for this codebase

- **Periodic** — matches this codebase's PBC-only design (see
  `ARCHITECTURE.md`).
- **Dispersion-corrected (D3)** — the reference forces/energies actually
  contain dispersion physics, which is exactly the kind of "extra long-range
  kernel" `fmm.NeuralFMM`'s far-field energy channel is meant to learn and a
  fixed-form analytic kernel (`les.LESModel`'s Ewald baseline) structurally
  cannot capture.
- **Polar/H-bonded** — water's O/H electronegativity contrast gives the QEq
  charge-chemistry eval metric (see `scripts/eval.py`) something meaningful
  to check: trained O atoms should come out net-negative, H net-positive.
- **Right-sized for this first draft** — 192 atoms/frame runs in well under
  a second per structure (forward + forces + backward) on CPU with this
  repo's current O(N^2) Ewald/tree implementation (see the scaling caveats
  in `ARCHITECTURE.md`); a full epoch over 604 structures takes a few
  minutes on a laptop CPU.

`H2O_BEC.xyz` (also in that repo, not bundled here) additionally carries
Born effective charge tensors for the same kind of system — a natural
follow-up if you want a direct reference for the predicted QEq charges
instead of only the indirect chemistry check `scripts/eval.py` currently
does.
