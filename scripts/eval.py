"""Evaluate a trained NeuralFMM4GHDNN checkpoint with metrics that go beyond
plain MAE -- MAE alone can look fine while the model gets the *shape* of the
potential energy surface, the *direction* of forces, or the basic chemistry
of the predicted charges wrong. Metrics reported:

  1. Energy / force MAE & RMSE           -- baseline accuracy numbers
  2. Force cosine similarity             -- are forces pointing the right way,
                                             independent of magnitude error
  3. Spearman rank correlation of energy -- does the model get the *ordering*
                                             of configurations by energy right
                                             (the part that matters for MD/
                                             relaxation, more than absolute
                                             offset)
  4. Force/energy consistency check      -- confirms autograd forces are
                                             actually -dE/dR for this trained
                                             model to the precision autograd
                                             promises (a correctness gate,
                                             not an accuracy number)
  5. Predicted charge chemistry          -- per-species charge mean/std and
                                             neutrality residual: e.g. do O
                                             atoms come out net-negative and H
                                             net-positive in water, as basic
                                             electronegativity ordering
                                             demands, even though this
                                             dataset has no reference charges
                                             to fit against directly
  6. (--decay-test) long-range asymptotic decay of a synthetic separated
     two-cluster system -- the one diagnostic that specifically probes what
     the Neural FMM switch is *for*: does interaction energy fall off
     sensibly with distance, and does switching use_neural_fmm on change it

Example:
    python scripts/eval.py --checkpoint-dir checkpoints/fmm --test-file data/test-H2O_RPBE-D3.xyz
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neuralfmm import NeuralFMM4GHDNN
from neuralfmm.data import AtomicSystem
from neuralfmm.dataset import load_extxyz


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint-dir", required=True)
    p.add_argument("--checkpoint-name", default="best.pt")
    p.add_argument("--test-file", required=True)
    p.add_argument("--max-test-samples", type=int, default=None)
    p.add_argument("--total-charge", type=float, default=0.0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--decay-test", action="store_true")
    p.add_argument("--decay-max-separation", type=float, default=15.0)
    p.add_argument("--decay-steps", type=int, default=8)
    return p.parse_args()


def load_model(checkpoint_dir: Path, checkpoint_name: str, device):
    with open(checkpoint_dir / "config.json") as f:
        cfg = json.load(f)
    model = NeuralFMM4GHDNN(**cfg["model_config"]).to(device)
    ckpt = torch.load(checkpoint_dir / checkpoint_name, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, cfg["species_map"]


def collect_predictions(model, samples, device):
    energies_true, energies_pred = [], []
    forces_true, forces_pred = [], []
    n_atoms_list = []
    charges_pred, species_all, neutrality_residual = [], [], []

    for sample in samples:
        sys_ = sample.system.to(device)
        out = model.energy_and_forces(sys_.positions, sys_.species, sys_.cell, sys_.total_charge)

        energies_true.append(sample.energy.item())
        energies_pred.append(out["energy"].item())
        forces_true.append(sample.forces.detach().cpu().numpy())
        forces_pred.append(out["forces"].detach().cpu().numpy())
        n_atoms_list.append(sys_.num_atoms)

        q = out["charges"].detach().cpu().numpy()
        charges_pred.append(q)
        species_all.append(sys_.species.cpu().numpy())
        neutrality_residual.append(q.sum())

    return {
        "energies_true": np.array(energies_true),
        "energies_pred": np.array(energies_pred),
        "forces_true": forces_true,
        "forces_pred": forces_pred,
        "n_atoms": np.array(n_atoms_list),
        "charges_pred": charges_pred,
        "species": species_all,
        "neutrality_residual": np.array(neutrality_residual),
    }


def energy_metrics(pred):
    e_true, e_pred, n = pred["energies_true"], pred["energies_pred"], pred["n_atoms"]
    err = e_pred - e_true
    err_per_atom = err / n
    return {
        "energy_MAE_total": float(np.mean(np.abs(err))),
        "energy_RMSE_total": float(np.sqrt(np.mean(err**2))),
        "energy_MAE_per_atom": float(np.mean(np.abs(err_per_atom))),
        "energy_RMSE_per_atom": float(np.sqrt(np.mean(err_per_atom**2))),
    }


def force_metrics(pred):
    f_true = np.concatenate([f.reshape(-1, 3) for f in pred["forces_true"]], axis=0)
    f_pred = np.concatenate([f.reshape(-1, 3) for f in pred["forces_pred"]], axis=0)
    err = f_pred - f_true

    mag_true = np.linalg.norm(f_true, axis=-1)
    mag_pred = np.linalg.norm(f_pred, axis=-1)
    cos_sim = np.sum(f_true * f_pred, axis=-1) / (mag_true * mag_pred + 1e-8)
    # undefined direction when the reference force is ~0; exclude those atoms
    valid = mag_true > 1e-3

    return {
        "force_MAE": float(np.mean(np.abs(err))),
        "force_RMSE": float(np.sqrt(np.mean(err**2))),
        "force_cosine_similarity_mean": float(np.mean(cos_sim[valid])),
        "force_cosine_similarity_median": float(np.median(cos_sim[valid])),
        "force_well_directed_frac_cos>0.9": float(np.mean(cos_sim[valid] > 0.9)),
        "force_magnitude_relative_error_median": float(
            np.median(np.abs(mag_pred[valid] - mag_true[valid]) / mag_true[valid])
        ),
    }


def energy_ranking_metrics(pred):
    from scipy.stats import spearmanr

    if len(pred["energies_true"]) < 3:
        return {"energy_spearman_rho": None}
    rho, _ = spearmanr(pred["energies_true"], pred["energies_pred"])
    return {"energy_spearman_rho": float(rho)}


def charge_chemistry_metrics(pred, species_map):
    inv_map = {v: k for k, v in species_map.items()}
    all_species = np.concatenate(pred["species"])
    all_charges = np.concatenate(pred["charges_pred"])

    per_species = {}
    for sid, symbol in inv_map.items():
        mask = all_species == sid
        if mask.sum() == 0:
            continue
        per_species[symbol] = {
            "mean_charge": float(all_charges[mask].mean()),
            "std_charge": float(all_charges[mask].std()),
        }

    return {
        "per_species_charge": per_species,
        "neutrality_residual_max_abs": float(np.max(np.abs(pred["neutrality_residual"]))),
        "neutrality_residual_mean_abs": float(np.mean(np.abs(pred["neutrality_residual"]))),
    }


def force_energy_consistency_check(model, samples, device, n_checks=5, eps=1e-4, seed=0):
    """Confirms autograd forces really are -dE/dR for this trained model: a
    correctness gate (should always pass to ~O(eps^2)), not an accuracy
    metric -- included because a training bug that corrupts this would
    otherwise be invisible in MAE numbers."""
    rng = np.random.default_rng(seed)
    errors = []
    for sample in samples[:n_checks]:
        sys_ = sample.system.to(device)
        out = model.energy_and_forces(sys_.positions, sys_.species, sys_.cell, sys_.total_charge)
        direction = torch.tensor(rng.normal(size=sys_.positions.shape), dtype=sys_.positions.dtype)
        direction = direction / direction.norm()

        predicted_de = -(out["forces"].detach() * direction).sum().item() * eps
        e0 = out["energy"].item()
        pos_step = sys_.positions.detach() + eps * direction
        e1 = model.compute(pos_step, sys_.species, sys_.cell, sys_.total_charge)["energy"].item()
        actual_de = e1 - e0
        errors.append(abs(actual_de - predicted_de))
    return {
        "force_energy_consistency_max_abs_error": float(np.max(errors)),
        "force_energy_consistency_note": f"step size eps={eps}, error should scale ~eps^2",
    }


def _cluster_system(species_ids, center, spread, seed):
    rng = np.random.default_rng(seed)
    n = len(species_ids)
    offsets = rng.normal(scale=spread, size=(n, 3))
    positions = torch.tensor(center + offsets, dtype=torch.float32)
    species = torch.tensor(species_ids, dtype=torch.long)
    return positions, species


def long_range_decay_test(model, species_map, device, max_sep, n_steps):
    """Two small synthetic clusters (built from whatever species this model
    was trained on) placed at increasing separation inside a large periodic
    box. Reports E_interaction(R) = E(both) - E(A alone) - E(B alone) as a
    function of separation R. This does not use any dataset labels -- it
    directly probes whether the far-field pathway behaves sensibly (decays
    with distance rather than blowing up or staying flat), and is the metric
    to compare between a use_neural_fmm=False and =True checkpoint.
    """
    species_ids = list(species_map.values())
    n_a, n_b = 3, 3
    cluster_a_species = [species_ids[i % len(species_ids)] for i in range(n_a)]
    cluster_b_species = [species_ids[(i + 1) % len(species_ids)] for i in range(n_b)]

    # generous margin beyond max_sep so the periodic image of one cluster
    # doesn't itself sit within interaction range of the other and confound
    # the decay curve we're trying to measure
    box = 2.5 * max_sep + 10.0
    cell = torch.eye(3) * box

    pos_a, spec_a = _cluster_system(cluster_a_species, np.array([2.0, box / 2, box / 2]), 0.4, seed=1)
    pos_b_local, spec_b = _cluster_system(cluster_b_species, np.array([0.0, 0.0, 0.0]), 0.4, seed=2)

    def energy_of(positions, species):
        out = model.compute(positions, species, cell, total_charge=0.0)
        return out["energy"].item()

    e_a = energy_of(pos_a, spec_a)

    far_offset = np.array([box - 2.0, box / 2, box / 2])
    e_b = energy_of(pos_b_local + torch.tensor(far_offset - np.array([0.0, 0.0, 0.0]), dtype=torch.float32), spec_b)

    results = []
    separations = np.linspace(3.0, max_sep, n_steps)
    combined_species = torch.cat([spec_a, spec_b])
    for r in separations:
        offset = np.array([2.0 + r, box / 2, box / 2])
        pos_b = pos_b_local + torch.tensor(offset, dtype=torch.float32)
        combined_pos = torch.cat([pos_a, pos_b], dim=0)
        e_ab = energy_of(combined_pos, combined_species)
        results.append((float(r), e_ab - e_a - e_b))

    return results


def main():
    args = parse_args()
    device = torch.device(args.device)
    ckpt_dir = Path(args.checkpoint_dir)

    model, species_map = load_model(ckpt_dir, args.checkpoint_name, device)
    test_samples, _ = load_extxyz(
        args.test_file, species_map=species_map, total_charge=args.total_charge, max_samples=args.max_test_samples
    )
    print(f"loaded {len(test_samples)} test structures, species_map={species_map}")
    print(f"use_neural_fmm={model.use_neural_fmm}")

    pred = collect_predictions(model, test_samples, device)

    report = {}
    report.update(energy_metrics(pred))
    report.update(force_metrics(pred))
    report.update(energy_ranking_metrics(pred))
    report.update(charge_chemistry_metrics(pred, species_map))
    report.update(force_energy_consistency_check(model, test_samples, device))

    print("\n=== Evaluation report ===")
    print(json.dumps(report, indent=2))

    if args.decay_test:
        print("\n=== Long-range decay diagnostic ===")
        curve = long_range_decay_test(model, species_map, device, args.decay_max_separation, args.decay_steps)
        for r, e_int in curve:
            print(f"  R={r:6.2f} Ang   E_interaction={e_int: .6f} eV")

    out_path = ckpt_dir / "eval_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nsaved report to {out_path}")


if __name__ == "__main__":
    main()
