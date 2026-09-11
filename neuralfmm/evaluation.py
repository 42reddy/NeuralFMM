"""Evaluation metrics that go beyond plain MAE -- MAE alone can look fine
while the model gets the *shape* of the potential energy surface or the
*direction* of forces wrong. See Evaluator.evaluate() for the full list.
"""
import json
from pathlib import Path

import numpy as np
import torch

from .fmm import NeuralFMM
from .les import LESModel

MODEL_REGISTRY = {"les": LESModel, "fmm": NeuralFMM}

# per-model-class name of the per-atom auxiliary array `compute()` returns,
# alongside "energy"/"e_local"/"e_long_range" -- LES has no atomic-feature
# vector to report and NeuralFMM has no latent charges (see fmm.model.NeuralFMM's
# docstring: "Charge prediction: none required"), so each model exposes
# exactly one of these two keys.
AUX_PRED_KEY = {"les": "latent_charges", "fmm": "atomic_features"}


def load_evaluator_from_checkpoint(checkpoint_dir, checkpoint_name="best.pt", device="cpu"):
    """Rebuild a model from a checkpoint saved by Trainer and wrap it in an
    Evaluator. checkpoint_dir must contain config.json (model_class,
    model_config, species_map, written by Trainer) and the checkpoint file
    itself."""
    checkpoint_dir = Path(checkpoint_dir)
    with open(checkpoint_dir / "config.json") as f:
        cfg = json.load(f)
    model_cls = MODEL_REGISTRY[cfg["model_class"]]
    model = model_cls(**cfg["model_config"])
    ckpt = torch.load(checkpoint_dir / checkpoint_name, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    return Evaluator(model, cfg["species_map"], device=device, model_class=cfg["model_class"])


class Evaluator:
    def __init__(self, model, species_map, device="cpu", model_class=None):
        self.model = model.to(device)
        self.model.eval()
        self.species_map = species_map
        self.device = device
        if model_class is None:
            model_class = "fmm" if isinstance(model, NeuralFMM) else "les"
        self.model_class = model_class
        self.aux_pred_key = AUX_PRED_KEY[model_class]

    # ---- prediction collection ----

    def collect_predictions(self, samples):
        energies_true, energies_pred = [], []
        forces_true, forces_pred = [], []
        n_atoms_list = []
        aux_pred, species_all = [], []

        for sample in samples:
            sys_ = sample.system.to(self.device)
            out = self.model.energy_and_forces(sys_.positions, sys_.species, sys_.cell)

            energies_true.append(sample.energy.item())
            energies_pred.append(out["energy"].item())
            forces_true.append(sample.forces.detach().cpu().numpy())
            forces_pred.append(out["forces"].detach().cpu().numpy())
            n_atoms_list.append(sys_.num_atoms())

            aux_pred.append(out[self.aux_pred_key].detach().cpu().numpy())
            species_all.append(sys_.species.cpu().numpy())

        return {
            "energies_true": np.array(energies_true),
            "energies_pred": np.array(energies_pred),
            "forces_true": forces_true,
            "forces_pred": forces_pred,
            "n_atoms": np.array(n_atoms_list),
            "aux_pred": aux_pred,
            "species": species_all,
        }

    # ---- metrics ----

    def energy_metrics(self, pred):
        e_true, e_pred, n = pred["energies_true"], pred["energies_pred"], pred["n_atoms"]
        err = e_pred - e_true
        err_per_atom = err / n
        return {
            "energy_MAE_total": float(np.mean(np.abs(err))),
            "energy_RMSE_total": float(np.sqrt(np.mean(err**2))),
            "energy_MAE_per_atom": float(np.mean(np.abs(err_per_atom))),
            "energy_RMSE_per_atom": float(np.sqrt(np.mean(err_per_atom**2))),
        }

    def force_metrics(self, pred):
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

    def energy_ranking_metrics(self, pred):
        from scipy.stats import spearmanr

        if len(pred["energies_true"]) < 3:
            return {"energy_spearman_rho": None}
        rho, _ = spearmanr(pred["energies_true"], pred["energies_pred"])
        return {"energy_spearman_rho": float(rho)}

    def aux_pred_metrics(self, pred):
        """Per-species, per-channel mean/std of the model's per-atom
        auxiliary array -- for LES, the latent "charges" (no physical charge
        or neutrality constraint to check, QEq is gone entirely); for
        NeuralFMM, the atomic features h_i handed to the octree (there is no
        charge concept in that architecture at all). Purely descriptive,
        included so a channel that has collapsed to ~0 for every species
        (dead/unused) is easy to spot. Report key is named after whichever
        aux array this model produces so LES/FMM reports stay visually
        distinct."""
        inv_map = {v: k for k, v in self.species_map.items()}
        all_species = np.concatenate(pred["species"])
        all_aux = np.concatenate(pred["aux_pred"])  # (total_atoms, C)

        per_species = {}
        for sid, symbol in inv_map.items():
            mask = all_species == sid
            if mask.sum() == 0:
                continue
            per_species[symbol] = {
                "mean_per_channel": all_aux[mask].mean(axis=0).tolist(),
                "std_per_channel": all_aux[mask].std(axis=0).tolist(),
            }

        return {f"per_species_{self.aux_pred_key}": per_species}

    def force_energy_consistency_check(self, samples, n_checks=5, eps=1e-4, seed=0):
        """Confirms autograd forces really are -dE/dR for this trained
        model: a correctness gate (should always pass to ~O(eps^2)), not an
        accuracy metric -- included because a training bug that corrupts
        this would otherwise be invisible in MAE numbers."""
        rng = np.random.default_rng(seed)
        errors = []
        for sample in samples[:n_checks]:
            sys_ = sample.system.to(self.device)
            out = self.model.energy_and_forces(sys_.positions, sys_.species, sys_.cell)
            direction = torch.tensor(
                rng.normal(size=sys_.positions.shape), dtype=sys_.positions.dtype, device=sys_.positions.device
            )
            direction = direction / direction.norm()

            predicted_de = -(out["forces"].detach() * direction).sum().item() * eps
            e0 = out["energy"].item()
            pos_step = sys_.positions.detach() + eps * direction
            e1 = self.model.compute(pos_step, sys_.species, sys_.cell)["energy"].item()
            actual_de = e1 - e0
            errors.append(abs(actual_de - predicted_de))
        return {
            "force_energy_consistency_max_abs_error": float(np.max(errors)),
            "force_energy_consistency_note": f"step size eps={eps}, error should scale ~eps^2",
        }

    def long_range_decay_test(self, max_separation=15.0, n_steps=8):
        """Two small synthetic clusters (built from whatever species this
        model was trained on) placed at increasing separation inside a large
        periodic box. Reports E_interaction(R) = E(both) - E(A alone) - E(B
        alone) as a function of separation R. Uses no dataset labels --
        directly probes whether the far-field pathway behaves sensibly
        (decays with distance rather than blowing up or staying flat), and
        is the metric to compare between an `les.LESModel` and an
        `fmm.NeuralFMM` checkpoint.
        """

        device = self.device

        def cluster(species_ids, center, spread, seed):
            rng = np.random.default_rng(seed)
            offsets = rng.normal(scale=spread, size=(len(species_ids), 3))
            positions = torch.tensor(center + offsets, dtype=torch.float32, device=device)
            species = torch.tensor(species_ids, dtype=torch.long, device=device)
            return positions, species

        species_ids = list(self.species_map.values())
        n_a, n_b = 3, 3
        cluster_a_species = [species_ids[i % len(species_ids)] for i in range(n_a)]
        cluster_b_species = [species_ids[(i + 1) % len(species_ids)] for i in range(n_b)]

        # generous margin beyond max_separation so the periodic image of one
        # cluster doesn't itself sit within interaction range of the other
        # and confound the decay curve we're trying to measure
        box = 2.5 * max_separation + 10.0
        cell = torch.eye(3, device=device) * box

        pos_a, spec_a = cluster(cluster_a_species, np.array([2.0, box / 2, box / 2]), 0.4, seed=1)
        pos_b_local, spec_b = cluster(cluster_b_species, np.array([0.0, 0.0, 0.0]), 0.4, seed=2)

        def energy_of(positions, species):
            return self.model.compute(positions, species, cell)["energy"].item()

        e_a = energy_of(pos_a, spec_a)
        far_offset = torch.tensor([box - 2.0, box / 2, box / 2], dtype=torch.float32, device=device)
        e_b = energy_of(pos_b_local + far_offset, spec_b)

        results = []
        combined_species = torch.cat([spec_a, spec_b])
        for r in np.linspace(3.0, max_separation, n_steps):
            offset = torch.tensor([2.0 + r, box / 2, box / 2], dtype=torch.float32, device=device)
            pos_b = pos_b_local + offset
            combined_pos = torch.cat([pos_a, pos_b], dim=0)
            e_ab = energy_of(combined_pos, combined_species)
            results.append((float(r), e_ab - e_a - e_b))

        return results

    # ---- top-level entry point ----

    def evaluate(self, samples, decay_test=False, decay_max_separation=15.0, decay_steps=8):
        pred = self.collect_predictions(samples)

        report = {}
        report.update(self.energy_metrics(pred))
        report.update(self.force_metrics(pred))
        report.update(self.energy_ranking_metrics(pred))
        report.update(self.aux_pred_metrics(pred))
        report.update(self.force_energy_consistency_check(samples))

        if decay_test:
            report["long_range_decay"] = self.long_range_decay_test(decay_max_separation, decay_steps)

        return report
