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
# alongside "energy"/"e_local"/"e_coulomb" -- LES has no atomic-feature
# vector to report, so it's reported by its latent charges instead.
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

    def vacuum_padding_test(
        self, sample, n_molecules=8, box_lengths=(15.0, 20.0, 30.0, 45.0, 65.0, 90.0), o_symbol="O", h_symbol="H"
    ):
        """Non-periodic/free-space probe built entirely from the existing
        periodic bulk-water dataset and a trained checkpoint -- no new
        simulation, no new labels. Carves a compact, non-periodic droplet of
        `n_molecules` water molecules out of one bulk configuration (see
        `cluster.extract_water_cluster`) and re-evaluates that SAME fixed
        geometry inside cubic cells of growing `box_lengths`, i.e. growing
        amounts of vacuum padding around one physical cluster.

        The true energy of an isolated cluster does not depend on an
        arbitrary padding choice once the box is large enough that the
        cluster's own periodic images are irrelevant -- so any drift in the
        reported energy across `box_lengths` is purely an artifact of how
        an architecture's periodicity assumption behaves as it is pushed
        toward the free-space limit neither `les.LESModel` nor
        `fmm.NeuralFMM` was trained on. `les.LESModel`'s reciprocal-space
        kernel sums over a k-grid tied to the box volume at a FIXED number
        of shells (`ewald_kmax`), so inflating the box while holding kmax
        fixed shrinks the resolvable k-range and is expected to visibly
        drift; `fmm.NeuralFMM`'s octree only ever stores OCCUPIED boxes, so
        the empty padding region simply contributes nothing and the
        prediction is expected to plateau as soon as the box exceeds the
        cluster's own extent -- this is the concrete, reusable-data
        instantiation of that hypothesis. Both architectures also expose
        `e_coulomb`, the explicit analytic Ewald term shared by both (see
        `fmm.model.NeuralFMM`'s docstring) -- the single most
        periodicity-sensitive piece of either model, and the fairest
        like-for-like slice to compare.
        """
        from .cluster import extract_water_cluster, pad_into_vacuum

        device = self.device
        o_id = self.species_map[o_symbol]
        h_id = self.species_map[h_symbol]

        sys_ = sample.system.to(device)
        cluster_pos, cluster_species = extract_water_cluster(
            sys_.positions, sys_.species, sys_.cell, o_id, h_id, n_molecules
        )

        results = []
        for box_length in box_lengths:
            positions, cell = pad_into_vacuum(cluster_pos, box_length)
            with torch.no_grad():
                out = self.model.compute(positions, cluster_species, cell)
            entry = {
                "box_length": float(box_length),
                "energy_per_molecule": out["energy"].item() / n_molecules,
            }
            if "e_coulomb" in out:
                entry["e_coulomb_per_molecule"] = out["e_coulomb"].item() / n_molecules
            if "e_long_range" in out:
                entry["e_long_range_per_molecule"] = out["e_long_range"].item() / n_molecules
            results.append(entry)

        reference = results[-1]["energy_per_molecule"]  # largest box = closest to the free-space limit
        for entry in results:
            entry["drift_from_largest_box"] = entry["energy_per_molecule"] - reference

        return results

    def size_transfer_test(self, samples_large):
        """Evaluate this checkpoint -- trained at one fixed box size/density
        (e.g. NACL_ATOM_COUNT=194 atoms, see data.py) -- on an independently
        DFT-computed LARGER box of the same stoichiometry/density it never
        trained on (e.g. dataset.prepare_nacl_size_transfer_dataset's
        449-atom set). Real DFT labels exist at this size (unlike
        `density_scan_test` below), so this reports actual accuracy there,
        directly comparable to the in-distribution val report from
        `evaluate()` -- the headline comparison this project is built
        around: `les.LESModel`'s reciprocal-space kernel sums over a k-grid
        tied to the box volume at a FIXED number of shells (`ewald_kmax`),
        so a bigger box at training-time resolution is expected to degrade
        its long-range term; `fmm.NeuralFMM`'s octree re-partitions space
        into occupied boxes at whatever size it's given, so it's expected to
        transfer with much less drift. Whether that's actually true is an
        empirical question this method answers, not an assumption it bakes
        in."""
        pred = self.collect_predictions(samples_large)
        report = {}
        report.update(self.energy_metrics(pred))
        report.update(self.force_metrics(pred))
        report.update(self.energy_ranking_metrics(pred))
        return report

    def density_scan_test(self, sample, scale_factors=(0.90, 0.95, 1.0, 1.05, 1.1, 1.2)):
        """Synthetic isotropic compression/expansion probe: ONE real
        structure's positions and cell are both scaled by the same factor
        (an isotropic dilation preserves fractional coordinates exactly, so
        this is a clean density change with no other structural change),
        and re-evaluated with no retraining. No DFT reference exists at
        these synthetic densities, so -- like `vacuum_padding_test` -- this
        reports *drift* relative to the structure's own real density
        (scale=1.0) rather than accuracy against ground truth. Same
        hypothesis as `size_transfer_test`, complementary axis: does the
        long-range term react sensibly to a density it never trained on, or
        does whichever fixed-resolution assumption it carries (LES's
        k-space grid; FMM's hard box-boundary partition) show up as drift?
        """
        device = self.device
        sys_ = sample.system.to(device)
        n_atoms = sys_.num_atoms()

        results = []
        for scale in scale_factors:
            positions = sys_.positions * scale
            cell = sys_.cell * scale
            with torch.no_grad():
                out = self.model.compute(positions, sys_.species, cell)
            entry = {"scale": float(scale), "energy_per_atom": out["energy"].item() / n_atoms}
            if "e_coulomb" in out:
                entry["e_coulomb_per_atom"] = out["e_coulomb"].item() / n_atoms
            if "e_long_range" in out:
                entry["e_long_range_per_atom"] = out["e_long_range"].item() / n_atoms
            results.append(entry)

        reference = next((e["energy_per_atom"] for e in results if e["scale"] == 1.0), results[0]["energy_per_atom"])
        for entry in results:
            entry["drift_from_scale_1"] = entry["energy_per_atom"] - reference

        return results

    # ---- top-level entry point ----

    def evaluate(
        self,
        samples,
        decay_test=False,
        decay_max_separation=15.0,
        decay_steps=8,
        vacuum_test=False,
        vacuum_n_molecules=8,
        vacuum_box_lengths=(15.0, 20.0, 30.0, 45.0, 65.0, 90.0),
        size_transfer_samples=None,
        density_scan=False,
        density_scale_factors=(0.90, 0.95, 1.0, 1.05, 1.1, 1.2),
    ):
        pred = self.collect_predictions(samples)

        report = {}
        report.update(self.energy_metrics(pred))
        report.update(self.force_metrics(pred))
        report.update(self.energy_ranking_metrics(pred))
        report.update(self.aux_pred_metrics(pred))
        report.update(self.force_energy_consistency_check(samples))

        if decay_test:
            report["long_range_decay"] = self.long_range_decay_test(decay_max_separation, decay_steps)

        if vacuum_test:
            report["vacuum_padding"] = self.vacuum_padding_test(samples[0], vacuum_n_molecules, vacuum_box_lengths)

        if size_transfer_samples is not None:
            report["size_transfer"] = self.size_transfer_test(size_transfer_samples)

        if density_scan:
            report["density_scan"] = self.density_scan_test(samples[0], density_scale_factors)

        return report
