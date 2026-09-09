"""Dataset prep -> model init -> training -> evaluation,

Run this twice -- once with USE_NEURAL_FMM = True, once False -- and compare
the two checkpoint directories' eval reports to see what the far-field
pathway is (or isn't) buying you over the plain 4G-HDNN baseline.
"""
import json

import torch

from neuralfmm import NeuralFMM4GHDNN
from neuralfmm.dataset import prepare_water_dataset
from neuralfmm.evaluation import Evaluator
from neuralfmm.training import TrainConfig, Trainer

# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------
DATA_DIR = "data"
TOTAL_CHARGE = 0.0
MAX_TRAIN_SAMPLES = None  # None = use all 604 training structures
MAX_VAL_SAMPLES = None  # None = use all 50 test structures

# ----------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------
USE_NEURAL_FMM = True

MODEL_HYPERPARAMS = dict(
    hidden_dim=512,
    local_layers=4,
    n_rbf=8,
    local_r_cut=5.0,
    use_neural_fmm=USE_NEURAL_FMM,
    tree_depth=4,
    fmm_hidden_dim=128,
    fmm_blocks=4,
    operator_depth=4,
    ewald_alpha=0.35,
    ewald_r_cutoff=5.5,
    ewald_kmax=4,
)

# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------
CHECKPOINT_DIR = "checkpoints/fmm" if USE_NEURAL_FMM else "checkpoints/local_only"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TRAIN_CONFIG = TrainConfig(
    epochs=20,
    batch_size=4,
    lr=1e-3,
    energy_weight=1.0,
    force_weight=100.0,
    grad_clip=10.0,
    log_every=1,
    checkpoint_dir=CHECKPOINT_DIR,
    device=DEVICE,
    seed=0,
)

# ----------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------
RUN_DECAY_TEST = True
DECAY_MAX_SEPARATION = 15.0
DECAY_STEPS = 8


def main():
    train_dataset, val_dataset, species_map = prepare_water_dataset(
        DATA_DIR, TOTAL_CHARGE, MAX_TRAIN_SAMPLES, MAX_VAL_SAMPLES
    )
    print(f"species map: {species_map}")
    print(f"train: {len(train_dataset)} structures, val: {len(val_dataset)} structures")
    print(f"device: {TRAIN_CONFIG.device}")

    model_config = {**MODEL_HYPERPARAMS, "num_species": len(species_map)}
    model = NeuralFMM4GHDNN(**model_config)

    trainer = Trainer(
        model=model,
        model_config=model_config,
        species_map=species_map,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=TRAIN_CONFIG,
    )
    trainer.fit()

    evaluator = Evaluator(model, species_map, device=TRAIN_CONFIG.device)
    report = evaluator.evaluate(
        val_dataset.samples,
        decay_test=RUN_DECAY_TEST,
        decay_max_separation=DECAY_MAX_SEPARATION,
        decay_steps=DECAY_STEPS,
    )

    print("\n=== Evaluation report ===")
    print(json.dumps(report, indent=2))

    report_path = trainer.checkpoint_dir / "eval_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nsaved report to {report_path}")


if __name__ == "__main__":
    main()
