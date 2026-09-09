"""Train NeuralFMM4GHDNN on an extxyz energy/forces dataset.

Example (the bundled bulk-water RPBE-D3 benchmark, see data/README.md):

    python scripts/train.py \
        --train-file data/train-H2O_RPBE-D3.xyz \
        --val-file data/test-H2O_RPBE-D3.xyz \
        --use-neural-fmm \
        --checkpoint-dir checkpoints/fmm \
        --epochs 20

Run twice, with and without --use-neural-fmm, to compare against the
4G-HDNN (local-only) baseline with scripts/eval.py.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neuralfmm import NeuralFMM4GHDNN
from neuralfmm.dataset import AtomicDataset, list_collate, load_extxyz


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train-file", required=True)
    p.add_argument("--val-file", required=True)
    p.add_argument("--max-train-samples", type=int, default=None)
    p.add_argument("--max-val-samples", type=int, default=None)
    p.add_argument("--total-charge", type=float, default=0.0)

    p.add_argument("--use-neural-fmm", action="store_true")
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--local-layers", type=int, default=3)
    p.add_argument("--n-rbf", type=int, default=8)
    p.add_argument("--local-r-cut", type=float, default=4.0)
    p.add_argument("--tree-depth", type=int, default=3)
    p.add_argument("--fmm-hidden-dim", type=int, default=32)
    p.add_argument("--fmm-blocks", type=int, default=2)
    p.add_argument("--operator-depth", type=int, default=2)
    p.add_argument("--ewald-alpha", type=float, default=0.35)
    p.add_argument("--ewald-r-cutoff", type=float, default=5.5)
    p.add_argument("--ewald-kmax", type=int, default=4)

    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--energy-weight", type=float, default=1.0)
    p.add_argument("--force-weight", type=float, default=100.0)
    p.add_argument("--grad-clip", type=float, default=10.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--checkpoint-dir", required=True)
    return p.parse_args()


def model_config_from_args(args, num_species: int) -> dict:
    return dict(
        num_species=num_species,
        hidden_dim=args.hidden_dim,
        local_layers=args.local_layers,
        n_rbf=args.n_rbf,
        local_r_cut=args.local_r_cut,
        use_neural_fmm=args.use_neural_fmm,
        tree_depth=args.tree_depth,
        fmm_hidden_dim=args.fmm_hidden_dim,
        fmm_blocks=args.fmm_blocks,
        operator_depth=args.operator_depth,
        ewald_alpha=args.ewald_alpha,
        ewald_r_cutoff=args.ewald_r_cutoff,
        ewald_kmax=args.ewald_kmax,
    )


def run_epoch(model, loader, device, args, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)

    total_loss, total_e_mae, total_f_mae, n_samples = 0.0, 0.0, 0.0, 0

    for batch in loader:
        if is_train:
            optimizer.zero_grad()

        batch_loss = 0.0
        for sample in batch:
            sys_ = sample.system.to(device)
            energy_true = sample.energy.to(device)
            forces_true = sample.forces.to(device)
            n_atoms = sys_.num_atoms

            out = model.energy_and_forces(sys_.positions, sys_.species, sys_.cell, sys_.total_charge)
            e_loss = ((out["energy"] - energy_true) / n_atoms) ** 2
            f_loss = ((out["forces"] - forces_true) ** 2).mean()
            loss = args.energy_weight * e_loss + args.force_weight * f_loss
            batch_loss = batch_loss + loss

            total_e_mae += (out["energy"] - energy_true).abs().item() / n_atoms
            total_f_mae += (out["forces"] - forces_true).abs().mean().item()
            n_samples += 1

        batch_loss = batch_loss / len(batch)

        if is_train:
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

        total_loss += batch_loss.item() * len(batch)

    return {
        "loss": total_loss / n_samples,
        "energy_mae_per_atom": total_e_mae / n_samples,
        "force_mae": total_f_mae / n_samples,
    }


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    train_samples, species_map = load_extxyz(
        args.train_file, total_charge=args.total_charge, max_samples=args.max_train_samples
    )
    val_samples, _ = load_extxyz(
        args.val_file, species_map=species_map, total_charge=args.total_charge, max_samples=args.max_val_samples
    )
    print(f"species map: {species_map}")
    print(f"train: {len(train_samples)} structures, val: {len(val_samples)} structures")

    train_loader = DataLoader(
        AtomicDataset(train_samples), batch_size=args.batch_size, shuffle=True, collate_fn=list_collate
    )
    val_loader = DataLoader(
        AtomicDataset(val_samples), batch_size=args.batch_size, shuffle=False, collate_fn=list_collate
    )

    cfg = model_config_from_args(args, num_species=len(species_map))
    model = NeuralFMM4GHDNN(**cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    with open(ckpt_dir / "config.json", "w") as f:
        json.dump({"model_config": cfg, "species_map": species_map}, f, indent=2)

    best_val = float("inf")
    for epoch in range(args.epochs):
        t0 = time.time()
        train_stats = run_epoch(model, train_loader, device, args, optimizer)
        val_stats = run_epoch(model, val_loader, device, args, optimizer=None)
        scheduler.step(val_stats["loss"])
        dt = time.time() - t0

        print(
            f"epoch {epoch:3d} ({dt:5.1f}s) "
            f"train loss {train_stats['loss']:.4f} E_MAE/atom {train_stats['energy_mae_per_atom']:.4f} F_MAE {train_stats['force_mae']:.4f} | "
            f"val loss {val_stats['loss']:.4f} E_MAE/atom {val_stats['energy_mae_per_atom']:.4f} F_MAE {val_stats['force_mae']:.4f}"
        )

        torch.save({"model_state": model.state_dict(), "epoch": epoch}, ckpt_dir / "last.pt")
        if val_stats["loss"] < best_val:
            best_val = val_stats["loss"]
            torch.save({"model_state": model.state_dict(), "epoch": epoch}, ckpt_dir / "best.pt")

    print(f"done. best val loss {best_val:.4f}. checkpoints in {ckpt_dir}")


if __name__ == "__main__":
    main()
