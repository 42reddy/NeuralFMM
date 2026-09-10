import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .dataset import list_collate
from .fmm.octree import merge_trees


class TrainConfig:

    def __init__(
        self,
        epochs=20,
        batch_size=4,
        lr=1e-3,
        energy_weight=1.0,
        force_weight=100.0,
        grad_clip=10.0,
        log_every=1,
        checkpoint_dir="checkpoints",
        device="cpu",
        seed=0,
    ):
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.energy_weight = energy_weight
        self.force_weight = force_weight
        self.grad_clip = grad_clip
        self.log_every = log_every
        self.checkpoint_dir = checkpoint_dir
        self.device = device
        self.seed = seed


class Trainer:
    """Per atom energy MSE + force MSE training loop."""

    def __init__(self, model, model_config, species_map, train_dataset, val_dataset, config):
        torch.manual_seed(config.seed)
        self.model = model.to(config.device)
        self.model_config = model_config
        self.species_map = species_map
        self.config = config

        self.train_loader = DataLoader(
            train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=list_collate
        )
        self.val_loader = DataLoader(
            val_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=list_collate
        )

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=3
        )

        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        with open(self.checkpoint_dir / "config.json", "w") as f:
            json.dump({"model_config": model_config, "species_map": species_map}, f, indent=2)

    def run_epoch(self, loader, train, desc=None):
        self.model.train(train)
        device = self.config.device
        cfg = self.config

        total_loss = torch.zeros((), device=device)
        total_e_mae = torch.zeros((), device=device)
        total_f_mae = torch.zeros((), device=device)
        n_samples = 0

        pbar = tqdm(loader, desc=desc, leave=False, unit="batch", dynamic_ncols=True, mininterval=0.3)
        for batch in pbar:
            if train:
                self.optimizer.zero_grad()

            positions_list, species_list, cell_list = [], [], []
            n_atoms_list, trees, local_graphs = [], [], []
            energy_true = torch.empty(len(batch), device=device)
            forces_true_list = []

            for i, sample in enumerate(batch):

                sys_ = sample.system.to_cached(device)
                energy_t, forces_t = sample.targets_on(device)

                positions_list.append(sys_.positions)
                species_list.append(sys_.species)
                cell_list.append(sys_.cell)
                n_atoms_list.append(sys_.num_atoms())
                energy_true[i] = energy_t
                forces_true_list.append(forces_t)

                local_graphs.append(sample.system.get_neighbor_graph(self.model.local.r_cut, device))
                if self.model.use_neural_fmm:
                    trees.append(sample.system.get_octree(self.model.tree_depth, device))

            tree = merge_trees(trees) if trees else None
            out = self.model.energy_and_forces_batched(
                positions_list, species_list, cell_list, tree=tree, local_graphs=local_graphs
            )

            n_atoms = torch.tensor(n_atoms_list, device=device, dtype=out["energy"].dtype)
            e_loss = ((out["energy"] - energy_true) / n_atoms) ** 2  # (B,)
            f_loss = torch.stack(
                [((f - ft) ** 2).mean() for f, ft in zip(out["forces"], forces_true_list)]
            )  # (B,)
            per_sample_loss = cfg.energy_weight * e_loss + cfg.force_weight * f_loss
            batch_loss = per_sample_loss.mean()

            if train:
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
                self.optimizer.step()

            batch_e_mae = ((out["energy"] - energy_true).abs().detach() / n_atoms).sum()
            batch_f_mae = torch.stack(
                [(f - ft).abs().mean().detach() for f, ft in zip(out["forces"], forces_true_list)]
            ).sum()

            total_loss += batch_loss.detach() * len(batch)
            total_e_mae += batch_e_mae
            total_f_mae += batch_f_mae
            n_samples += len(batch)

            # one sync per batch (not per sample) just to refresh the bar
            loss_val, f_mae_val = (total_loss / n_samples).item(), (total_f_mae / n_samples).item()
            pbar.set_postfix(loss=loss_val, f_mae=f_mae_val)

        return {
            "loss": (total_loss / n_samples).item(),
            "energy_mae_per_atom": (total_e_mae / n_samples).item(),
            "force_mae": (total_f_mae / n_samples).item(),
        }

    def fit(self):
        cfg = self.config
        best_val = float("inf")
        best_stats = None

        for epoch in range(cfg.epochs):
            train_stats = self.run_epoch(self.train_loader, train=True, desc=f"epoch {epoch} [train]")
            val_stats = self.run_epoch(self.val_loader, train=False, desc=f"epoch {epoch} [val]")
            self.scheduler.step(val_stats["loss"])

            if epoch % cfg.log_every == 0 or epoch == cfg.epochs - 1:
                print(
                    f"epoch {epoch:3d} "
                    f"train loss {train_stats['loss']:.4f} E_MAE/atom {train_stats['energy_mae_per_atom']:.4f} "
                    f"F_MAE {train_stats['force_mae']:.4f} | "
                    f"val loss {val_stats['loss']:.4f} E_MAE/atom {val_stats['energy_mae_per_atom']:.4f} "
                    f"F_MAE {val_stats['force_mae']:.4f}"
                )

            torch.save({"model_state": self.model.state_dict(), "epoch": epoch}, self.checkpoint_dir / "last.pt")
            if val_stats["loss"] < best_val:
                best_val = val_stats["loss"]
                best_stats = val_stats
                torch.save({"model_state": self.model.state_dict(), "epoch": epoch}, self.checkpoint_dir / "best.pt")

        print(f"done. best val loss {best_val:.4f}. checkpoints in {self.checkpoint_dir}")
        return best_stats
