import time
from pathlib import Path
import torch

## ====== Helper Classes and Functions ====== ##
class ProgressTracker:
    def __init__(self, total_batches):
        self.total_batches = total_batches
        self.start_time = time.time()
        self.last_time = self.start_time

    def update(self, batch_idx):
        now = time.time()
        batch_time = now - self.last_time
        elapsed = now - self.start_time
        avg_batch_time = elapsed / max(batch_idx + 1, 1)
        remaining = self.total_batches - (batch_idx + 1)
        eta = remaining * avg_batch_time
        self.last_time = now

        print(
            f"\r{batch_idx + 1}/{self.total_batches} | batch {batch_time:.3f}s | avg {avg_batch_time:.3f}s | eta {eta:.1f}s",
            end="",
            flush=True,
        )

    def finish(self):
        print()



def _count_correct(preds: torch.Tensor, target: torch.Tensor) -> int:
    pred_labels = preds.argmax(dim=1)
    return (pred_labels == target).sum().item()


def _compute_accuracy(num_correct: int, num_samples: int) -> float:
    if num_samples == 0:
        return 0.0
    return num_correct / num_samples
# ================================================


# Checkpointing
def save_checkpoint(path, model, optimizer, scheduler, epoch, metrics: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": None if scheduler is None else scheduler.state_dict(),
            "metrics": metrics,
        },
        path,
    )


def load_checkpoint(path, device="cpu"):
    path = Path(path)
    return torch.load(path, map_location=device)


def apply_checkpoint_load(model, optimizer, scheduler, checkpoint_cfg: dict, device="cpu"):
    if checkpoint_cfg is None:
        return None

    path = checkpoint_cfg.get("path")
    if not path:
        return None

    ckpt = load_checkpoint(path, device=device)

    if checkpoint_cfg.get("load_model", True):
        model.load_state_dict(ckpt["model_state_dict"])

    if checkpoint_cfg.get("load_optimizer", False):
        if optimizer is None:
            raise ValueError("Checkpoint requested optimizer load, but optimizer is None")
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    if checkpoint_cfg.get("load_scheduler", False):
        if scheduler is None:
            raise ValueError("Checkpoint requested scheduler load, but scheduler is None")
        scheduler_state = ckpt.get("scheduler_state_dict")
        if scheduler_state is not None:
            scheduler.load_state_dict(scheduler_state)

    return ckpt

class CheckpointManager:
    def __init__(self, checkpoint_config: dict, run_dir):
        self.enabled = checkpoint_config.get("enabled", False)
        self.mode = checkpoint_config.get("mode", "save_last")
        self.dir = Path(run_dir) / checkpoint_config.get("dir", "checkpoints")
        self.monitor = checkpoint_config.get("monitor", "val_loss")
        self.every_n_epochs = checkpoint_config.get("every_n_epochs", 1)
        self.best_value = None

    def step(self, epoch, model, optimizer, scheduler, metrics: dict):
        if not self.enabled:
            return

        if self.mode == "last":
            save_checkpoint(self.dir / "last.pt", model, optimizer, scheduler, epoch, metrics)

        elif self.mode == "best":
            current = metrics[self.monitor]
            if self.best_value is None or current < self.best_value:
                self.best_value = current
                save_checkpoint(self.dir / "best.pt", model, optimizer, scheduler, epoch, metrics)

        elif self.mode == "every":
            if epoch % self.every_n_epochs == 0:
                save_checkpoint(self.dir / f"epoch_{epoch}.pt", model, optimizer, scheduler, epoch, metrics)

        else:
            raise ValueError(f"Unknown checkpoint mode: {self.mode}")