"""Training callbacks for pretraining and finetuning."""
import torch
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping based on validation metric (higher = better)."""

    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("-inf")
        self.counter = 0
        self.should_stop = False

    def __call__(self, val_metric: float) -> bool:
        if val_metric > self.best + self.min_delta:
            self.best = val_metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                logger.info(f"Early stopping triggered after {self.patience} epochs without improvement")
        return self.should_stop


class CheckpointCallback:
    """Save best and latest model checkpoints."""

    def __init__(self, save_dir: str, model_name: str = "stella"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.best_metric = float("-inf")

    def __call__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: dict,
        val_metric: float,
    ) -> bool:
        """Returns True if this is a new best checkpoint."""
        # Always save latest
        latest_path = self.save_dir / f"{self.model_name}_latest.pt"
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
        }, latest_path)

        # Save best
        if val_metric > self.best_metric:
            self.best_metric = val_metric
            best_path = self.save_dir / f"{self.model_name}_best.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "metrics": metrics,
                "val_metric": val_metric,
            }, best_path)
            logger.info(f"New best checkpoint (epoch {epoch}, metric={val_metric:.4f})")
            return True
        return False

    def load_best(self, model: torch.nn.Module) -> dict:
        best_path = self.save_dir / f"{self.model_name}_best.pt"
        ckpt = torch.load(best_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info(f"Loaded best checkpoint from epoch {ckpt['epoch']}")
        return ckpt
