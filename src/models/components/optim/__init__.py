"""PyTorch optimizers for adapting pretrained translation policies."""

from src.models.components.optim.transfer import TransferAdamW

__all__ = ["TransferAdamW"]
