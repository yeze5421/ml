"""Simple machine-learning interatomic potential (MLIP) project."""

from .config import MLIPConfig
from .model import TinyMLIP

__all__ = ["MLIPConfig", "TinyMLIP"]
