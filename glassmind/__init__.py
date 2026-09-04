"""GlassMind: portables selektiv-rekurrentes Sprachmodell."""

from glassmind.model.config import ModelConfig
from glassmind.model.lm import GlassMindLM, ModelState

__all__ = ["GlassMindLM", "ModelConfig", "ModelState"]
__version__ = "0.1.0"

