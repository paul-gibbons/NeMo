# custom_gpt_model.py

from nemo.collections.llm.gpt.model.base import GPTModel
from nemo.lightning.megatron_parallel import MaskedTokenLossReduction

from dataclasses import dataclass
from typing import Dict, Tuple

from nemo.collections.llm.gpt.model.base import GPTConfig

@dataclass
class MultiModalGPTConfig(GPTConfig):
    """Configuration class for MultiModalGPT model that adds modality ranges"""
    
    modality_ranges: Dict[str, Tuple[int, int]] = None
    
    def __post_init__(self):
        super().__post_init__()
        if self.modality_ranges is None:
            self.modality_ranges = {
                "text": (0, 25000),
                "image": (25000, 50257)
            }

class MultiModalGPTModel(GPTModel):
    """Custom GPT model that handles early-fusion multi-modal loss tracking"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._training_loss_reduction = None
        self._validation_loss_reduction = None
        
        # Define modality ranges
        self.modality_ranges = self.config.modality_ranges

    @property
    def training_loss_reduction(self) -> MaskedTokenLossReduction:
        if not self._training_loss_reduction:
            self._training_loss_reduction = MaskedTokenLossReduction(
                modality_ranges=self.modality_ranges
            )
        return self._training_loss_reduction

    @property 
    def validation_loss_reduction(self) -> MaskedTokenLossReduction:
        if not self._validation_loss_reduction:
            self._validation_loss_reduction = MaskedTokenLossReduction(
                modality_ranges=self.modality_ranges,
                validation_step=True
            )
        return self._validation_loss_reduction
    

__all__ = [
    "MultiModalGPTModel",
    "MultiModalGPTConfig"
]