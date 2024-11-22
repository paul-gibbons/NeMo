# custom_gpt_model.py

import torch
from nemo.collections.llm.gpt.model.base import GPTModel
from nemo.lightning.megatron_parallel import MaskedTokenLossReduction, masked_token_loss, masked_token_loss_context_parallel

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

class MultiModalLossReduction(MaskedTokenLossReduction):
    """Custom loss reduction class for multi-modal losses"""
    def __init__(self, modality_ranges: Dict[str, Tuple[int, int]], validation_step: bool = False, val_drop_last: bool = True) -> None:
        super().__init__()
        self.modality_ranges = modality_ranges
        self.validation_step = validation_step
        self.val_drop_last = val_drop_last

    def forward(
        self, 
        batch: Dict[str, torch.Tensor], 
        forward_out: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        from megatron.core import parallel_state
        from nemo.collections.nlp.modules.common.megatron.utils import average_losses_across_data_parallel_group

        cp_size = parallel_state.get_context_parallel_world_size()
        if cp_size == 1:
            loss_for_ub = masked_token_loss(forward_out, batch["loss_mask"])
        else:
            loss_for_ub = masked_token_loss_context_parallel(
                forward_out, batch["loss_mask"],batch['num_valid_tokens_in_ub']
            )

        # Calculate per-modality losses
        modality_losses = {}
        tokens = batch["tokens"]
        
        for modality, (start_idx, end_idx) in self.modality_ranges.items():
            # Create mask for tokens in this modality's range
            modality_mask = (tokens >= start_idx) & (tokens < end_idx)
            
            # Combine with original loss mask
            combined_mask = batch["loss_mask"] * modality_mask
            
            if cp_size == 1:
                modality_loss = masked_token_loss(forward_out, combined_mask)
            else:
                # For context parallel, we need valid token count for this modality
                num_valid_modal = combined_mask.sum()
                modality_loss = masked_token_loss_context_parallel(
                    forward_out,
                    combined_mask, 
                    num_valid_modal
                )
            
            modality_losses[f"{modality}_loss"] = modality_loss * cp_size

        # Handle validation step case
        if self.validation_step and not self.val_drop_last:
            num_valid_tokens = batch["loss_mask"].sum()
            if loss_for_ub.isnan():
                assert batch["loss_mask"].count_nonzero() == 0
                loss_sum = torch.zeros_like(num_valid_tokens)
            else:
                loss_sum = num_valid_tokens * loss_for_ub

            loss_sum_and_size = torch.cat([
                loss_sum.clone().detach().view(1),
                torch.tensor([num_valid_tokens], device=torch.cuda.current_device()).clone().detach()
            ])
            
            torch.distributed.all_reduce(
                loss_sum_and_size, 
                group=parallel_state.get_data_parallel_group()
            )
            
            return loss_for_ub * cp_size, {
                "loss_sum_and_ub_size": loss_sum_and_size,
                **modality_losses
            }

        # Regular training/inference case
        reduced_loss = average_losses_across_data_parallel_group([loss_for_ub])
        return loss_for_ub * cp_size, {
            "avg": reduced_loss,
            **modality_losses
        }

    def reduce(self, losses_reduced_per_micro_batch) -> Dict[str, torch.Tensor]:
        """Returns both total loss and per-modality losses"""
        # Calculate total loss (previously done by super().reduce())
        if losses_reduced_per_micro_batch:
            if "avg" in losses_reduced_per_micro_batch[0]:
                loss_tensors_list = [loss_reduced["avg"] for loss_reduced in losses_reduced_per_micro_batch]
                total_loss = torch.concat(loss_tensors_list).mean()
            else:
                # Handle validation case
                loss_sum_tensors_list = [
                    loss_sum["loss_sum_and_ub_size"]
                    for loss_sum in losses_reduced_per_micro_batch
                    if loss_sum["loss_sum_and_ub_size"][1] > 0
                ]
                total_loss = (
                    torch.vstack(loss_sum_tensors_list).sum(dim=0)
                    if len(loss_sum_tensors_list) > 0
                    else torch.tensor([0.0, 0.0], device=torch.cuda.current_device())
                )
        else:
            total_loss = torch.tensor(0.0, device=torch.cuda.current_device())

        # Calculate modality losses
        modality_losses = {}
        if losses_reduced_per_micro_batch:
            for modality in self.modality_ranges.keys():
                loss_key = f"{modality}_loss"
                modality_tensors = [
                    loss_reduced[loss_key] 
                    for loss_reduced in losses_reduced_per_micro_batch
                ]
                if modality_tensors:
                    modality_losses[loss_key] = torch.stack(modality_tensors).mean()
                else:
                    modality_losses[loss_key] = torch.tensor(0.0, device=torch.cuda.current_device())

        return {
            "total_loss": total_loss,
            **modality_losses
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
    def training_loss_reduction(self) -> MultiModalLossReduction:
        if not self._training_loss_reduction:
            self._training_loss_reduction = MultiModalLossReduction(
                modality_ranges=self.modality_ranges
            )
        return self._training_loss_reduction

    @property 
    def validation_loss_reduction(self) -> MultiModalLossReduction:
        if not self._validation_loss_reduction:
            self._validation_loss_reduction = MultiModalLossReduction(
                modality_ranges=self.modality_ranges,
                validation_step=True
            )
        return self._validation_loss_reduction
    



__all__ = [
    "MultiModalGPTModel",
    "MultiModalGPTConfig"
]