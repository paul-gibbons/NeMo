import dataclasses
from dataclasses import dataclass
from typing import Any, List, Dict, Optional, Tuple, Union
import torch
from megatron.energon import DefaultTaskEncoder, batch_list, batch_stack
from megatron.energon import Batch, CaptioningSample, DefaultTaskEncoder, OCRSample, VQASample, SimilarityInterleavedSample, InterleavedSample
from transformers import CLIPImageProcessor, SiglipImageProcessor
import re
from nemo.collections.common.tokenizers import SentencePieceTokenizer
from neva_dataset import (
    process_image,
    preprocess_multimodal,
    preprocess_nvgpt,
    preprocess_nv_dpo,
    preprocess_v1,
    preprocess_llama_2,
    preprocess_llama_3,
    preprocess_plain,
    DEFAULT_IMAGE_TOKEN,
)

@dataclass
class NevaSample:
    __key__: str
    __subflavor__: str
    conversations: List[dict]
    image: Optional[Union[str, List[str]]] = None
    video: Optional[Union[str, List[str]]] = None

@dataclass
class NevaBatch(Batch):
    __keys__: List[str]
    tokens: torch.Tensor
    labels: torch.Tensor
    attention_mask: torch.Tensor
    loss_mask: torch.Tensor
    position_ids: torch.Tensor
    media: Optional[torch.Tensor] = None
    cu_seqlens: Optional[torch.Tensor] = None



class NevaTaskEncoder(DefaultTaskEncoder[Union[NevaSample, VQASample, InterleavedSample], Union[NevaSample, VQASample], NevaBatch, NevaBatch]):
    def __init__(self, tokenizer, image_processor, multimodal_cfg: dict, data_cfg: dict):
        super().__init__(batch_type=NevaBatch)
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.multimodal_cfg = multimodal_cfg
        self.data_cfg = data_cfg
        self.conv_template = multimodal_cfg["conv_template"]

    def encode_sample(self, sample: Union[NevaSample, VQASample, InterleavedSample]) -> NevaSample:
        if isinstance(sample, InterleavedSample):
            return self.encode_interleaved(sample)
        
        if self.multimodal_cfg['is_multimodal']:
            if hasattr(sample, 'image') and sample.image is not None:
                sample.image = self.process_images(sample.image)
            elif hasattr(sample, 'video') and sample.video:
                # Implement video processing if needed
                pass
        return sample

    def encode_interleaved(self, sample: InterleavedSample) -> NevaSample:
        interleaved_text = []
        images = []
        for item in sample.sequence:
            if isinstance(item, str):
                interleaved_text.append(item)
            elif isinstance(item, torch.Tensor) or isinstance(item, str):
                interleaved_text.append(DEFAULT_IMAGE_TOKEN)
        
                images.append(item)
            else:
                raise ValueError(f"Unsupported type in interleaved sequence: {type(item)}")
        processed_images = self.process_images(images)
        
        combined_text = ' '.join(interleaved_text)


        return NevaSample(
            __key__=sample.__key__,
            __subflavor__=sample.__subflavors__.get("interleaved", ""),
            conversations=[
            {"from": "human", "value": combined_text},
            {"from": "gpt", "value": ""}  # Empty response placeholder
        ],
            image=torch.stack(processed_images) if images else None
        )

    def process_images(self, images):
        if not isinstance(images, list):
            images = [images]
        processed_images = []
        for image in images:
            image = process_image(self.image_processor, image, self.multimodal_cfg['image_aspect_ratio'])
            processed_images.append(image)
        return processed_images

    def batch(self, samples: List[Union[NevaSample, VQASample, InterleavedSample]]) -> NevaBatch:
        sources = []
        for s in samples:
            if isinstance(s, VQASample):
                conversations = [
                    {"from": "human", "value": s.context},
                    {"from": "gpt", "value": s.answers}
                ]
            else:
                conversations = s.conversations
            sources.append({"conversations": conversations})

        if self.multimodal_cfg['is_multimodal']:
            cur_token_len = self.calculate_token_length(samples[0].image[0])
            sources = preprocess_multimodal(
                sources,
                self.multimodal_cfg,
                cur_token_len,
                use_plain=(self.conv_template == "plain")
            )

        data_dict = self.preprocess_conversations(sources)

        tokens = data_dict["tokens"]
        labels = data_dict["labels"]
        attention_mask, loss_mask, position_ids = self.get_masks_and_position_ids(tokens, labels)

        media = None
        if self.multimodal_cfg['is_multimodal']:
            media_list = []
            for s in samples:
                if hasattr(s, 'image') and s.image is not None:
                    if isinstance(s.image, list):
                        # Multiple images case
                        media_list.extend(s.image)
                    else:
                        # Single image case
                        media_list.append(s.image)
            if media_list:
                media = torch.stack(media_list)

        return NevaBatch(
            __keys__=[s.__key__ for s in samples],
            tokens=tokens,
            labels=labels,
            attention_mask=attention_mask,
            loss_mask=loss_mask,
            position_ids=position_ids,
            media=media
        )

    def preprocess_conversations(self, sources):
        if self.conv_template == "nvgpt":
            return preprocess_nvgpt(sources, self.tokenizer, self.multimodal_cfg)
        elif self.conv_template == "nv_dpo":
            return preprocess_nv_dpo(sources, self.tokenizer, self.multimodal_cfg)
        elif self.conv_template == "v1":
            return preprocess_v1(sources, self.tokenizer, self.multimodal_cfg)
        elif self.conv_template == "llama_2":
            return preprocess_llama_2(sources, self.tokenizer, self.multimodal_cfg)
        elif self.conv_template == "llama_3":
            return preprocess_llama_3(sources, self.tokenizer, self.multimodal_cfg)
        elif self.conv_template == "mistral":
            return preprocess_llama_2(sources, self.tokenizer, self.multimodal_cfg, is_mistral=True)
        elif self.conv_template == "plain":
            return preprocess_plain(sources, self.tokenizer, self.multimodal_cfg)
        else:
            raise ValueError(f"Conversation template `{self.conv_template}` is not supported in Neva now.")



    def encode_batch(self, batch: NevaBatch) -> dict:
        raw = dataclasses.asdict(batch)
        return raw

    def calculate_token_length(self, media_tensor):
        patch_dim = self.multimodal_cfg['patch_dim']
        height_num_patches = media_tensor.shape[1] // patch_dim
        width_num_patches = media_tensor.shape[2] // patch_dim
        if self.multimodal_cfg['mm_mlp_adapter_type'] == 'mlp_downsample':
            height_num_patches = (height_num_patches + 1) // 2 * 2
            width_num_patches = (width_num_patches + 1) // 2 * 2
        return height_num_patches * width_num_patches

    def get_masks_and_position_ids(self, tokens, labels):
        from nemo.collections.nlp.modules.common.megatron.utils import get_ltor_masks_and_position_ids

        attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
            data=tokens,
            eod_token=self.tokenizer.eos_id,
            eod_mask_loss=self.data_cfg.get("eod_mask_loss", False),
            reset_attention_mask=False,
            reset_position_ids=False,
        )

        loss_mask[labels == -1] = 0.0
        tokens[tokens == -1] = 0
        labels[labels == -1] = 0
        return attention_mask, loss_mask, position_ids

# Usage example:
# neva_encoder = NevaTaskEncoder(
#     tokenizer=your_tokenizer,
#     image_processor=your_image_processor,
#     multimodal_cfg=your_multimodal_cfg,
#     data_cfg=your_data_cfg
# )

# train_loader = get_loader(get_train_dataset(
#     '/path/to/your/dataset',
#     batch_size=2,
#     shuffle_buffer_size=100,
#     task_encoder=neva_encoder,
#     image_decode="torchrgb",
# ))
