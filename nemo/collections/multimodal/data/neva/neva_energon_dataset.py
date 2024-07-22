import dataclasses
from dataclasses import dataclass
from typing import Any, List, Dict, Optional, Tuple, Union
import torch
from einops import rearrange
from megatron.energon import DefaultTaskEncoder, batch_list, batch_stack
from megatron.energon import batch_pad_stack
from megatron.energon import Batch, CaptioningSample, DefaultTaskEncoder, OCRSample, VQASample, SimilarityInterleavedSample, InterleavedSample
from transformers import CLIPImageProcessor, SiglipImageProcessor
import re
import numpy as np
from nemo.collections.common.tokenizers import SentencePieceTokenizer
from nemo.collections.multimodal.data.neva.neva_dataset import (
    process_image,
    preprocess_multimodal,
    preprocess_nvgpt,
    preprocess_nv_dpo,
    preprocess_v1,
    preprocess_llama_2,
    preprocess_llama_3,
    preprocess_plain,
    DEFAULT_IMAGE_TOKEN,
    preprocess_interleaved_prompt
)

@dataclass
class ImageTaskSample:
    __key__: str
    __subflavor__: str
    conversations: List[dict]
    image: Optional[Union[str, List[str], torch.Tensor]] = None
    video: Optional[Union[str, List[str]]] = None

    tokens: Optional[torch.Tensor] = None
    labels: Optional[torch.Tensor] = None
    attention_mask: Optional[torch.Tensor] = None
    loss_mask: Optional[torch.Tensor] = None
    position_ids: Optional[torch.Tensor] = None

@dataclass
class ImageTaskBatch(Batch):
    tokens: torch.Tensor
    labels: torch.Tensor
    attention_mask: torch.Tensor
    loss_mask: torch.Tensor
    position_ids: torch.Tensor
    media: Optional[torch.Tensor] = None



class TaskEncoder(DefaultTaskEncoder[VQASample, InterleavedSample, ImageTaskBatch, dict]):
    def __init__(self, tokenizer, image_processor, multimodal_cfg: dict, data_cfg: dict):
        super().__init__(batch_type=ImageTaskBatch)
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.multimodal_cfg = multimodal_cfg
        self.data_cfg = data_cfg
        self.conv_template = multimodal_cfg["conv_template"]
        self.max_num_images = 6
        self.image_following_text_only = False

    def encode_sample(self, sample: Union[ImageTaskSample, VQASample, InterleavedSample, SimilarityInterleavedSample]) -> dict:
        if isinstance(sample, InterleavedSample):
            return self.encode_interleaved(sample)
        elif isinstance(sample, VQASample):
            return self.encode_pretrain(sample)
        elif isinstance(sample, SimilarityInterleavedSample):
            #return self.encode_similarity_interleaved(sample)
            return self.encode_sft(sample)
        else:
            return self.encode_sft(sample)

    def encode_pretrain(self, sample: VQASample) -> dict:
        conversations = [
            {"from": "human", "value": sample.context},
            {"from": "gpt", "value": sample.answers}
        ]
        processed_sample = {"conversations": conversations}
        
        if self.multimodal_cfg['is_multimodal']:
            if hasattr(sample, 'image') and sample.image is not None:
                processed_sample["image"] = self.process_images(sample.image)
                cur_token_len = self.calculate_token_length(processed_sample["image"])
                processed_sample = preprocess_multimodal(
                    [processed_sample],
                    self.multimodal_cfg,
                    cur_token_len,
                    use_plain=(self.conv_template == "plain")
                )[0]
        
        processed = self.preprocess_conversations([processed_sample])
        tokens = processed["tokens"]
        labels = processed["labels"]
        attention_mask, loss_mask, position_ids = self.get_masks_and_position_ids(tokens, labels)
        
        return ImageTaskSample(
            __key__=sample.__key__,
            __subflavor__=sample.__subflavor__,
            conversations=conversations,
            image=processed_sample.get("image"),
            video=processed_sample.get("video"),
            tokens=tokens.squeeze(0),
            labels=labels.squeeze(0),
            attention_mask=attention_mask.squeeze(0),
            loss_mask=loss_mask.squeeze(0),
            position_ids=position_ids
            
        )

    def encode_sft(self, sample: Union[ImageTaskSample, VQASample, InterleavedSample]) -> dict:
        conversations = sample.texts if hasattr(sample, 'texts') else sample.conversations
        processed_sample = {"conversations": conversations}
        
        if self.multimodal_cfg['is_multimodal']:
            image_present = False
            if hasattr(sample, 'image') and sample.image is not None:
                processed_sample["image"] = self.process_images(sample.image)
                image_present = True
            elif hasattr(sample, 'images') and sample.images:
                processed_sample["image"] = self.process_images(sample.images[0])
                image_present = True
            elif hasattr(sample, 'video') and sample.video:
                # Implement video processing if needed
                pass
            
            if image_present:
                processed_sample = preprocess_multimodal(
                    [processed_sample],
                    self.multimodal_cfg,
                    self.calculate_token_length(processed_sample["image"]),
                    use_plain=(self.conv_template == "plain")
                )[0]
        
        processed = self.preprocess_conversations([processed_sample])
        tokens = processed["tokens"]
        labels = processed["labels"]
        attention_mask, loss_mask, position_ids = self.get_masks_and_position_ids(tokens, labels)
        
<<<<<<< HEAD
        if not image_present:
            processed_sample["image"] = torch.zeros(1, 3, self.multimodal_cfg["crop_size"][0], self.multimodal_cfg["crop_size"][1])

=======
        #add 1 dummy pad image when no image is present
        if not image_present:
            processed_sample["image"] = torch.zeros(1, 3, self.multimodal_cfg["crop_size"][0], self.multimodal_cfg["crop_size"][1])
        
>>>>>>> 45fcc13e270279dee445c78f571e8d9fadc47764
        return ImageTaskSample(
            __key__=sample.__key__,
            __subflavor__=sample.__subflavor__,
            conversations=conversations,
            #rewrite image so it creates tensor of zeros if not present
            image=processed_sample.get("image", torch.tensor([])),
            tokens=tokens.squeeze(0),
            labels=labels.squeeze(0),
            attention_mask=attention_mask.squeeze(0),
            loss_mask=loss_mask.squeeze(0),
            position_ids=position_ids
        )

    def encode_similarity_interleaved(self, sample: SimilarityInterleavedSample) -> dict:
        
        # 4 fields: sample.images, sample.texts, sample.similarity_matrix, sample.matched_text_index
        images, sentence_ixs = [], []
        for sample_image, sim_vec in zip(sample.images, sample.matched_text_index):
            images.append(sample_image)
            sentence_ixs.append(sim_vec)
            
        # constrain max num images
        max_num_images = self.max_num_images
        if len(images) > max_num_images:
            images = images[:max_num_images]
            sentence_ixs = sentence_ixs[:max_num_images]
        
        images = [images[i] for i in np.argsort(sentence_ixs)]
        
        for ix in sentence_ixs:
            sample.texts[ix] = f"{DEFAULT_IMAGE_TOKEN} {sample.texts[ix]}"
        
        if self.image_following_text_only:
            # use pad token to divide sentence pieces
            text = self.tokenizer.pad_id.join(sample.texts)
        else:
            text = " ".join(sample.texts)
        
        text = text.replace("<image> ", "<image>").replace(" <image>", "<image>")
        text = f"{text}{self.tokenizer.eos_id}"
        
        if len(images) > 0:
            processed_images = self.process_images(images)
        else:
            processed_images = None
        
        # check the case where the last token is the image token, do we ALSO need this for strictly interleaved?
        if text.endswith(DEFAULT_IMAGE_TOKEN):
            text = text[:-len(DEFAULT_IMAGE_TOKEN)]
        
        n_im_patch = text.count(DEFAULT_IMAGE_TOKEN)
        processed_images = processed_images[:n_im_patch]
        assert len(processed_images) == n_im_patch
        
        processed_sample = {
            "conversations": text,
            "image": processed_images
        }
        
        if self.multimodal_cfg['is_multimodal']:
            if images:
                cur_token_len = self.calculate_token_length(processed_sample["image"])
                processed_sample = preprocess_multimodal(
                    [processed_sample],
                    self.multimodal_cfg,
                    cur_token_len,
                    use_plain=(self.conv_template == "plain")
                )[0]
                
        processed = self.preprocess_conversations([processed_sample])
        
        tokens = processed["tokens"]
        labels = processed["labels"]
        attention_mask, loss_mask, position_ids = self.get_masks_and_position_ids(tokens, labels)
        
        #pad images
        if images:
            processed_sample["image"] = self.pad_images(processed_sample["image"], self.max_num_images)
        
        return ImageTaskSample(
            __key__=sample.__key__,
            __subflavor__=sample.__subflavor__,
            conversations=processed_sample["conversations"],
            image=processed_sample["image"],
            tokens=tokens.squeeze(0),
            labels=labels.squeeze(0),
            attention_mask=attention_mask.squeeze(0),
            loss_mask=loss_mask.squeeze(0),
            position_ids=position_ids
        )
    
    
    def encode_similarity_interleaved(self, sample: SimilarityInterleavedSample) -> dict:
        
        # 4 fields: sample.images, sample.texts, sample.similarity_matrix, sample.matched_text_index
        images, sentence_ixs = [], []
        for sample_image, sim_vec in zip(sample.images, sample.matched_text_index):
            images.append(sample_image)
            sentence_ixs.append(sim_vec)
            
        # constrain max num images
        max_num_images = self.max_num_images
        if len(images) > max_num_images:
            images = images[:max_num_images]
            sentence_ixs = sentence_ixs[:max_num_images]
        
        images = [images[i] for i in np.argsort(sentence_ixs)]
        
        for ix in sentence_ixs:
            sample.texts[ix] = f"{DEFAULT_IMAGE_TOKEN} {sample.texts[ix]}"
        
        if self.image_following_text_only:
            # use pad token to divide sentence pieces
            text = self.tokenizer.pad_id.join(sample.texts)
        else:
            text = " ".join(sample.texts)
        
        text = text.replace("<image> ", "<image>").replace(" <image>", "<image>")
        text = f"{text}{self.tokenizer.eos_id}"
        
        if len(images) > 0:
            processed_images = self.process_images(images)
        else:
            processed_images = None
        
        # check the case where the last token is the image token, do we ALSO need this for strictly interleaved?
        if text.endswith(DEFAULT_IMAGE_TOKEN):
            text = text[:-len(DEFAULT_IMAGE_TOKEN)]
        
        n_im_patch = text.count(DEFAULT_IMAGE_TOKEN)
        processed_images = processed_images[:n_im_patch]
        assert len(processed_images) == n_im_patch
        
        processed_sample = {
            "conversations": text,
            "image": processed_images
        }
        
        if self.multimodal_cfg['is_multimodal']:
            if images:
                cur_token_len = self.calculate_token_length(processed_sample["image"])
                processed_sample = preprocess_multimodal(
                    [processed_sample],
                    self.multimodal_cfg,
                    cur_token_len,
                    use_plain=(self.conv_template == "plain")
                )[0]
                
        processed = self.preprocess_conversations([processed_sample])
        
        tokens = processed["tokens"]
        labels = processed["labels"]
        attention_mask, loss_mask, position_ids = self.get_masks_and_position_ids(tokens, labels)
        
        #pad images
        if images:
            processed_sample["image"] = self.pad_images(processed_sample["image"], self.max_num_images)
        
        return ImageTaskSample(
            __key__=sample.__key__,
            __subflavor__=sample.__subflavor__,
            conversations=processed_sample["conversations"],
            image=processed_sample["image"],
            tokens=tokens,
            labels=labels,
            attention_mask=attention_mask,
            loss_mask=loss_mask,
            position_ids=position_ids
        )
    
    def encode_interleaved(self, sample: InterleavedSample) -> dict:
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
            
        # constrain max num images
        max_num_images = self.max_num_images
        if len(images) > max_num_images:
            images = images[:max_num_images]

        if len(images) > 0:
            processed_images = self.process_images(images)
        else:
            processed_images = None
        
        combined_text = ' '.join(interleaved_text)
        
        processed_sample = {
            "conversations": combined_text,
            "image": processed_images
        }
        
        if self.multimodal_cfg['is_multimodal']:
            if images:
                cur_token_len = self.calculate_token_length(processed_sample["image"])
                processed_sample = preprocess_multimodal(
                    [processed_sample],
                    self.multimodal_cfg,
                    cur_token_len,
                    use_plain=(self.conv_template == "plain")
                )[0]
                
        
        processed = self.preprocess_conversations([processed_sample])
        
        tokens = processed["tokens"]
        labels = processed["labels"]
        attention_mask, loss_mask, position_ids = self.get_masks_and_position_ids(tokens, labels)
        
        #pad images
        if images:
            processed_sample["image"] = self.pad_images(processed_sample["image"], self.max_num_images)
        
        return ImageTaskSample(
            __key__=sample.__key__,
            __subflavor__=sample.__subflavor__,
            conversations=processed_sample["conversations"],
            image=processed_sample["image"],
            tokens=tokens.squeeze(0),
            labels=labels.squeeze(0),
            attention_mask=attention_mask.squeeze(0),
            loss_mask=loss_mask.squeeze(0),
            position_ids=position_ids
        )
        
    def process_images(self, images):
        if not isinstance(images, list):
            images = [images]
        processed_images = []
        for image in images:
            image = process_image(self.image_processor, image, self.multimodal_cfg['image_aspect_ratio'])
            processed_images.append(image)
        return torch.stack(processed_images) #make it always 4D, otherwise has problem when len(images) > 1

    def pad_images(self, images, max_num_images):
        if len(images) < max_num_images:
            pad_size = max_num_images - len(images)
            padded_images = torch.cat([images, torch.zeros(pad_size, *images.size()[1:])], dim=0)
            return padded_images
        return images

    def batch(self, samples: List[ImageTaskSample]) -> ImageTaskBatch:
        
        batch = ImageTaskBatch(
            tokens=batch_pad_stack([s.tokens for s in samples]),
            labels=batch_pad_stack([s.labels for s in samples]),
            attention_mask=batch_pad_stack([s.attention_mask for s in samples]),
            loss_mask=batch_pad_stack([s.loss_mask for s in samples]),
            position_ids=batch_pad_stack([s.position_ids for s in samples]),
            media=torch.stack([s.image for s in samples if s.image is not None]) if self.multimodal_cfg['is_multimodal'] else None
        )
        
        #TODO: cleanup, this is following logic in neva_dataset when we rearrange media tensor
        if batch.media.shape[1] == 1:
            batch.media = rearrange(batch.media, "b T c h w -> b T 1 c h w")
        else:
            batch.media = rearrange(batch.media, "b T c h w -> b 1 T c h w")
        
        return batch


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
        elif self.conv_template == "interleaved":
            return preprocess_interleaved_prompt(sources, self.tokenizer, self.multimodal_cfg)
        else:
            raise ValueError(f"Conversation template `{self.conv_template}` is not supported in Neva now.")



    def encode_batch(self, batch: ImageTaskBatch) -> dict:
        raw = dataclasses.asdict(batch)
        return raw

    def calculate_token_length(self, media_tensor):
        if len(media_tensor.shape) == 4:
            height = media_tensor.shape[2]
            width = media_tensor.shape[3]
        else:
            raise ValueError("Media tensor must be 4-dimensional")
        patch_dim = self.multimodal_cfg['patch_dim']
        height_num_patches = height // patch_dim
        width_num_patches = width // patch_dim
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
# neva_encoder = TaskEncoder(
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
