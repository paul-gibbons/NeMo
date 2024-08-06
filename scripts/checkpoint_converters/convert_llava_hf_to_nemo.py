# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
   python3 /opt/NeMo/scripts/checkpoint_converters/convert_llava_hf_to_nemo.py \
   --input_name_or_path llava-hf/llava-1.5-7b-hf \
   --output_path /path/to/llava-7b.nemo \
   --tokenizer_path /path/to/tokenizer.model
"""

from collections import defaultdict
import os
from argparse import ArgumentParser

import torch
from omegaconf import OmegaConf
from transformers import LlamaTokenizer, LlavaForConditionalGeneration

from nemo.collections.multimodal.models.multimodal_llm.neva.neva_model import MegatronNevaModel
from nemo.collections.nlp.modules.common.megatron.utils import get_ltor_masks_and_position_ids
from nemo.collections.nlp.parts.megatron_trainer_builder import MegatronTrainerBuilder
from nemo.collections.nlp.parts.utils_funcs import torch_dtype_from_precision
from nemo.utils import logging


from llava.model.language_model.builder import build_llm_and_tokenizer
from llava.model.builder import load_pretrained_model

import re
def get_model_info(model, model_name):
    info = []
    layer_groups = defaultdict(list)

    for name, param in model.named_parameters():
        # Extract layer number and component
        match = re.match(r'(.+\.layers\.)(\d+)(\..*)', name)
        if match:
            prefix, layer_num, suffix = match.groups()
            layer_num = int(layer_num)
            key = (prefix, suffix)
            layer_groups[key].append(layer_num)
        else:
            info.append(f"{model_name}: {name} {param.shape}")

    # Process grouped layers
    for (prefix, suffix), layers in layer_groups.items():
        if len(layers) > 2:
            info.append(f"{model_name}: {prefix}[{min(layers)}-{max(layers)}]{suffix} {param.shape}")
        else:
            for layer in layers:
                info.append(f"{model_name}: {prefix}{layer}{suffix} {param.shape}")

    return sorted(info)

def save_model_info(hf_info, nemo_info, output_file):
    with open(output_file, 'w') as f:
        f.write("HuggingFace Model:\n")
        f.write("\n".join(hf_info))
        f.write("\n\n============================================================\n\n")
        f.write("Nemo Model:\n")
        f.write("\n".join(nemo_info))



"""HuggingFace Model:
HF: llm.lm_head.weight torch.Size([64000, 7168])
HF: llm.model.embed_tokens.weight torch.Size([64000, 7168])
HF: llm.model.layers.[0-59].input_layernorm.weight torch.Size([7168])
HF: llm.model.layers.[0-59].mlp.down_proj.weight torch.Size([7168])
HF: llm.model.layers.[0-59].mlp.gate_proj.weight torch.Size([7168])
HF: llm.model.layers.[0-59].mlp.up_proj.weight torch.Size([7168])
HF: llm.model.layers.[0-59].post_attention_layernorm.weight torch.Size([7168])
HF: llm.model.layers.[0-59].self_attn.k_proj.weight torch.Size([7168])
HF: llm.model.layers.[0-59].self_attn.o_proj.weight torch.Size([7168])
HF: llm.model.layers.[0-59].self_attn.q_proj.weight torch.Size([7168])
HF: llm.model.layers.[0-59].self_attn.v_proj.weight torch.Size([7168])
HF: llm.model.norm.weight torch.Size([7168])
HF: mm_projector.layers.[1-4].bias torch.Size([7168])
HF: mm_projector.layers.[1-4].weight torch.Size([7168])
HF: vision_tower.vision_tower.embeddings.class_embedding torch.Size([1, 1, 3200])
HF: vision_tower.vision_tower.embeddings.patch_embedding.bias torch.Size([3200])
HF: vision_tower.vision_tower.embeddings.patch_embedding.weight torch.Size([3200, 3, 14, 14])
HF: vision_tower.vision_tower.embeddings.position_embedding torch.Size([1, 1025, 3200])
HF: vision_tower.vision_tower.encoder.layers.[0-44].attn.k_norm.weight torch.Size([7168])
HF: vision_tower.vision_tower.encoder.layers.[0-44].attn.proj.bias torch.Size([7168])
HF: vision_tower.vision_tower.encoder.layers.[0-44].attn.proj.weight torch.Size([7168])
HF: vision_tower.vision_tower.encoder.layers.[0-44].attn.q_norm.weight torch.Size([7168])
HF: vision_tower.vision_tower.encoder.layers.[0-44].attn.qkv.weight torch.Size([7168])
HF: vision_tower.vision_tower.encoder.layers.[0-44].ls1 torch.Size([7168])
HF: vision_tower.vision_tower.encoder.layers.[0-44].ls2 torch.Size([7168])
HF: vision_tower.vision_tower.encoder.layers.[0-44].mlp.fc1.bias torch.Size([7168])
HF: vision_tower.vision_tower.encoder.layers.[0-44].mlp.fc1.weight torch.Size([7168])
HF: vision_tower.vision_tower.encoder.layers.[0-44].mlp.fc2.bias torch.Size([7168])
HF: vision_tower.vision_tower.encoder.layers.[0-44].mlp.fc2.weight torch.Size([7168])
HF: vision_tower.vision_tower.encoder.layers.[0-44].norm1.weight torch.Size([7168])
HF: vision_tower.vision_tower.encoder.layers.[0-44].norm2.weight torch.Size([7168])"""
def create_rename_keys(num_hidden_layers):
    rename_keys = []
    for i in range(num_hidden_layers):
        rename_keys.extend([
            (f"llm.model.layers.{i}.input_layernorm.weight", f"model.module.decoder.layers.{i}.self_attention.linear_qkv.layer_norm_weight"),
            (f"llm.model.layers.{i}.mlp.down_proj.weight", f"model.module.decoder.layers.{i}.mlp.linear_fc2.weight"),
            (f"llm.model.layers.{i}.mlp.gate_proj.weight", f"model.module.decoder.layers.{i}.mlp.linear_fc1_gate.weight"),
            (f"llm.model.layers.{i}.mlp.up_proj.weight", f"model.module.decoder.layers.{i}.mlp.linear_fc1_proj.weight"),
            (f"llm.model.layers.{i}.post_attention_layernorm.weight", f"model.module.decoder.layers.{i}.mlp.linear_fc1.layer_norm_weight"),
            (f"llm.model.layers.{i}.self_attn.o_proj.weight", f"model.module.decoder.layers.{i}.self_attention.linear_proj.weight"),
            (f"llm.model.layers.{i}.self_attn.q_proj.weight", f"model.module.decoder.layers.{i}.self_attention.linear_q.weight"),
            (f"llm.model.layers.{i}.self_attn.k_proj.weight", f"model.module.decoder.layers.{i}.self_attention.linear_k.weight"),
            (f"llm.model.layers.{i}.self_attn.v_proj.weight", f"model.module.decoder.layers.{i}.self_attention.linear_v.weight"),
        ])

    # Non-layer dependent keys
    rename_keys.extend([
        ("llm.lm_head.weight", "model.module.output_layer.weight"),
        ("llm.model.embed_tokens.weight", "model.module.embedding.word_embeddings.weight"),
        ("llm.model.norm.weight", "model.module.decoder.final_layernorm.weight"),
        ("mm_projector.layers.0.bias","model.module.embedding.word_embeddings.adapter_layer.mm_projector_adapter.mm_projector.0.bias"),
        ("mm_projector.layers.0.weight","model.module.embedding.word_embeddings.adapter_layer.mm_projector_adapter.mm_projector.0.weight"),
        ("mm_projector.layers.1.bias","model.module.embedding.word_embeddings.adapter_layer.mm_projector_adapter.mm_projector.1.bias"),
        ("mm_projector.layers.1.weight","model.module.embedding.word_embeddings.adapter_layer.mm_projector_adapter.mm_projector.1.weight"),
        ("mm_projector.layers.2.bias","model.module.embedding.word_embeddings.adapter_layer.mm_projector_adapter.mm_projector.2.bias"),
        ("mm_projector.layers.2.weight","model.module.embedding.word_embeddings.adapter_layer.mm_projector_adapter.mm_projector.2.weight"),
        ("mm_projector.layers.3.bias","model.module.embedding.word_embeddings.adapter_layer.mm_projector_adapter.mm_projector.3.bias"),
        ("mm_projector.layers.3.weight","model.module.embedding.word_embeddings.adapter_layer.mm_projector_adapter.mm_projector.3.weight"),
        ("mm_projector.layers.4.bias","model.module.embedding.word_embeddings.adapter_layer.mm_projector_adapter.mm_projector.4.bias"),
        ("mm_projector.layers.4.weight","model.module.embedding.word_embeddings.adapter_layer.mm_projector_adapter.mm_projector.4.weight"),
    ])

    return rename_keys


def rename_model_keys(model_state_dict, rename_keys):
    """
    Rename keys in the model's state dictionary based on the provided mappings.

    Parameters:
    model_state_dict (dict): The state dictionary of the model.
    rename_keys (list): A list of tuples with the mapping (old_key, new_key).

    Returns:
    dict: A new state dictionary with updated key names.
    """

    # Create a new state dictionary with updated key names
    new_state_dict = {}

    # Track keys from the original state dict to ensure all are processed
    remaining_keys = set(model_state_dict.keys())

    # Iterate over the rename mappings
    for old_key, new_key in rename_keys:
        if old_key in model_state_dict:
            # Rename the key and remove it from the tracking set
            new_state_dict[new_key] = model_state_dict[old_key]
            remaining_keys.remove(old_key)

    # Check if any keys were not converted from old to new
    for old_key in remaining_keys:
        print(f"Warning: Key '{old_key}' was not converted.")

    return new_state_dict


def adjust_tensor_shapes(model, nemo_state_dict):
    """
    Adapt tensor shapes in the state dictionary to ensure compatibility with a different model structure.

    Parameters:
    nemo_state_dict (dict): The state dictionary of the model.

    Returns:
    dict: The updated state dictionary with modified tensor shapes for compatibility.
    """
    model_config = model.cfg
    num_query_groups = model_config["num_query_groups"]
    head_num = model_config["num_attention_heads"]
    hidden_size = model_config["hidden_size"]
    head_size = model_config["kv_channels"]
    heads_per_group = head_num // num_query_groups

    # Note: For 'key' and 'value' weight and biases, NeMo uses a consolidated tensor 'query_key_value'.
    for key_ in list(nemo_state_dict.keys()):
        if 'vision_towel' in key_:
            del nemo_state_dict[key_]

        #if 'word_embeddings.weight' in key_ or 'output_layer.weight' in key_:
        #    # padding
        #    loaded_weight = nemo_state_dict[key_]
        #    new_weight = model.state_dict()[key_]
        #    new_weight[: loaded_weight.shape[0], : loaded_weight.shape[1]] = loaded_weight
        #    nemo_state_dict[key_] = new_weight

        if 'mlp.linear_fc1_gate.weight' in key_:
            key_gate = key_
            key_proj = key_.replace('mlp.linear_fc1_gate.weight', 'mlp.linear_fc1_proj.weight')
            new_key = key_.replace('mlp.linear_fc1_gate.weight', 'mlp.linear_fc1.weight')
            gate_weight = nemo_state_dict[key_gate]
            proj_weight = nemo_state_dict[key_proj]
            nemo_state_dict[new_key] = torch.cat((gate_weight, proj_weight))
            del nemo_state_dict[key_gate], nemo_state_dict[key_proj]

        if 'self_attention.linear_q.weight' in key_:
            key_q = key_
            key_k = key_.replace('linear_q', 'linear_k')
            key_v = key_.replace('linear_q', 'linear_v')
            key_qkv = key_.replace('linear_q', 'linear_qkv')

            # [(head_num + 2 * num_query_groups) * head_size, hidden_size]
            # -> [head_num, head_size, hidden_size], 2 * [num_query_groups, head_size, hidden_size]
            q_weight, k_weight, v_weight = nemo_state_dict[key_q], nemo_state_dict[key_k], nemo_state_dict[key_v]
            q_weight = q_weight.reshape(head_num, head_size, hidden_size)
            k_weight = k_weight.reshape(num_query_groups, head_size, hidden_size)
            v_weight = v_weight.reshape(num_query_groups, head_size, hidden_size)

            qkv_weight = torch.empty((0, head_size, hidden_size), device=q_weight.device)
            for i in range(num_query_groups):
                qkv_weight = torch.cat((qkv_weight, q_weight[i * heads_per_group : (i + 1) * heads_per_group, :, :]))
                qkv_weight = torch.cat((qkv_weight, k_weight[i : i + 1, :, :]))
                qkv_weight = torch.cat((qkv_weight, v_weight[i : i + 1, :, :]))
            qkv_weight = qkv_weight.reshape([head_size * (head_num + 2 * num_query_groups), hidden_size])
            nemo_state_dict[key_qkv] = qkv_weight
            del nemo_state_dict[key_q], nemo_state_dict[key_k], nemo_state_dict[key_v]

    return nemo_state_dict


def adjust_nemo_config(model_config, ref_config):
    model_config.mm_cfg.mm_mlp_adapter_type = "mlp_downsample"
    model_config.mm_cfg.vision_encoder.from_pretrained = "/raid/pgibbons/models/VILA1.5-40b/vision_tower"

    model_config["encoder_seq_length"] = ref_config.llm_cfg["max_position_embeddings"]
    model_config["num_layers"] = ref_config.llm_cfg["num_hidden_layers"]
    model_config["ffn_hidden_size"] = ref_config.llm_cfg["intermediate_size"]
    model_config["hidden_size"] = ref_config.llm_cfg["hidden_size"]
    model_config["num_attention_heads"] = ref_config.llm_cfg["num_attention_heads"]
    model_config["num_query_groups"] = ref_config.llm_cfg["num_key_value_heads"]
    model_config["layernorm_epsilon"] = ref_config.llm_cfg["rms_norm_eps"]
    model_config["init_method_std"] = ref_config.llm_cfg["initializer_range"]
    model_config["kv_channels"] = model_config["hidden_size"] // model_config["num_attention_heads"]
    #if ref_config.get("rope_scaling") is not None:
    #    if ref_config.llm_cfg["rope_scaling"]["type"] == "linear":
    #        model_config["seq_len_interpolation_factor"] = ref_config.llm_cfg["rope_scaling"]["factor"]
    #    else:
    #        raise ValueError("Only linear rope scaling type is supported now")
    model_config["use_cpu_initialization"] = True

    return model_config


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--input_name_or_path", type=str)
    parser.add_argument("--tokenizer_path", type=str)
    parser.add_argument("--conv_template", default="v1", type=str)
    parser.add_argument(
        "--hparams_file",
        type=str,
        default=os.path.join(
            os.path.dirname(__file__), '../../examples/multimodal/multimodal_llm/neva/conf/llava_config.yaml'
        ),
        required=False,
        help="Path config for restoring. It's created during training and may need to be modified during restore if restore environment is different than training. Ex: /raid/nemo_experiments/megatron_gpt/hparams.yaml",
    )
    parser.add_argument("--output_path", type=str, default=None, help="Path to output .nemo file.")
    parser.add_argument(
        "--precision", type=str, default="bf16", choices=["bf16", "32"], help="Precision for checkpoint weight saved"
    )
    parser.add_argument("--skip_verification", action="store_true")

    args = parser.parse_args()
    return args


def convert(args):
    logging.info(f"Loading checkpoint from HF Llava: `{args.input_name_or_path}`")
    hf_tokenizer, hf_model, hf_image_processor, hf_context_len = load_pretrained_model("/raid/pgibbons/models/VILA1.5-40b", model_name="vial-v1.5-40b", model_base=None, load_8bit=False, load_4bit=False,device='cpu')
    logging.info("HF Model loading done.")

    nemo_config = OmegaConf.load(args.hparams_file)
    nemo_config.model = adjust_nemo_config(nemo_config.model, hf_model.config)
    nemo_config.model.data["conv_template"] = args.conv_template
    nemo_config.model.mm_cfg.llm["model_type"] = args.conv_template
    nemo_config.model.tokenizer["model"] = args.tokenizer_path

    nemo_config.trainer["precision"] = args.precision
    trainer = MegatronTrainerBuilder(nemo_config).create_trainer()
    model = MegatronNevaModel(nemo_config.model, trainer)
    print("getting model info")
    hf_info = get_model_info(hf_model, "HF")
    nemo_info = get_model_info(model, "NeMo")
    print("saving model info")
    save_model_info(hf_info, nemo_info, "/raid/pgibbons/models/model_info.txt")
    rename_keys = create_rename_keys(nemo_config.model.num_layers)
    old_state_dict = hf_model.state_dict()
    new_state_dict = rename_model_keys(model_state_dict=old_state_dict, rename_keys=rename_keys)

    nemo_state_dict = adjust_tensor_shapes(model, new_state_dict)
    model.load_state_dict(nemo_state_dict, strict=False)

    logging.info(f'=' * 100)
    if not args.skip_verification:
        # Verifications
        input_texts = [
            'query: how much protein should a female eat',
        ]
        logging.info(f"Running verifications {input_texts} ...")

        # Tokenize the input texts
        hf_tokenizer.pad_token = hf_tokenizer.eos_token
        batch_dict = hf_tokenizer(input_texts, max_length=512, padding=True, truncation=True, return_tensors='pt')
        batch_dict_cuda = {k: v.cuda() for k, v in batch_dict.items()}
        hf_model = hf_model.cuda().eval()
        model = model.cuda().eval()

        hf_outputs = hf_model(**batch_dict_cuda, output_hidden_states=True)
        ids = batch_dict_cuda['input_ids']

        id_tensors = [torch.unsqueeze(torch.LongTensor(id_list), dim=0) for id_list in ids.cpu()]

        masks_and_position_ids = [
            get_ltor_masks_and_position_ids(id_tensor, hf_tokenizer.eos_token, False, False, False)
            for id_tensor in id_tensors
        ]
        for tokens, attn_mask_and_pos_ids in zip(id_tensors, masks_and_position_ids):
            attn_mask, _, pos_ids = attn_mask_and_pos_ids

            outputs = model(
                tokens=tokens.cuda(), text_position_ids=pos_ids.cuda(), attention_mask=attn_mask.cuda(), labels=None
            )

        hf_next_token = hf_outputs.logits[0, -1].argmax()
        next_token = outputs.squeeze()[-1].argmax()

        logging.info(f"HF predicted next token is: '{hf_tokenizer._convert_id_to_token(int(hf_next_token))}'.")
        logging.info(f"NeMo predicted next token is: '{hf_tokenizer._convert_id_to_token(int(next_token))}'.")
        assert (
            hf_next_token == next_token
        ), f'prediction mismatch: {hf_tokenizer.decode(hf_next_token)} != {hf_tokenizer.decode(next_token)}'
        logging.info(f'=' * 100)

    dtype = torch_dtype_from_precision(args.precision)
    model = model.to(dtype=dtype)
    model.save_to(args.output_path)
    logging.info(f'NeMo model saved to: {args.output_path}')


if __name__ == '__main__':
    args = get_args()
    convert(args)
