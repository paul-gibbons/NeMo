import torch
from torch.utils.data import DataLoader
from megatron.energon import get_loader, get_train_dataset
from nemo.collections.common.tokenizers import SentencePieceTokenizer
from transformers import CLIPImageProcessor

# Import your NevaTaskEncoder class
from neva_energon_dataset import NevaTaskEncoder  # Replace 'your_module' with the actual module name

# Configuration
multimodal_cfg = {
    "is_multimodal": True,
    "sep_image_conv_front": False,
    "model_type": "plain",
    "conv_template": "plain",
    "patch_dim": 14,
    "crop_size": (224, 224),
    "image_aspect_ratio": 'square',
    "use_im_start_end": False,
    "image_processor": "openai/clip-vit-large-patch14",
    "add_extra_token": 0,
    "context_length": 4096,
    "media_type": 'image',
    "num_frames": -1,
    "mm_mlp_adapter_type": 'linear',
}

data_cfg = {
    "splice_single_frame": None,
    "num_frames": -1,
    "sep_token_between_frames": False,
}

# Initialize tokenizer and image processor
tokenizer = SentencePieceTokenizer(
    model_path='/raid/pgibbons/models/tokenizer_add_special.model',
    special_tokens=None,
    legacy=False
)

image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")

# Initialize NevaTaskEncoder
neva_encoder = NevaTaskEncoder(
    tokenizer=tokenizer,
    image_processor=image_processor,
    multimodal_cfg=multimodal_cfg,
    data_cfg=data_cfg
)

# Set up dataset and data loader
#data_path = "/raid/pgibbons/data/energon_datasets/LLaVA-Pretrain-LCS-558K"
data_path = "/raid/pgibbons/data/energon_datasets/energon_datasets/obelisc/stage4/no-partial"
batch_size = 1
shuffle_buffer_size = 2

train_dataset = get_train_dataset(
    data_path,
    batch_size=batch_size,
    shuffle_buffer_size=shuffle_buffer_size,
    max_samples_per_sequence=2,
    task_encoder=neva_encoder,
    image_decode="torchrgb",
)

train_loader = get_loader(train_dataset)

# Test the data loader
def test_data_loader(loader, num_batches=4):
    for i, batch in enumerate(loader):
        if i >= num_batches:
            break
        
        print(f"Batch {i + 1}:")
        print(f"  Keys: {batch.keys()}")
        
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}:")
                print(f"    Shape: {value.shape}")
                print(f"    Dtype: {value.dtype}")
                print(f"    Device: {value.device}")
                print(f"    Matrix:")
                print(value)
            elif isinstance(value, list):
                print(f"  {key}:")
                print(f"    Type: List")
                print(f"    Length: {len(value)}")
                print(f"    Content:")
                print(value)
            else:
                print(f"  {key}:")
                print(f"    Type: {type(value)}")
                print(f"    Value: {value}")
        
        print("\n")

# Run the test
print("Testing data loader...")
test_data_loader(train_loader)

# Run the test
print("Testing data loader...")
test_data_loader(train_loader)
