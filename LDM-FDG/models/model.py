# models/model.py

from monai.bundle import ConfigParser
import torch
from pathlib import Path

def load_KL_autoencoder(weights_path, config_file="train_autoencoder.json", device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup paths
    base_dir = Path(__file__).parent.resolve().parent
    config_path = base_dir / "configs" / config_file
    #weights_path = base_dir / "models" / weights_file

    print(config_path)

    # Read config
    config = ConfigParser()
    config.read_config(str(config_path))

    # Parse model
    model = config.get_parsed_content("gnetwork")

    # Load checkpoint
    checkpoint = torch.load(weights_path, map_location=device)

    # Key remapping
    key_mapping = {
        "encoder.blocks.10.to_q.weight": "encoder.blocks.10.attn.to_q.weight",
        "encoder.blocks.10.to_q.bias": "encoder.blocks.10.attn.to_q.bias",
        "encoder.blocks.10.to_k.weight": "encoder.blocks.10.attn.to_k.weight",
        "encoder.blocks.10.to_k.bias": "encoder.blocks.10.attn.to_k.bias",
        "encoder.blocks.10.to_v.weight": "encoder.blocks.10.attn.to_v.weight",
        "encoder.blocks.10.to_v.bias": "encoder.blocks.10.attn.to_v.bias",
        "encoder.blocks.10.proj_attn.weight": "encoder.blocks.10.attn.out_proj.weight",
        "encoder.blocks.10.proj_attn.bias": "encoder.blocks.10.attn.out_proj.bias",
        "decoder.blocks.2.to_q.weight": "decoder.blocks.2.attn.to_q.weight",
        "decoder.blocks.2.to_q.bias": "decoder.blocks.2.attn.to_q.bias",
        "decoder.blocks.2.to_k.weight": "decoder.blocks.2.attn.to_k.weight",
        "decoder.blocks.2.to_k.bias": "decoder.blocks.2.attn.to_k.bias",
        "decoder.blocks.2.to_v.weight": "decoder.blocks.2.attn.to_v.weight",
        "decoder.blocks.2.to_v.bias": "decoder.blocks.2.attn.to_v.bias",
        "decoder.blocks.2.proj_attn.weight": "decoder.blocks.2.attn.out_proj.weight",
        "decoder.blocks.2.proj_attn.bias": "decoder.blocks.2.attn.out_proj.bias",
        "decoder.blocks.6.conv.conv.weight": "decoder.blocks.6.postconv.conv.weight",
        "decoder.blocks.6.conv.conv.bias": "decoder.blocks.6.postconv.conv.bias",
        "decoder.blocks.9.conv.conv.weight": "decoder.blocks.9.postconv.conv.weight",
        "decoder.blocks.9.conv.conv.bias": "decoder.blocks.9.postconv.conv.bias",
    }

    # Remap keys
    new_state_dict = {key_mapping.get(k, k): v for k, v in checkpoint.items()}

    # Load state
    model.load_state_dict(new_state_dict, strict=False)
    model.to(device)

    return model

#%%

def load_LDM(weights_path, config_file="train_diffusion.json", device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup paths
    base_dir = Path(__file__).parent.resolve().parent
    config_path = base_dir / "configs" / config_file
    #weights_path = base_dir / "models" / weights_file

    # Read config
    config = ConfigParser()
    config.read_config(str(config_path))

    # Parse model
    model = config.get_parsed_content("diffusion")

    # Load checkpoint
    checkpoint = torch.load(weights_path, map_location=device)

    # Load state
    model.load_state_dict(checkpoint, strict=False)
    model.to(device)

    return model