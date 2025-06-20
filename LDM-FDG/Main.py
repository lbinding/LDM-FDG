# In Main.py

#Move directory (for testing purposes)
#__file__='/Users/lawrencebinding/Desktop/projects/github/LDM-FDG/LDM-FDG/Main.py'
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

#%% Libs
from models.model import load_KL_autoencoder, load_LDM
from dataset import FDG_Dataset
from torchio_aug import Augmentations
from monai_aug import MonaiAugmentations2D
from KLautoencoder_losses import generator_loss, load_perceptual_loss, load_discriminator
from train_model import train_autoencoder, train_LDM
from configs.wandb_setup import load_wandb_config_vAE, load_wandb_config_LDM
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
from monai.networks.schedulers import DDPMScheduler
from monai.inferers.inferer import DiffusionInferer
from torch.optim.lr_scheduler import MultiStepLR
import argparse
from pathlib import Path
import numpy as np

#%% Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#%% Setup paths
#Define directories
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(parent_dir, "training_data", "slices")
base_dir = os.path.dirname(os.path.abspath(__file__))

#%% Input arguments (Sets training to true)
# Set default values:
train_autoencoder_enabled   = True
train_ldm_enabled           = False
autoencoder_weights         = os.path.join(base_dir, 'models', 'model_autoencoder.pt')
LDM_weights                 = os.path.join(base_dir, 'models', 'model.pt')

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Set boolean variables for training.")
parser.add_argument("--train_vAE", dest='train_autoencoder', action="store_true", help="Enable training of the autoencoder.", required=False)
parser.add_argument("--train_LDM", dest='train_ldm', action="store_true", help="Enable training of the latent diffusion model.", required=False)
parser.add_argument("--vAE_model", dest='vAE_weights', help="Filepath to a custom pre-trained vAE model.", required=False)
parser.add_argument("--LDM_model", dest='LDM_weights', help="Filepath to a custom pre-trained LDM model.", required=False)
args = parser.parse_args()

# Boolean variables
train_autoencoder_enabled   = args.train_autoencoder
train_ldm_enabled           = args.train_ldm

# Overwrite default weights if they exist
if args.vAE_weights:
    print("Loading in custom vAE weights")
    autoencoder_weights = os.path.abspath(args.vAE_weights)
if args.LDM_weights:
    print("Loading in custom LDM weights")
    LDM_weights         = os.path.abspath(args.LDM_weights)

#%% Load vAE wandb config & Setup output directories
wandb_config_vAE = load_wandb_config_vAE()
#Create output directory
output_dir       = os.path.join(parent_dir, "output", wandb_config_vAE['architecture'], wandb_config_vAE['dataset'])
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# =================================================================================
# NEW: Data Splitting Section
# =================================================================================
print("Splitting data into training, validation, and test sets...")

# 1. Define split ratios and random seed for reproducibility
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1 # Must sum to 1.0 with the others
random_seed = 42

# 2. Get all file paths from the data directory
all_files = sorted(list(Path(data_dir).glob("*.nii.gz")))
num_files = len(all_files)
print(f"Found {num_files} total files.")

# 3. Shuffle the files
np.random.seed(random_seed)
np.random.shuffle(all_files)

# 4. Calculate split indices
train_split_idx = int(num_files * train_ratio)
val_split_idx = int(num_files * (train_ratio + val_ratio))

# 5. Create file lists for each set
train_files = all_files[:train_split_idx]
val_files = all_files[train_split_idx:val_split_idx]
test_files = all_files[val_split_idx:]

print(f"Training set size: {len(train_files)}")
print(f"Validation set size: {len(val_files)}")
print(f"Test set size: {len(test_files)}")

# =================================================================================
# Setup Datasets and DataLoaders for vAE Training
# =================================================================================
# Augmentations are applied only to the training set for the vAE
aug_transforms = Augmentations()

# Create Dataset instances
train_dataset = FDG_Dataset(file_paths=train_files, transform=aug_transforms)
val_dataset       = FDG_Dataset(file_paths=val_files, transform=None) # No augmentation for validation
test_dataset      = FDG_Dataset(file_paths=test_files, transform=None)  # No augmentation for testing

# Create DataLoader instances
# Note: shuffle=False for val and test loaders for consistent evaluation
train_loader = DataLoader(train_dataset, batch_size=wandb_config_vAE['batch_size'], shuffle=True)
val_loader       = DataLoader(val_dataset, batch_size=wandb_config_vAE['batch_size'], shuffle=False)
test_loader      = DataLoader(test_dataset, batch_size=wandb_config_vAE['batch_size'], shuffle=False)


#%% Load losses
perceptual_loss = load_perceptual_loss(device=device).float()
discriminator   = load_discriminator(device=device).float()

#%% Load the vAE model and optimizer
KL_autoencoder  = load_KL_autoencoder(autoencoder_weights,
                                        config_file="train_autoencoder.json",
                                        device=device).float()

optimizer_g   = optim.Adam(params=list(KL_autoencoder.parameters()), lr=wandb_config_vAE["learning_rate"])
optimizer_d   = optim.Adam(params=list(discriminator.parameters()), lr=wandb_config_vAE["learning_rate"])


#%% Train the vAE model
if train_autoencoder_enabled is True:
    print("Training vAE")
    # Pass both training and validation loaders to the training function
    # You might need to adapt your `train_autoencoder` function to use the validation loader
    KL_autoencoder = train_autoencoder(KL_autoencoder,
                                        discriminator,
                                        perceptual_loss,
                                        generator_loss,
                                        optimizer_g,
                                        optimizer_d,
                                        train_loader, # Use the vAE specific training loader
                                        val_loader,       # Pass validation loader for evaluation
                                        output_dir=output_dir,
                                        device=device)

# # =================================================================================
# # LDM Training Section
# # =================================================================================
# #%% Load the LDM wandb_config
# wandb_config_LDM = load_wandb_config_LDM()

# #%% Load the LDM model
# LDM_model = load_LDM(LDM_weights,
#                      config_file="train_diffusion.json",
#                      device=device).float()

# optimizer_LDM       = optim.Adam(params=list(LDM_model.parameters()), lr=wandb_config_LDM.get("learning_rate", 5e-5))
# lr_scheduler        = MultiStepLR(optimizer=optimizer_LDM, milestones=[1000], gamma=0.1)

# #%% Setup a new data-loader for LDM (without augmentations, using the SAME training split)
# print("Setting up DataLoader for LDM training...")
# # Create a new dataset instance from the *same* training file list, but without augmentations
# train_dataset_LDM = FDG_Dataset(data_dir=train_files, transform=None)
# train_loader_LDM = DataLoader(train_dataset_LDM, batch_size=wandb_config_LDM['batch_size'], shuffle=True)

# #Compute scale factor (optional, as in your original script)
# # with torch.no_grad():
# #     check_data = next(iter(train_loader_LDM))
# #     z = KL_autoencoder.encode_stage_2_inputs(check_data.float().to(device))
# #     scale_factor = 1 / torch.std(z)
# #     print(f"Calculated LDM scale_factor: {scale_factor}")


# #%% Create scheduler / inferer
# num_train_timesteps = 1000
# scheduler = DDPMScheduler(num_train_timesteps=num_train_timesteps, schedule='scaled_linear_beta', beta_start=0.0015, beta_end=0.0195).to(device)
# inferer = DiffusionInferer(scheduler)

# #%% Train LDM
# if train_ldm_enabled is True:
#     print("Training LDM")
#     # Pass the LDM-specific training loader and the same validation loader
#     # You might need to adapt your `train_LDM` function to use the validation loader
#     train_LDM(LDM_model,
#               KL_autoencoder,
#               inferer,
#               scheduler,
#               optimizer_LDM,
#               lr_scheduler,
#               train_loader_LDM, # Use the LDM specific training loader
#               val_loader,       # Re-use the same validation loader
#               output_dir,
#               device)