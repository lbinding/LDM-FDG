#Move directory (for testing purposes)
#__file__='/Users/lawrencebinding/Desktop/projects/github/LDM-FDG/LDM-FDG/Main.py'  
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

#%% Libs
from models.model import load_KL_autoencoder, load_LDM
from dataset import FDG_Dataset
from torchio_aug import Augmentations
from KLautoencoder_losses import generator_loss, load_perceptual_loss, load_discriminator
from train_model import train_autoencoder, train_LDM
from configs.wandb_setup import load_wandb_config_vAE, load_wandb_config_LDM
from torch.utils.data import DataLoader
#from monai.data import DataLoader
import torch
import torch.optim as optim
#import wandb
from monai.networks.schedulers import DDPMScheduler 
from monai.inferers.inferer import DiffusionInferer
from torch.optim.lr_scheduler import MultiStepLR
import argparse

#%% Setup device 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#%% Setup paths
#Define directories
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(parent_dir, "training_data", "slices")
base_dir = os.path.dirname(os.path.abspath(__file__))

#%% Input arguments (Sets training to true)
#       -- REMOVE THIS FOR SUMMER SCHOOL --
#           Hacky solution for resuming 

# Set default values: 
train_autoencoder_enabled   = False
train_ldm_enabled           = False
autoencoder_weights         = os.path.join(base_dir, 'models', 'model_autoencoder.pt')
LDM_weights                 = os.path.join(base_dir, 'models', 'model.pt')

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Set boolean variables for training.")
parser.add_argument("--train_vAE", dest='train_autoencoder', action="store_true", help="Enable training of the autoencoder.", required=False)
parser.add_argument("--train_LDM", dest='train_ldm', action="store_true", help="Enable training of the latent diffusion model.", required=False)
parser.add_argument("--vAE_model", dest='vAE_weights',
                    help="Filepath to a custom pre-trained vAE model. Overrides the default model path.", required=False)
parser.add_argument("--LDM_model", dest='LDM_weights',
                    help="Filepath to a custom pre-trained LDM model. Overrides the default model path.", required=False)
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
wandb_config    = load_wandb_config_vAE()
#Create output directory 
output_dir      = os.path.join(parent_dir, "output", wandb_config['architecture'], wandb_config['dataset'])

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

#%% Setup dataset and augmentation
transforms  = Augmentations()
dataset     = FDG_Dataset(data_dir=data_dir, transform=transforms)
data_loader = DataLoader(dataset, batch_size=wandb_config['batch_size'], shuffle=True)

#%% Load losses 
perceptual_loss = load_perceptual_loss(device=device).float()
discriminator   = load_discriminator(device=device).float()

#%% Load the model and optimizer 
KL_autoencoder  = load_KL_autoencoder(autoencoder_weights, 
                                        config_file="train_autoencoder.json", 
                                        device=device).float()

optimizer       = optim.Adam(params=list(KL_autoencoder.parameters())+ list(discriminator.parameters()) , lr=wandb_config["learning_rate"])

#%% Train the model 
if train_autoencoder_enabled is True:
    print("Training vAE")
    KL_autoencoder = train_autoencoder(KL_autoencoder,
                                        discriminator,
                                        perceptual_loss,
                                        generator_loss,
                                        optimizer,
                                        data_loader,
                                        output_dir=output_dir,
                                        device=device)

#%% Load the LDM model
LDM_model = load_LDM(LDM_weights,
                     config_file="train_diffusion.json",  
                     device=device).float()

optimizer           = optim.Adam(params=list(LDM_model.parameters()) , lr=5e-05)
lr_scheduler        = MultiStepLR(optimizer=optimizer, milestones=[1000], gamma=0.1)
#%% Setup a new data-loader (without augmentations)
#Load in the LDM wandb_config 
wandb_config = load_wandb_config_LDM()

#Setup params 
num_train_timesteps=1000

#Create dataset / loader without transforms 
dataset     = FDG_Dataset(data_dir=data_dir, transform=None)
data_loader = DataLoader(dataset, batch_size=wandb_config['batch_size'], shuffle=True)

#Compute scale factor 
#check_data = next(iter(data_loader))
#z = KL_autoencoder.encode_stage_2_inputs(check_data.float().to(device))
#scale_factor = 1 / torch.std(z)

#Create scheduler / inferer 
scheduler = DDPMScheduler(num_train_timesteps=num_train_timesteps, schedule='scaled_linear_beta', beta_start=0.0015, beta_end=0.0195).to(device)
#inferer = LatentDiffusionInferer(scheduler, scale_factor=scale_factor, ldm_latent_shape=[64,64],autoencoder_latent_shape=[60,60])
inferer = DiffusionInferer(scheduler)

#%% Train LDM 
if train_ldm_enabled is True: 
    print("Training LDM")
    train_LDM(LDM_model, 
                KL_autoencoder, 
                inferer,
                scheduler,
                optimizer,
                lr_scheduler, 
                data_loader, 
                output_dir,
                device)

