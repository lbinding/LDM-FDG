#%% Import libraries and setup directories
import wandb
from utils import save_model
from configs.wandb_setup import load_wandb_config_vAE, load_wandb_config_LDM
import torch
from torch.nn import MSELoss
from monai.transforms import DivisiblePad
import os
from torchvision.utils import save_image
from torch.amp import autocast

import numpy as np

intensity_loss = torch.nn.L1Loss()
#%% Train vAE 
def train_autoencoder(KL_autoencoder, discriminator, perceptual_loss, generator_loss, optimizer_g, optimizer_d, train_loader, val_loader, output_dir, device):
    
    wandb_config = load_wandb_config_vAE()
    KL_autoencoder.train()

    best_loss = np.inf

    for epoch in range(wandb_config["epochs"]):
        wandb.log({"epoch": epoch})
        total_g_loss = 0
        total_d_loss = 0

        for data_augmented in train_loader:
            data_augmented = data_augmented.to(device).float()

            optimizer_g.zero_grad()
            optimizer_d.zero_grad()
            with autocast(device_type='cuda', enabled=True):
                recon, z_mu, z_sigma = KL_autoencoder(data_augmented)
                gen_loss, disc_loss = generator_loss(recon, data_augmented, z_mu, z_sigma, discriminator, perceptual_loss, device) 

            gen_loss.backward(retain_graph=True)
            optimizer_g.step()

            disc_loss.backward()
            optimizer_d.step()    

            total_g_loss += gen_loss.item()
            total_d_loss += disc_loss.item()
        # Save model every epoch

        valid_loss = 0
        with torch.no_grad():
            for data in val_loader:
                with autocast(device_type='cuda', enabled=True):
                    data = data.to(device).float()
                    recon, _, _ = KL_autoencoder(data)

                    recon_loss = intensity_loss(data, recon)

                    valid_loss += recon_loss.item() # Accumulates only the Python number
        
        epoch_valid_loss = valid_loss / len(val_loader)
        wandb.log({'valid_loss': epoch_valid_loss})

        if epoch_valid_loss < best_loss:
            best_loss = epoch_valid_loss

            save_model(vAE=KL_autoencoder, model_dir=output_dir, epoch=epoch)
            save_model(discrim=discriminator, model_dir=output_dir, epoch=epoch)
                       

    return KL_autoencoder


#%% Train diffusion model 
def train_LDM(LDM_model, vAE_model, inferer, scheduler, optimizer, lr_scheduler, data_loader, output_dir, device):
    #Load in wandb config 
    wandb_config        = load_wandb_config_LDM()
    #Setup params of the latent space
    latent_channels     = 1
    latent_shape        = (64, 64)
    #Setup the loss 
    mse_loss = MSELoss()
    #Set padding method for latent 
    pad_transform = DivisiblePad(k=(1, 32, 32))

    #Set the modes of the models used 
    vAE_model.eval()
    LDM_model.train()
    #Loop through epochs 
    for epoch in range(wandb_config["epochs"]):
        wandb.log({"epoch": epoch})
        total_loss = 0
        for data_raw in data_loader:
            data_raw = data_raw.to(device).float()
            '''
            Commented out the below as this is for use with diffusion not the LDM inferer 
            '''
            # Get reconstructions and latent variables from the vAE
            with torch.no_grad():
                z_mu, z_sigma = vAE_model.encode(data_raw)
            # Pad z_mu 
            z_mu_padded = pad_transform(z_mu)
            # Create noise like the padded 
            noise = torch.randn_like(z_mu_padded).to(device)

            #noise_shape = [wandb_config['batch_size'], latent_channels, *latent_shape]
            #noise = torch.randn_like(noise_shape).to(device)
            
            # Create the timesteps to learn 
            timesteps = torch.randint(0, inferer.scheduler.num_train_timesteps, (noise.shape[0],), device=data_raw.device)
            
            # Set grads to zero 
            optimizer.zero_grad()

            # Predict noise 
            noise_pred = inferer(
                inputs=z_mu_padded, diffusion_model=LDM_model.float(), noise=noise, timesteps=timesteps#, autoencoder_model=vAE_model.float()
            )

            # Calculate loss 
            loss = mse_loss(noise_pred.float(), noise.float())
            # Update weights/biases 
            loss.backward()
            # Step optimizer 
            optimizer.step()
            # Add loss to total loss 
            total_loss += loss.item()
        
            wandb.log({"mse_loss": loss.mean().item()}) 
        #Step the learning rate scheduler
        lr_scheduler.step()
        
        #If epoch is divisable by 5
        if epoch % 5 == 0:
            #Save model
            save_model(LDM=LDM_model, model_dir=output_dir, epoch=epoch)

            #Generate some noise 
            z = torch.randn((1, latent_channels, *latent_shape))
            z = z.to(device)
            
            #Sample the LDM 
            decoded = inferer.sample(
                input_noise=z, diffusion_model=LDM_model, scheduler=scheduler#, autoencoder_model=vAE_model, save_intermediates=True, intermediate_steps=100
            )
            #
            recon = vAE_model.decode(decoded)

            # Ensure the output directory exists
            os.makedirs(output_dir, exist_ok=True)

            # Save the decoded image as a PNG
            save_image(recon, os.path.join(output_dir, f"decoded_epoch_{epoch}.png"))

        #Print epoch 
        print(f"Epoch {epoch+1}/{wandb_config['epochs']}, Loss: {total_loss/len(data_loader):.4f}")
