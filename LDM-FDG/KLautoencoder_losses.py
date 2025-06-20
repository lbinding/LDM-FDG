#%% Libs
from pathlib import Path
import torch
import wandb
from monai.bundle import ConfigParser
from monai.losses import PerceptualLoss
from monai.losses.adversarial_loss import PatchAdversarialLoss
from torch.amp import autocast

#%% Paths and Config
base_dir = Path(__file__).parent.resolve()
weights_path = base_dir / "models" / "model_discriminator.pt"
config = ConfigParser()
config.read_config(base_dir / "configs" / "train_autoencoder.json")

#%% Weights for each component
adv_weight = 0.5
perceptual_weight = 1.0
kl_weight = 1e-6  # KL regularization weight

#%% Loss components
intensity_loss = torch.nn.L1Loss()
adv_loss = PatchAdversarialLoss(criterion="least_squares")

#%% KL divergence loss
def compute_kl_loss(z_mu, z_sigma):
    kl_loss = 0.5 * torch.sum(
        z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1,
        dim=list(range(1, len(z_sigma.shape)))
    )
    return torch.mean(kl_loss)

#%% Perceptual loss (ResNet50)
def load_perceptual_loss(device):
    perceptual_loss = PerceptualLoss(
        spatial_dims=2,
        network_type="resnet50",
        pretrained=True,
    )
    perceptual_loss.to(device)
    return perceptual_loss

#%% Discriminator
def load_discriminator(device):
    discriminator = config.get_parsed_content("dnetwork")
    discriminator.to(device)
    return discriminator

#%% Generator loss function
def generator_loss(gen_images, real_images, z_mu, z_sigma, disc_net, loss_perceptual, device):
    
    with autocast(device_type=device, enabled=True):

        recons_loss = intensity_loss(gen_images, real_images)
        wandb.log({"intensity loss": recons_loss})

        kl = compute_kl_loss(z_mu, z_sigma)
        wandb.log({"kl loss": kl})

        p_loss = loss_perceptual(gen_images, real_images)
        wandb.log({"perceptual loss": p_loss})

        # Base generator loss (reconstruction + KL + perceptual)
        loss_g = recons_loss + kl_weight * kl + perceptual_weight * p_loss
        wandb.log({"gen base loss": loss_g})

        # Adversarial component
        logits_fake = disc_net(gen_images)[-1]
        gen_adv_loss = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
        wandb.log({"adversarial loss": gen_adv_loss})

        loss_g += adv_weight * gen_adv_loss
        wandb.log({"total generator loss": loss_g})


        d_loss_fake = adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
        logits_real = disc_net(real_images.contiguous().detach())[-1]
        d_loss_real = adv_loss(logits_real, target_is_real=True, for_discriminator=True)
        discriminator_loss = (d_loss_fake + d_loss_real) * 0.5
        loss_d = adv_weight * discriminator_loss


    return loss_g, loss_d
