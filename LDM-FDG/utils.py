import torch 
import os 

def save_model(vAE=None, LDM=None, discrim=None, model_dir=None, epoch=None):
    """
    Save the model state dictionary to a specified directory with epoch information.
    
    Args:
        vAE (torch.nn.Module, optional): The Variational Autoencoder model to save.
        LDM (torch.nn.Module, optional): The Latent Diffusion Model to save.
        model_dir (str): Directory where the models will be saved.
        epoch (int): Current epoch number for naming the file.
    """
    if model_dir is None or epoch is None:
        raise ValueError("Both 'model_dir' and 'epoch' must be provided.")

    if vAE is not None:
        model_save_path = os.path.join(model_dir, "models", f"trained_vAE_epoch_{epoch}.pt")
        torch.save(vAE.state_dict(), model_save_path)
        print(f"vAE model saved at {model_save_path}")
    
    if LDM is not None:
        model_save_path = os.path.join(model_dir, "models", f"trained_LDM_epoch_{epoch}.pt")
        torch.save(LDM.state_dict(), model_save_path)
        print(f"LDM model saved at {model_save_path}")

    if discrim is not None:
        discrim_save_path = os.path.join(model_dir, "models", f"trained_discriminator_epoch_{epoch}.pt")
        torch.save(discrim.state_dict(), discrim_save_path)
        print(f"Discriminator saved at {discrim_save_path}")

