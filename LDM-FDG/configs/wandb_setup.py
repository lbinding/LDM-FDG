import wandb

def load_wandb_config_vAE():
    """
    Load the Weights & Biases configuration 
            
    Returns:
        dict: wandb config.
    """
    wandb.login(key="76c124f9bfc89b958db96f3de53b29ddbfa1feb5")

    wandb_config = {
        "learning_rate": 0.00005,
        "architecture": "Autoencoder_KL",
        "dataset": "FDG_2D_slices",
        "epochs": 500,
        "batch_size": 5,
        }

    wandb.init(
        # set the wandb project where this run will be logged
        project="LDM_FDG_vae",
        # track hyperparameters and run metadata
        config=wandb_config,
        mode="online"  # Ensure wandb is properly initialized
    )
    return wandb_config

def load_wandb_config_LDM():
    """
    Load the Weights & Biases configuration 
            
    Returns:
        dict: wandb config.
    """
    wandb.login(key="76c124f9bfc89b958db96f3de53b29ddbfa1feb5")

    wandb_config = {
        "learning_rate": 0.00005,
        "architecture": "LDM",
        "dataset": "FDG_2D_slices",
        "epochs": 500,
        "batch_size": 24,
        }

    wandb.init(
        # set the wandb project where this run will be logged
        project="LDM_FDG_ldm",
        # track hyperparameters and run metadata
        config=wandb_config,
        mode="online"  # Ensure wandb is properly initialized
    )
    return wandb_config
