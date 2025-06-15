import torch
import torch.nn as nn
from typing import Optional, Union, Tuple, List, TypeVar
from tqdm import tqdm
from scheduler import DDPMScheduler

# Placeholder for type hints
T_Tensor = TypeVar("T_Tensor", bound=torch.Tensor)

# --- MODIFIED DiffusionInferer ---
class DiffusionInferer(nn.Module):
    """
    DiffusionInferer orchestrates the interaction between a trained diffusion model
    and a scheduler for training data preparation and inference (sampling).

    This simplified version focuses on:
    - Training forward pass (adding noise and getting model prediction).
    - Image generation (sampling) with standard cross-attention conditioning.
    - It assumes the DDPMScheduler provides a linear beta schedule, epsilon prediction,
      and fixed small variance.
    - It does NOT include log-likelihood computation or specialized SPADE model handling.

    Args:
        scheduler: An instance of DDPMScheduler.
        diffusion_model: An **instance** of your diffusion model (any torch.nn.Module).
                         This model instance must implement a `forward(x, timesteps, context=None)`
                         method and will be used directly for predictions.
    """

    def __init__(self, scheduler: DDPMScheduler, diffusion_model: nn.Module) -> None:
        super().__init__()
        self.scheduler = scheduler
        self.diffusion_model = diffusion_model # Store the model instance directly

    def __call__(
        self,
        inputs: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Implements the **forward pass for a supervised training iteration**.
        In training, we typically add noise to the input (x_0) and ask the model
        to predict the noise (epsilon) or the denoised sample (x_0).

        Args:
            inputs: Input image to which noise is added (x_0).
            noise: Random noise, of the same shape as the input (epsilon).
            timesteps: Random timesteps (t) for each image in the batch.
            condition: Conditioning for network input (e.g., class label embeddings, text embeddings).

        Returns:
            The model's prediction (e.g., predicted noise, predicted x_0).
        """
        # Add noise to the original samples to get noisy_image (x_t)
        noisy_image: torch.Tensor = self.scheduler.add_noise(original_samples=inputs, noise=noise, timesteps=timesteps)
        
        # Predict the data: The diffusion model takes noisy_image, timesteps, and context.
        # It's assumed to predict the noise (epsilon) for DDPM training.
        prediction: torch.Tensor = self.diffusion_model(x=noisy_image, timesteps=timesteps, context=condition)

        return prediction

    @torch.no_grad()
    def sample(
        self,
        input_noise: torch.Tensor,
        scheduler: Optional[DDPMScheduler] = None,
        save_intermediates: Optional[bool] = False,
        intermediate_steps: Optional[int] = 100,
        conditioning: Optional[torch.Tensor] = None,
        verbose: bool = True,
        generator: Optional[torch.Generator] = None, 
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        Samples a new image from the diffusion model by reversing the diffusion process.
        This is the **inference (generation) loop**.

        Args:
            input_noise: Random noise, of the same shape as the desired sample (initial x_T).
            scheduler: Diffusion scheduler. If None provided, will use the class attribute scheduler.
            save_intermediates: Whether to return intermediate samples along the sampling chain.
            intermediate_steps: If save_intermediates is True, saves every n steps.
            conditioning: Conditioning for network input (e.g., class label embeddings).
            verbose: If true, prints a progression bar of the sampling process.
            generator: Optional torch.Generator for reproducible sampling noise.

        Returns:
            The generated image (x_0) or a tuple of (generated_image, list_of_intermediates).
        """
        # Use the provided scheduler or the class's default scheduler
        scheduler = scheduler if scheduler is not None else self.scheduler
        
        # Start with pure noise (x_T)
        image = input_noise 
        
        # Initialize progress bar for visualization
        progress_bar = tqdm(scheduler.timesteps, desc="Sampling Progress")

        intermediates = []
        for t in progress_bar:
            # Ensure timestep is a tensor and on the correct device for the model
            timesteps_tensor = torch.full((image.shape[0],), t, device=input_noise.device, dtype=torch.long)

            # 1. Predict model_output (noise, assumed epsilon) using the diffusion model
            # Now accessing the model instance via self.diffusion_model
            model_output = self.diffusion_model(image, timesteps=timesteps_tensor, context=conditioning)

            # 2. Compute previous image: x_t -> x_t-1 using the scheduler's step function
            image, _ = scheduler.step(model_output, t, image, generator=generator) 

            # Save intermediates if enabled and at the specified interval
            if save_intermediates and t % intermediate_steps == 0:
                intermediates.append(image.cpu()) 

        if save_intermediates:
            return image, intermediates
        else:
            return image
