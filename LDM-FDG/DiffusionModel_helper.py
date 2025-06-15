import torch
import torch.nn as nn
from typing import Optional, Union, Tuple, List, TypeVar
from tqdm import tqdm

# Placeholder for type hints
T_Tensor = TypeVar("T_Tensor", bound=torch.Tensor)

# --- MODIFIED DDPMScheduler ---
class DDPMScheduler(nn.Module):
    """
    DDPMScheduler implements the noise scheduling and reverse diffusion steps for a Denoising Diffusion
    Probabilistic Model (DDPM). It's responsible for managing the noise levels and performing the
    iterative denoising process during inference.

    This simplified version of monai's implementation uses a fixed linear beta schedule, assumes the model predicts
    the noise (epsilon), and uses a fixed small variance for the reverse steps.
    """

    def __init__(
        self,
        num_train_timesteps: int,
        clip_sample: bool = True,
        clip_sample_values: Tuple[float, float] = (-1.0, 1.0),
    ):
        #Super __init__ to initialise vars within nn.Module
        super().__init__()

        # Setup Linear noise scheduler (betas)
        beta_start  = 1e-4
        beta_end    = 2e-2
        noise_sched = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
        
        #Set the number of training timesteps to a global var 
        self.num_train_timesteps = num_train_timesteps
        
        #Set the linear noise scheduler to the beta values 
        self.betas = noise_sched # These are the beta values (noise schedule)
        
        #Set alpha to inverse of betas 
        self.alphas = 1.0 - self.betas
        
        #Compute the culumative product of alphas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0) # alpha-bar values
        
        # Timesteps for inference typically go from largest (T-1) to smallest (0)
        self.timesteps = torch.arange(num_train_timesteps - 1, -1, -1, dtype=torch.long)
        # Used for t=0 boundary conditions
        self.one = torch.tensor(1.0) 
        
        #This can be set during inferece – it reduces the number of steps for inference
        self.num_inference_steps: Optional[int] = None 

        # Parameters for clipping predicted x_0
        #   This is used to clip the predicted output values between the expected range (e.g., -1:1)
        #   This helps create numerical stability and realistic outputs 
        self.clip_sample = clip_sample
        self.clip_sample_values = clip_sample_values

    def unsqueeze_right(self, arr: T_Tensor, ndim: int) -> T_Tensor:
        """Appends 1-sized dimensions to `arr` to ensure it has `ndim` dimensions for broadcasting."""
        if arr.ndim >= ndim:
            return arr # Already has enough dimensions or more
        return arr[(...,) + (None,) * (ndim - arr.ndim)]

    def add_noise(
        self, original_samples: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor
    ) -> torch.Tensor:
        """
        Adds noise to the original samples according to the diffusion process.
        This is the **forward diffusion step**, used during training to create noisy inputs.

        Args:
            original_samples: The clean data (x_0).
            noise: Random noise to add (epsilon).
            timesteps: The timesteps to noise the samples to (t).

        Returns:
            The noisy samples (x_t) at the given timesteps.
        """
        # Ensure all tensors are on the same device and dtype as the original samples
        self.alphas_cumprod = self.alphas_cumprod.to(device=original_samples.device, dtype=original_samples.dtype)
        timesteps = timesteps.to(original_samples.device)

        # Reshape alphas_cumprod to match sample dimensions for element-wise multiplication
        sqrt_alpha_prod = self.unsqueeze_right(self.alphas_cumprod[timesteps] ** 0.5, original_samples.ndim)
        sqrt_one_minus_alpha_prod = self.unsqueeze_right(
            (1 - self.alphas_cumprod[timesteps]) ** 0.5, original_samples.ndim
        )
        
        # Create noisy samples 
        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples

    def _get_variance(self, timestep: int) -> torch.Tensor:
        """
        Computes the fixed variance for the reverse diffusion step at a given timestep.
        This corresponds to the 'fixed_small' variance type from the original DDPM paper.

        Args:
            timestep: Current timestep.

        Returns:
            The variance for the reverse step.
        """
        # Ensure tensors are on the correct device and dtype
        device, dtype = self.betas.device, self.betas.dtype
        alpha_prod_t = self.alphas_cumprod[timestep].to(device=device, dtype=dtype)
        alpha_prod_t_prev = self.alphas_cumprod[timestep - 1].to(device=device, dtype=dtype) if timestep > 0 else self.one.to(device=device, dtype=dtype)
        beta_t = self.betas[timestep].to(device=device, dtype=dtype)
        
        # Calculate fixed variance βt (formula 7 from https://arxiv.org/pdf/2006.11239.pdf)
        variance: torch.Tensor = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * beta_t

        # Clamp variance to prevent numerical issues with very small values
        variance = torch.clamp(variance, min=1e-20)
        return variance

    def step(
        self, model_output: torch.Tensor, timestep: int, sample: torch.Tensor, generator: Optional[torch.Generator] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predicts the sample at the previous timestep by reversing the diffusion process.
        This is the core function for **inference (sampling)**.

        Args:
            model_output: The direct output from the learned diffusion model (expected to be predicted noise, epsilon).
            timestep: Current discrete timestep in the diffusion chain (t).
            sample: Current noisy sample (x_t).
            generator: Random number generator for reproducibility.

        Returns:
            A tuple containing:
                - **pred_prev_sample**: The predicted sample at timestep t-1 (x_{t-1}).
                - **pred_original_sample**: The model's estimate of the original (denoised) sample (x_0).
        """
        # Ensure tensors are on the correct device and dtype
        device, dtype = sample.device, sample.dtype
        self.alphas = self.alphas.to(device=device, dtype=dtype)
        self.alphas_cumprod = self.alphas_cumprod.to(device=device, dtype=dtype)
        self.betas = self.betas.to(device=device, dtype=dtype)
        
        # 1. Compute alphas, betas for current and previous timestep
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[timestep - 1] if timestep > 0 else self.one
        beta_prod_t = 1 - alpha_prod_t # (1 - alpha_bar_t)
        beta_prod_t_prev = 1 - alpha_prod_t_prev # (1 - alpha_bar_t-1)

        # 2. Compute predicted original sample (x_0) from the model's predicted noise (epsilon)
        # Rearranging the forward diffusion equation: x_0 = (x_t - sqrt(1-alpha_bar_t) * epsilon) / sqrt(alpha_bar_t)
        pred_original_sample = (sample - self.unsqueeze_right(beta_prod_t ** 0.5, sample.ndim) * model_output) / \
                               self.unsqueeze_right(alpha_prod_t ** 0.5, sample.ndim)

        # 3. Clip "predicted x_0" if enabled, to keep values within a reasonable range
        if self.clip_sample:
            pred_original_sample = torch.clamp(
                pred_original_sample, self.clip_sample_values[0], self.clip_sample_values[1]
            )

        # 4. Compute coefficients for combining predicted x_0 and current x_t
        # These coefficients are derived from the posterior mean formula (7) in DDPM paper
        pred_original_sample_coeff = (self.unsqueeze_right(alpha_prod_t_prev ** 0.5, sample.ndim) * self.unsqueeze_right(self.betas[timestep], sample.ndim)) / \
                                     self.unsqueeze_right(beta_prod_t, sample.ndim)
        current_sample_coeff = (self.unsqueeze_right(self.alphas[timestep] ** 0.5, sample.ndim) * self.unsqueeze_right(beta_prod_t_prev, sample.ndim)) / \
                               self.unsqueeze_right(beta_prod_t, sample.ndim)

        # 5. Compute predicted previous sample (mu_t), which is the mean of the reverse step
        # mu_theta(x_t, t) = coeff_x0 * predicted_x0 + coeff_xt * x_t
        pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * sample

        # 6. Add noise to the predicted previous sample (sampling from N(mean, variance))
        # This noise accounts for the stochasticity in the reverse diffusion process
        variance = 0
        if timestep > 0: # No noise added at the final step (t=0 to x_0)
            noise = torch.randn(
                model_output.size(), # Use model_output size for consistency
                dtype=model_output.dtype,
                layout=model_output.layout,
                generator=generator,
                device=model_output.device,
            )
            # Get the fixed variance for the current timestep
            current_variance = self._get_variance(timestep) # No predicted_variance needed
            variance = self.unsqueeze_right(current_variance ** 0.5, sample.ndim) * noise

        pred_prev_sample = pred_prev_sample + variance

        return pred_prev_sample, pred_original_sample

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
