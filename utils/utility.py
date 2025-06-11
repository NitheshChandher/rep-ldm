from PIL import Image
import torch
import numpy as np
from diffusers import AutoencoderKL, VQModel
from diffusers import DDPMScheduler

def decode_img_latents(latents, config, vae=None):
    """
    Decodes a batch of latent vectors into PIL images using the VAE decoder.

    Args:
        latents (torch.Tensor): Latent tensors of shape (B, C, H, W).
        config (dict): Configuration dictionary (must contain 'encoder_name').
        vae (AutoencoderKL or VQModel, optional): Preloaded VAE. If None, will load as per config.

    Returns:
        List[Image.Image]: List of decoded PIL images.
    """
    # Load VAE if not provided
    if vae is None:
        vae_path = config["pretrained_model_name_or_path"]
        if config.get("encoder_name", "KL") == "KL":
            vae = AutoencoderKL.from_pretrained(vae_path, subfolder="vae").to(latents.device)
        else:
            vae = VQModel.from_pretrained(vae_path, subfolder="vae").to(latents.device)
        vae.eval()

    with torch.no_grad():
        # Scale back if needed (Stable Diffusion convention)
        latents = latents / 0.18215
        imgs = vae.decode(latents).sample
        imgs = (imgs.clamp(-1, 1) + 1) / 2  # [-1,1] to [0,1]
        imgs = imgs.cpu().permute(0, 2, 3, 1).numpy()  # (B,H,W,C)
        imgs = (imgs * 255).round().astype(np.uint8)
        pil_imgs = [Image.fromarray(img) for img in imgs]
    return pil_imgs

import torch

def produce_latents(config, encoder_hidden_states, unet, seed=42, noise_scheduler=None, steps=None, device=None):
    """
    Generates denoised latents from random noise, conditioned on encoder_hidden_states, using the UNet and DDPM scheduler.

    Args:
        config (dict): Configuration dictionary.
        encoder_hidden_states (torch.Tensor): Conditioning tensor (B, 1, D).
        unet (UNet2DConditionModel): Trained UNet model.
        seed (int): Random seed for reproducibility.
        noise_scheduler (DDPMScheduler, optional): If None, loads from config.
        steps (int, optional): Number of denoising steps. If None, uses scheduler default.
        device (torch.device, optional): Device to run on.

    Returns:
        torch.Tensor: Denoised latents of shape (B, C, H, W).
    """
    torch.manual_seed(seed)
    batch_size = encoder_hidden_states.shape[0]
    latent_shape = (
        batch_size,
        config["channels"],
        config["resolution"] // 8,
        config["resolution"] // 8,
    )
    device = device or next(unet.parameters()).device

    # Load scheduler if not provided
    if noise_scheduler is None:
        noise_scheduler = DDPMScheduler.from_pretrained(
            config["pretrained_model_name_or_path"], subfolder="scheduler"
        )

    # Start from pure noise
    latents = torch.randn(latent_shape, device=device)

    # Set number of inference steps
    num_inference_steps = steps or noise_scheduler.config.num_train_timesteps
    scheduler = noise_scheduler
    scheduler.set_timesteps(num_inference_steps, device=device)

    # Denoising loop
    for t in scheduler.timesteps:
        # UNet expects (B, C, H, W), timestep, and encoder_hidden_states
        with torch.no_grad():
            model_pred = unet(latents, t, encoder_hidden_states).sample

        if scheduler.config.prediction_type == "epsilon":
            latents = scheduler.step(model_pred, t, latents).prev_sample
        elif scheduler.config.prediction_type == "v_prediction":
            v = model_pred
            latents = scheduler.step(v, t, latents).prev_sample
        else:
            raise ValueError(f"Unknown prediction type {scheduler.config.prediction_type}")

    return latents
