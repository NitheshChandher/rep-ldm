from PIL import Image
import torch
import numpy as np
from torch import autocast
from tqdm.auto import tqdm
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
    del vae
    return pil_imgs

def produce_latents(config, unet, seed=42, encoder_hidden_states=None, noise_scheduler=None, steps=None, device=None):
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
    if encoder_hidden_states is None:
        batch_size = 1
    else:
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
    num_inference_steps = 100
    scheduler = noise_scheduler
    scheduler.set_timesteps(num_inference_steps, device=device)

    # Denoising loop
    with autocast('cuda'):
        progress_bar = tqdm(range(0, num_inference_steps))
        progress_bar.set_description("Steps")
        for i, t in enumerate(scheduler.timesteps):
            # UNet expects (B, C, H, W), timestep, and encoder_hidden_states
            with torch.no_grad():
                if config["method"] == 'dino-ldm' or config["method"] == 'clip-ldm' or config["method"] == 'diffae':
                    noise_pred = unet(latents, t, encoder_hidden_states)['sample']    
                else:    
                    noise_pred = unet(latents, t)['sample']
            latents = scheduler.step(noise_pred, t, latents)['prev_sample']
            progress_bar.update(1)
    del unet, scheduler  
    return latents

def make_image_grid(images, grid_size=(2, 2), padding=10, bg_color=(255, 255, 255)):
    """
    Arrange 4 PIL images into a 2x2 grid.
    
    Args:
        images (list): List of 4 PIL.Image objects.
        grid_size (tuple): The grid layout (rows, cols), default is (2, 2).
        padding (int): Padding between images in pixels.
        bg_color (tuple): Background color (R, G, B).
        
    Returns:
        PIL.Image: A new image containing the grid.
    """
    assert len(images) == 4, "You must provide exactly 4 images."

    widths, heights = zip(*(img.size for img in images))
    max_width, max_height = max(widths), max(heights)
    
    resized_images = [img.resize((max_width, max_height)) for img in images]
    
    grid_rows, grid_cols = grid_size
    total_width = grid_cols * max_width + (grid_cols - 1) * padding
    total_height = grid_rows * max_height + (grid_rows - 1) * padding

    grid_image = Image.new('RGB', (total_width, total_height), color=bg_color)

    for idx, img in enumerate(resized_images):
        row = idx // grid_cols
        col = idx % grid_cols
        x = col * (max_width + padding)
        y = row * (max_height + padding)
        grid_image.paste(img, (x, y))

    return grid_image