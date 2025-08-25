import argparse
import os
import random
import time
import calendar

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import autocast, inference_mode
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image
from diffusers import DDIMScheduler, AutoencoderKL

from dataset.custom import ImageDataset
from utils.inversion_utils import inversion_forward_process, inversion_reverse_process

matplotlib.use("Agg")  # Non-GUI backend for saving figures


def main():
    # -----------------------
    # Argument Parser
    # -----------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--device_num", type=int, default=0)
    parser.add_argument("--cfg_src", type=float, default=1.0)
    parser.add_argument("--cfg_tar", type=float, default=2.0)
    parser.add_argument("--num_diffusion_steps", type=int, default=100)
    parser.add_argument("--data_path", type=str, required=True, help="Dataset folder path")
    parser.add_argument("--output_path", type=str, required=True, help="Output folder path")
    parser.add_argument("--model_path", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--lamda", type=float, default=1.0, help="Noise perturbation strength")
    parser.add_argument("--num_samples", type=int, default=8, help="Number of samples to generate")
    parser.add_argument("--attribute1", type=str, default="Blond_Hair", help="Attribute name for saving")
    parser.add_argument("--attribute_file", type=str, default="annotations/list_attr_celebahq.csv")
    parser.add_argument("--rep_path", type=str, default="rep/ffhq/")
    parser.add_argument("--res", type=int, default=256, help="Image resolution")
    parser.add_argument("--eta", type=float, default=1)
    parser.add_argument("--skip", type=int, default=36)
    parser.add_argument("--index", type=int, default=0, help="Index of the sample to process")
    args = parser.parse_args()

    # -----------------------
    # Reproducibility
    # -----------------------
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    random.seed(42)

    model_id = "CompVis/stable-diffusion-v1-4"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -----------------------
    # Output Directory
    # -----------------------
    save_path = os.path.join(args.output_path, "inv_results")
    os.makedirs(save_path, exist_ok=True)

    # -----------------------
    # Dataset & Loader
    # -----------------------
    transform = transforms.Compose([
        transforms.Resize(args.res),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    dataset = ImageDataset(image_dir=args.data_path, rep_dir=args.rep_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)

    images, latents = next(iter(dataloader))
    real_imgs = images.to(device)
    latents = latents.unsqueeze(1).to(device)

    x0 = real_imgs[args.index].unsqueeze(0)
    z0 = latents[args.index].unsqueeze(0)

    # -----------------------
    # Scheduler & Models
    # -----------------------
    scheduler = DDIMScheduler.from_config(model_id, subfolder="scheduler")
    scheduler.set_timesteps(args.num_diffusion_steps)

    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae").to(device)

    with autocast("cuda"), inference_mode():
        unet = torch.load(args.model_path, map_location=device).to(device)
        unet.eval()

        # Encode with VAE
        w0 = (vae.encode(x0).latent_dist.mode() * 0.18215).float()
        del vae

        # Forward + Reverse processes
        _, zs, wts = inversion_forward_process(
            unet, scheduler, w0,
            etas=args.eta,
            encoder_hidden_states=z0,
            cfg_scale=args.cfg_src,
            prog_bar=True,
            num_inference_steps=args.num_diffusion_steps,
        )

        torch.cuda.empty_cache()

        w0, _ = inversion_reverse_process(
            unet, scheduler,
            xT=wts[args.num_diffusion_steps - args.skip],
            etas=args.eta,
            encoder_hidden_states=z0,
            cfg_scales=[args.cfg_tar],
            prog_bar=True,
            zs=zs[: (args.num_diffusion_steps - args.skip)],
        )

        del unet, scheduler

    # -----------------------
    # Decode Results
    # -----------------------
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae").to(device)
    with autocast("cuda"), inference_mode():
        x0_dec = vae.decode(1 / 0.18215 * w0).sample
    del vae

    # Convert to PIL
    def tensor_to_pil(tensor):
        arr = (tensor / 2 + 0.5).clamp(0, 1)
        arr = arr.detach().cpu().permute(0, 2, 3, 1).numpy()
        arr = (arr * 255).round().astype("uint8")
        return [Image.fromarray(im) for im in arr]

    recon_imgs = tensor_to_pil(x0_dec)
    real_imgs_pil = tensor_to_pil(real_imgs)

    # -----------------------
    # Save Side-by-Side Comparison
    # -----------------------
    for i, img in enumerate(recon_imgs):
        fig, ax = plt.subplots(1, 2, figsize=(7, 5), dpi=100)
        ax[0].imshow(real_imgs_pil[i])
        ax[0].set_title("Real")
        ax[0].axis("off")
        ax[1].imshow(img)
        ax[1].set_title("Reconstructed")
        ax[1].axis("off")
        plt.tight_layout()
        plt.savefig(
            f"{save_path}/compare_{args.attribute1}_{i}.png",
            dpi=100, bbox_inches="tight", pad_inches=0, transparent=True
        )
        plt.close(fig)


if __name__ == "__main__":
    main()