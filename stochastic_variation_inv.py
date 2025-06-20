import numpy as np
import os
import argparse
import random
import torch
from diffusers import DDIMScheduler, AutoencoderKL
from dataset.custom import ImageDataset
from PIL import Image
from torch.utils.data import DataLoader
from utils.inversion_utils import inversion_forward_process, inversion_reverse_process
from torch import autocast, inference_mode
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device_num", type=int, default=0)
    parser.add_argument("--cfg_src", type=float, default=1.0)
    parser.add_argument("--cfg_tar", type=float, default=1.0)
    parser.add_argument("--num_diffusion_steps", type=int, default=100)
    parser.add_argument("--num_samples", type=int, default=4, help="Number of samples to generate")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset folder to load data")
    parser.add_argument("--output_path", type=str, required=True, help="Path to the output folder to store images")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--rep_path", type=str, default="rep/ffhq/", help="Path to the directory containing dinov2 representations vectors")
    parser.add_argument("--index", type=int, default=0, help="Index of the first image in the dataset")
    parser.add_argument("--res", type=int, default=512, help="Resolution of the generated images")
    parser.add_argument("--eta", type=float, default=1)
    parser.add_argument("--skip", type=int, default=36)
    
    args = parser.parse_args()

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    random.seed(42)
    model_id = "CompVis/stable-diffusion-v1-4"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(args.data_path):
        raise ValueError(f'{args.data_path} does not exist!')
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path, exist_ok=True)
    if not os.path.exists(args.rep_path):
        raise ValueError(f'{args.rep_path} does not exist!')
    if not os.path.exists(args.model_path):
        raise ValueError(f'{args.model_path} does not exist!')
      
    save_path = os.path.join(args.output_path, "stochastic_variation")

    transform = transforms.Compose([
                transforms.Resize(args.res),    # Resize images to 256x256
                transforms.ToTensor(),          # Convert images to PyTorch tensors
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])
   
    dataset = ImageDataset(image_dir=args.data_path, rep_dir=args.rep_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
    data_iter = iter(dataloader)
    batch = next(data_iter)

    images = batch[0]  
    real_imgs = images.to(device)
    latents = batch[1].unsqueeze(1).to(device)

    cfg_src = args.cfg_src
    eta = args.eta

    x0 = real_imgs[args.index,:,:,:].unsqueeze(0)
    z0 = latents[args.index,:,:].unsqueeze(0)

    scheduler = DDIMScheduler.from_config(model_id, subfolder="scheduler")
    scheduler.set_timesteps(args.num_diffusion_steps)
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", revision=None).to(device)

    with autocast("cuda"), inference_mode():
        # Encode the input image (x0) into latent space
        w0 = (vae.encode(x0).latent_dist.mode() * 0.18215).float()
        del vae
        # Load the UNet model
        unet = torch.load(args.model_path, map_location=device, weights_only=False).to(device)

        # Call the inversion forward process, passing scheduler as an argument
        wt, zs, wts = inversion_forward_process(unet, scheduler, w0, etas=eta,
                                                encoder_hidden_states=z0,
                                                cfg_scale=cfg_src,
                                                prog_bar=True,
                                                num_inference_steps=args.num_diffusion_steps)

        torch.cuda.empty_cache()   

        w0, _ = inversion_reverse_process(unet, scheduler, xT=wts[args.num_diffusion_steps - args.skip],
                                        etas=eta, encoder_hidden_states=z0,
                                        cfg_scales=[args.cfg_tar], prog_bar=True,
                                        zs=zs[:(args.num_diffusion_steps - args.skip)])
        print(f"Generated Reconstructed Image!")
        w0_ls = []
        w0_ls.append(w0)

        for i in range(args.num_samples):
            xT = torch.randn_like(wts[args.num_diffusion_steps - args.skip])
            zs1 = torch.randn_like(zs[:(args.num_diffusion_steps - args.skip)])
            w0, _ = inversion_reverse_process(unet, scheduler, xT=wts[args.num_diffusion_steps - args.skip],
                                        etas=eta, encoder_hidden_states=z0,
                                        cfg_scales=[args.cfg_tar], prog_bar=True,
                                        zs=zs1)
                                        #zs=zs[:(args.num_diffusion_steps - args.skip)])
            w0_ls.append(w0)
            print(f"Generated Sample {i}!")

    del unet, scheduler
    imgs=[]
    vae = AutoencoderKL.from_pretrained('CompVis/stable-diffusion-v1-4', subfolder="vae", revision=None).to(device)
    for i in range(len(w0_ls)):
        with autocast("cuda"), inference_mode():
            x0_dec = vae.decode(1 / 0.18215 * w0_ls[i]).sample
        img = (x0_dec / 2 + 0.5).clamp(0, 1)
        img = img.detach().cpu().permute(0, 2, 3, 1).numpy()
        img = (img * 255).round().astype('uint8')
        imgs.append(Image.fromarray(img[0]))

    x0 = (x0 / 2 + 0.5).clamp(0, 1)
    x0 = x0.detach().cpu().permute(0, 2, 3, 1).numpy()
    x0 = (x0 * 255).round().astype('uint8')
    x0 = Image.fromarray(x0[0])
    
    fig, axes = plt.subplots(1, len(imgs)+1, figsize=(18, 4))  # 6 = 1 original + 5 samples

    # Plot original
    axes[0].imshow(x0)
    axes[0].set_title("Input")
    axes[0].axis("off")

    axes[1].imshow(imgs[0])
    axes[1].set_title("Reconstruction")
    axes[1].axis("off")

    for i in range(len(imgs)-1):
        axes[i + 2].imshow(imgs[i+1])
        axes[i + 2].set_title(f"Sample {i + 1}")
        axes[i + 2].axis("off")
    plt.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(f'{save_path}/compare_dino-ldm_{args.index}.png', dpi=100, bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close(fig)

if __name__ == "__main__":
    main()