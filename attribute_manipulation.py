import numpy as np
import os
import argparse
import random
import torch
from diffusers import DDIMScheduler, AutoencoderKL
from PIL import Image
from torch.utils.data import DataLoader
from dataset.custom import ImageDataset
from utils.inversion_utils import inversion_forward_process, inversion_reverse_process 
from torch import autocast, inference_mode
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import pandas as pd

def attribute_manipulation(directory_path, attr_file, attribute_name):
    # Load attribute file
    attr_df = pd.read_csv(attr_file, sep=",", header=0, index_col=0)
    
    if attribute_name not in attr_df.columns:
        raise ValueError(f"Attribute '{attribute_name}' not found in attribute file.")
    
    # Split dataset into two groups
    selected_files = attr_df[attr_df[attribute_name] == 1].index.tolist()
    remaining_files = attr_df[attr_df[attribute_name] != 1].index.tolist()  # Everything else (-1 or 0)
    
    selected_files = [file.replace('.jpg', '') for file in selected_files]
    remaining_files = [file.replace('.jpg', '') for file in remaining_files]

    selected_files = [file.replace('.png', '') for file in selected_files]
    remaining_files = [file.replace('.png', '') for file in remaining_files]

    npy_selected = [f"{file}.npy" for file in selected_files]
    npy_remaining = [f"{file}.npy" for file in remaining_files]

    # Load vectors
    selected_arrays = [np.load(os.path.join(directory_path, file)) for file in npy_selected if os.path.exists(os.path.join(directory_path, file))]
    remaining_arrays = [np.load(os.path.join(directory_path, file)) for file in npy_remaining if os.path.exists(os.path.join(directory_path, file))]

    # Ensure valid data exists
    if not selected_arrays:
        raise ValueError(f"No valid .npy files found for attribute '{attribute_name}' (positive class).")
    if not remaining_arrays:
        raise ValueError(f"No valid .npy files found for remaining data (negative class).")

    # Compute mean vectors
    mean_selected = np.mean(np.stack(selected_arrays, axis=0), axis=0)
    mean_remaining = np.mean(np.stack(remaining_arrays, axis=0), axis=0)

    # Compute difference
    mean_diff = mean_selected - mean_remaining

    return torch.tensor(mean_diff, dtype=torch.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device_num", type=int, default=0)
    parser.add_argument("--cfg_src", type=float, default=1.0)
    parser.add_argument("--cfg_tar", type=float, default=1.0)
    parser.add_argument("--num_diffusion_steps", type=int, default=100)
    parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset folder to load data")
    parser.add_argument("--output_path", type=str, required=True, help="Path to the output folder to store images")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--rep_path", type=str, default="rep/ffhq/", help="Path to the directory containing dinov2/unclip/diffae representations vectors")
    parser.add_argument("--rep_attr_path", type=str, default="annotations/list_attr_celeba.csv", help="Path to the attribute file")
    parser.add_argument("--attribute", type=str, default="Blond_Hair", help="Attribute to manipulate (Check annotations/list_attr_celebahq.csv)")
    parser.add_argument("--index", type=int, default=0, help="Index of the first image in the dataset")
    parser.add_argument("--lamda", type=float, default=1.0, help="Weight for the attribute manipulation")
    parser.add_argument("--res", type=int, default=256, help="Resolution of the generated images")
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
      
    save_path = os.path.join(args.output_path, "attribute_manipulation", args.attribute)

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
           
        # Perform attribute manipulation
        attribute_emb = attribute_manipulation(directory_path=args.rep_path, attr_file=args.rep_attr_path, attribute_name=args.attribute).to(device)
        attribute_emb = attribute_emb.unsqueeze(0).unsqueeze(1)
        z0 = z0 + args.lamda * attribute_emb

        w0, _ = inversion_reverse_process(unet, scheduler, xT=wts[args.num_diffusion_steps - args.skip],
                                        etas=eta, encoder_hidden_states=z0,
                                        cfg_scales=[args.cfg_tar], prog_bar=True,
                                        zs=zs[:(args.num_diffusion_steps - args.skip)])

    del unet, scheduler
    vae = AutoencoderKL.from_pretrained('CompVis/stable-diffusion-v1-4', subfolder="vae", revision=None).to(device)
    with autocast("cuda"), inference_mode():
        x0_dec = vae.decode(1 / 0.18215 * w0).sample
    del vae
    
    img = (x0_dec / 2 + 0.5).clamp(0, 1)
    img = img.detach().cpu().permute(0, 2, 3, 1).numpy()
    img = (img * 255).round().astype('uint8')
    pil_image = Image.fromarray(img[0]) 
    os.makedirs(save_path, exist_ok=True)

    x0 = (x0 / 2 + 0.5).clamp(0, 1)
    x0 = x0.detach().cpu().permute(0, 2, 3, 1).numpy()
    x0 = (x0 * 255).round().astype('uint8')
    x0 = Image.fromarray(x0[0])

    fig, ax = plt.subplots(1, 2, figsize=(7, 5), dpi=100)
    ax[0].imshow(x0)
    ax[0].set_title("Input")
    ax[1].imshow(pil_image)
    ax[1].set_title(f"{args.attribute}")
    plt.tight_layout()
    plt.savefig(f'{save_path}/compare_dino-ldm_{args.index}', dpi=100, bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close(fig)

if __name__ == "__main__":
    main()