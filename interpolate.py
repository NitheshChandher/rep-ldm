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

def cos(a, b):
    a = a.view(-1)
    b = b.view(-1)
    a = torch.nn.functional.normalize(a, dim=0)
    b = torch.nn.functional.normalize(b, dim=0)
    return (a * b).sum()

# Spherical linear interpolation between matrices
# See Eq. 67 in the DDIM paper (https://arxiv.org/pdf/2010.02502)
# See also diffusion AE example: https://github.com/phizaz/diffae/blob/master/interpolate.ipynb
def slerp(x0,x1,alpha):
    theta = torch.arccos(cos(x0, x1))
    x_shape = x0.shape
    x_interp = (torch.sin((1 - alpha) * theta) * x0.flatten() + torch.sin(alpha * theta) * x1.flatten()) / torch.sin(theta)
    x_interp = x_interp.view(*x_shape)

    return x_interp

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device_num", type=int, default=0)
    parser.add_argument("--cfg_src", type=float, default=1.0)
    parser.add_argument("--cfg_tar", type=float, default=1.0)
    parser.add_argument("--num_diffusion_steps", type=int, default=100)
    parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset folder to load data")
    parser.add_argument("--output_path", type=str, required=True, help="Path to the output folder to store images")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--rep_path", type=str, default="rep/ffhq/", help="Path to the directory containing dinov2 representations vectors")
    parser.add_argument("--sampling", type=str, default="linear-spherical", choices=["linear", "spherical", "linear-spherical"], help="Interpolate: 1) only rep, 2) only noise maps, or 3) both")
    parser.add_argument("--index1", type=int, default=0, help="Index of the first image in the dataset")
    parser.add_argument("--index2", type=int, default=1, help="Index of the second image in the dataset")
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
    if args.sampling not in ["linear", "spherical", "linear-spherical"]:
        raise ValueError(f'Invalid sampling method: {args.sampling}. Choose from "linear", "spherical", or "linear-spherical".')
      
    save_path = os.path.join(args.output_path, args.sampling)

    transform = transforms.Compose([
                transforms.Resize(args.res),    # Resize images to 256x256
                transforms.ToTensor(),          # Convert images to PyTorch tensors
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])
   
    dataset = ImageDataset(image_dir=args.data_path, rep_dir=args.rep_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
    data_iter = iter(dataloader)
    batch = next(data_iter)

    images = batch[0]  
    real_imgs = images.to(device)
    latents = batch[1].unsqueeze(1).to(device)

    cfg_src = args.cfg_src
    eta = args.eta

    x0 = real_imgs[args.index1,:,:,:].unsqueeze(0)
    x1 = real_imgs[args.index2,:,:,:].unsqueeze(0)
    z0 = latents[args.index1,:,:].unsqueeze(0)
    z1 = latents[args.index2,:,:].unsqueeze(0)
    inp = torch.cat([x0, x1], dim=0)
    z_pair = torch.cat([z0, z1], dim=0)

    scheduler = DDIMScheduler.from_config(model_id, subfolder="scheduler")
    scheduler.set_timesteps(args.num_diffusion_steps)
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", revision=None).to(device)

    with autocast("cuda"), inference_mode():
        # Encode the input image (x0) into latent space
        w0 = (vae.encode(inp).latent_dist.mode() * 0.18215).float()
        del vae
        # Load the UNet model
        unet = torch.load(args.model_path, map_location=device, weights_only=False).to(device)

        # Call the inversion forward process, passing scheduler as an argument
        wt, zs, wts = inversion_forward_process(unet, scheduler, w0, etas=eta,
                                                encoder_hidden_states=z_pair,
                                                cfg_scale=cfg_src,
                                                prog_bar=True,
                                                num_inference_steps=args.num_diffusion_steps)

    torch.cuda.empty_cache()   
    K = 9
    alpha = np.linspace(0,1,K)

    z_interp = torch.zeros([K,1,latents.shape[2]], device=device)
    zs_interp = torch.zeros([zs.shape[0],K,zs.shape[2],zs.shape[3],zs.shape[4]], device=device)
    wts_interp = torch.zeros([K,wts.shape[2],wts.shape[3],wts.shape[4]], device=device)

    if args.sampling == "linear-spherical":
    # Interpolation for each alpha value
        for k in range(K):
            # Linear interpolation of latents
            z_interp[k,:,:] = (1-alpha[k])*z0 + alpha[k]*z1

            # Spherical interpolation of noise maps, for each timestep
            for t in range(zs.shape[0]):
                zs_interp[t,k,:,:,:] = slerp(zs[t,0,:,:,:], zs[t,1,:,:,:], alpha[k])
                #zs_interp[t,k,:,:,:] = (1-alpha[k])*zs[t,0,:,:,:] + alpha[k]*zs[t,1,:,:,:] # linear interpolation

            # Spherical interpolation of initial image
            wts_interp[k,:,:,:] = slerp(wts[args.num_diffusion_steps-args.skip,0,:,:,:], wts[args.num_diffusion_steps-args.skip,1,:,:,:], alpha[k])
            #wts_interp[k,:,:,:] = (1-alpha[k])*wts[num_diffusion_steps-skip,0,:,:,:] + alpha[k]*wts[num_diffusion_steps-skip,1,:,:,:] # linear interpolation

    elif args.sampling == "linear":
        # Interpolation for each alpha value
        for k in range(K):
            # Linear interpolation of latents
            z_interp[k,:,:] = (1-alpha[k])*z0 + alpha[k]*z1

        # Use the first noise map for all alphas
        for t in range(zs.shape[0]):
            zs_interp[t,:,:,:] = zs[t,0,:,:,:]
        wts_interp = wts[args.num_diffusion_steps-args.skip,0,:,:,:]

    elif args.sampling == "spherical":
        # Interpolation for each alpha value
        for k in range(K):
            # Use the first latent for all alphas
            z_interp[k,:,:] = z0

            # Spherical interpolation of noise maps, for each timestep
            for t in range(zs.shape[0]):
                zs_interp[t,k,:,:,:] = slerp(zs[t,0,:,:,:], zs[t,1,:,:,:], alpha[k])

            # Spherical interpolation of initial image
            wts_interp[k,:,:,:] = slerp(wts[args.num_diffusion_steps-args.skip,0,:,:,:], wts[args.num_diffusion_steps-args.skip,1,:,:,:], alpha[k])

    else:
        raise ValueError(f'Invalid sampling method: {args.sampling}. Choose from "linear", "spherical", or "linear-spherical".')
    
    with autocast("cuda"), inference_mode():
        w0, _ = inversion_reverse_process(unet, scheduler, xT=wts_interp,
                                        etas=eta, encoder_hidden_states=z_interp,
                                        cfg_scales=[args.cfg_tar], prog_bar=True,
                                        zs=zs_interp[:(args.num_diffusion_steps - args.skip)])
    
    del unet, scheduler
    vae = AutoencoderKL.from_pretrained('CompVis/stable-diffusion-v1-4', subfolder="vae", revision=None).to(device)
    with autocast("cuda"), inference_mode():
        x0_dec = vae.decode(1 / 0.18215 * w0).sample
    del vae
   
    imgs = (x0_dec / 2 + 0.5).clamp(0, 1)
    imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
    imgs = (imgs * 255).round().astype('uint8')
    pil_outs = [Image.fromarray(image) for image in imgs] 
    os.makedirs(save_path, exist_ok=True)

    real_imgs = (real_imgs / 2 + 0.5).clamp(0, 1)
    real_imgs = real_imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
    real_imgs = (real_imgs * 255).round().astype('uint8')
    real_imgs = [Image.fromarray(image) for image in real_imgs]
    K = len(pil_outs)

    fig, ax = plt.subplots(1, 9, figsize=(5*9, 5))
    for i in range(len(alpha)):
        ax[i].imshow(pil_outs[i])
    plt.savefig('compare-512.png')
"""
    # Plot
    plt.figure(figsize=[2.5*K, 5])

    plt.subplot(2, K, 1)
    plt.imshow(np.array(real_imgs[args.index1]))
    plt.title('$I_1$')
    plt.axis('off')

    plt.subplot(2, K, K)
    plt.imshow(np.array(real_imgs[args.index2]))
    plt.title('$I_2$')
    plt.axis('off')

    for k in range(K):
        plt.subplot(2, K, K + k + 1)
        plt.imshow(pil_outs[k])
        plt.title(r'$\alpha = %0.2f$'%alpha[k], y=-0.15)
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(save_path + f'/{args.sampling}_indices_{args.index1}_{args.index2}.png', dpi=300, bbox_inches='tight')
    plt.close()
"""
    
if __name__ == "__main__":
    main()