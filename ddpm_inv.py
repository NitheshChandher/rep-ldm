import argparse
import numpy as np
import random
import torch
from diffusers import DDIMScheduler, AutoencoderKL
import os
from PIL import Image
from dataset import load_and_prepare_dataset
from sklearn.covariance import EmpiricalCovariance
from utils.inversion_utils import inversion_forward_process, inversion_reverse_process
from utils.attribute import mean_attribute, concatenate_npy_files, concatenate_attribute_npy
from torch import autocast, inference_mode
import matplotlib.pyplot as plt
import calendar
import time
from skimage.exposure import match_histograms
import matplotlib
matplotlib.use('Agg')  # Use a consistent non-GUI backend


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device_num", type=int, default=0)
    parser.add_argument("--cfg_src", type=float, default=1.0)
    parser.add_argument("--cfg_tar", type=float, default=2.0)
    parser.add_argument("--num_diffusion_steps", type=int, default=100)
    parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset folder to load data")
    parser.add_argument("--output_path", type=str, required=True, help="Path to the output folder to store images")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--sampling", type=str, default="RandomSeed", help="Sampling method (RandomSeed or NoisePerturbation or Interpolation or MeanAttributeVector or MeanAttributeDifference or PCA)")
    parser.add_argument("--lamda", type=float, default=1.0, help="Noise perturbation strength")
    parser.add_argument("--dataset", type=str, default="celeb-a", help="Name of the dataset (stl10 or cifar-10)")
    parser.add_argument("--num_samples", type=int, default=8, help="Number of samples to generate")
    parser.add_argument("--attribute1", type=str, default="Blond_Hair", help="Attribute name for MeanAttributeVector sampling")
    parser.add_argument("--attribute_file", type=str, default="rep/celeba/list_attr_celeba.csv", help="Path to the attribute file")
    parser.add_argument("--rep_dir", type=str, default="rep/celeba/", help="Path to the directory containing dinov2 representations vectors")
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

    current_GMT = time.gmtime()
    time_stamp = calendar.timegm(current_GMT)

    if not os.path.exists(args.data_path):
        raise ValueError(f'{args.data_path} does not exist!')
    
    save_path = os.path.join(args.output_path, "inv_results")
    
    if os.path.exists(save_path):
        print(f'{save_path} exist!')
    
    else:
        print(f"Creating a directory at {save_path}!")
        os.makedirs(save_path, exist_ok=True)
    
    train_dataloader,_ = load_and_prepare_dataset(dataset_name="ffhq", batch_size=2, 
                                                  img_size=(args.res, args.res),data_dir="test", rep_dir="rep/test", shuffle=False)

    #train_dataloader,_ = load_and_prepare_dataset(dataset_name=args.dataset, batch_size=args.num_samples, 
    #                                              img_size=(args.res, args.res),data_dir=args.data_path, rep_dir=args.rep_dir, shuffle=False)
    data_iter = iter(train_dataloader)
    batch = next(data_iter)

    # Unpacking the batch
    images = batch[0] 
    real_imgs = images.to(device) 
    x0 = images.to(device)

    if args.sampling == 'RandomSeed':
        seed = args.seed
        encoder_hidden_states = batch[1].unsqueeze(1).to(device)
        save_path = os.path.join(save_path, f"seed/{seed}")
        
    elif args.sampling == 'NoisePerturbation':
        encoder_hidden_states = batch[1].unsqueeze(1).to(device)
        noise = torch.randn(encoder_hidden_states.shape, dtype=encoder_hidden_states.dtype,
                                device=encoder_hidden_states.device) * args.lamda
        encoder_hidden_states = encoder_hidden_states + noise
        save_path = os.path.join(save_path, f"noise-perturbation/{args.lamda}")
    
    elif args.sampling == 'MeanAttributeVector':
        attribute1 = mean_attribute(directory_path=args.rep_dir, attr_file=args.attribute_file, attribute_name=args.attribute1).to(device)
        encoder_hidden_states = batch[1].unsqueeze(1).to(device)
        attribute1 = attribute1.unsqueeze(0).unsqueeze(1)
        encoder_hidden_states = encoder_hidden_states + args.lamda * attribute1
        save_path = os.path.join(save_path, f"mean-attribute-difference/{args.attribute1}/{args.lamda}")

    elif args.sampling == 'PCA':
        dino_reps = concatenate_npy_files(args.rep_dir)
        print(f'Shape of the representations: {dino_reps.shape}')
        ec = EmpiricalCovariance(assume_centered=True)
        ec.fit(dino_reps)
        eig_vals, eigen_vectors = np.linalg.eig(ec.covariance_)

        explained_var = np.cumsum(eig_vals) / np.sum(eig_vals)
        n_components = np.argmax(explained_var >= 0.50) + 1
        print(f"Number of components to explain 50% variance: {n_components}")

        sorted_indices = np.argsort(eig_vals)[::-1]
        sorted_eigenvectors = eigen_vectors[:, sorted_indices]

        top_eigenvectors = sorted_eigenvectors[:, :n_components]
        top_eigenvectors = top_eigenvectors.reshape(n_components,768).astype(np.float32)
        top_eigenvectors = torch.from_numpy(top_eigenvectors).to(device)
        encoder_hidden_states = batch[1].unsqueeze(1).to(device)
        encoder_hidden_states = encoder_hidden_states + args.lamda * top_eigenvectors[5]
        save_path = os.path.join(save_path, f"pca/strength_{args.lamda}_component_{6}")

    elif args.sampling == 'PCA-attribute':
        dino_reps = concatenate_attribute_npy(directory_path=args.rep_dir, attr_file=args.attribute_file, attribute_name=args.attribute1)
        print(f'Shape of the representations: {dino_reps.shape}')
        ec = EmpiricalCovariance(assume_centered=True)
        ec.fit(dino_reps)
        eig_vals, eigen_vectors = np.linalg.eig(ec.covariance_)

        explained_var = np.cumsum(eig_vals) / np.sum(eig_vals)
        n_components = np.argmax(explained_var >= 0.50) + 1
        print(f"Number of components to explain 50% variance: {n_components}")

        sorted_indices = np.argsort(eig_vals)[::-1]
        sorted_eigenvectors = eigen_vectors[:, sorted_indices]

        top_eigenvectors = sorted_eigenvectors[:, :n_components]
        top_eigenvectors = top_eigenvectors.reshape(n_components,768).astype(np.float32)
        top_eigenvectors = torch.from_numpy(top_eigenvectors).to(device)

        encoder_hidden_states = batch[1].unsqueeze(1).to(device)
        encoder_hidden_states = encoder_hidden_states + args.lamda * top_eigenvectors[0]
        save_path = os.path.join(save_path, f"pca/{args.attribute1}/{args.lamda}")

    else:
        raise ValueError(f"Invalid sampling method: {args.sampling}. Choose from 'RandomSeed', 'NoisePerturbation', 'MeanAttributeVector', 'MeanAttributeDifference', 'PCA' or 'PCA-attribute'.")
    

    # Define the scheduler only once before calling the function
    scheduler = DDIMScheduler.from_config(model_id, subfolder="scheduler")
    scheduler.set_timesteps(args.num_diffusion_steps)

    # Load the VAE model
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", revision=None).to(device)

    with autocast("cuda"), inference_mode():
        # Load the UNet model
        unet = torch.load(args.model_path, map_location=device, weights_only=False).to(device)
        unet.eval()

        w0 = (vae.encode(x0).latent_dist.mode() * 0.18215).float()
        del vae  # Free the VAE model after encoding
        wt, zs, wts = inversion_forward_process(unet, scheduler, w0, etas=args.eta, 
                                                encoder_hidden_states=encoder_hidden_states, 
                                                cfg_scale=args.cfg_src, 
                                                prog_bar=True, 
                                                num_inference_steps=args.num_diffusion_steps)
                
        w0, _ = inversion_reverse_process(unet, scheduler, xT=wts[args.num_diffusion_steps - args.skip],
                                            etas=args.eta, encoder_hidden_states=encoder_hidden_states, 
                                            cfg_scales=[args.cfg_tar], prog_bar=True, 
                                            zs=zs[:(args.num_diffusion_steps - args.skip)])
        
        del unet, scheduler

    vae = AutoencoderKL.from_pretrained('CompVis/stable-diffusion-v1-4', subfolder="vae", revision=None).to(device)
    with autocast("cuda"), inference_mode():
        x0_dec = vae.decode(1 / 0.18215 * w0).sample
    del vae
    imgs = (x0_dec / 2 + 0.5).clamp(0, 1)
    imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
    imgs = (imgs * 255).round().astype('uint8')
    pil_images = [Image.fromarray(image) for image in imgs] 
    os.makedirs(save_path, exist_ok=True)

    real_imgs = (real_imgs / 2 + 0.5).clamp(0, 1)
    real_imgs = real_imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
    real_imgs = (real_imgs * 255).round().astype('uint8')
    real_imgs = [Image.fromarray(image) for image in real_imgs]

    
    for i in range(len(pil_images)):
        matched = match_histograms(
                                np.array(pil_images[i]),
                                np.array(real_imgs[i]),
                                channel_axis=-1)
        matched_pil = Image.fromarray(matched)           
        fig, ax = plt.subplots(1, 2, figsize=(7, 5), dpi=100)
        ax[0].imshow(real_imgs[i])
        ax[1].imshow(matched_pil)
        plt.tight_layout()
        plt.savefig(f'{save_path}/compare_dino-ldm_{args.attribute1}_{i}_(hist_match).png', dpi=100, bbox_inches='tight', pad_inches=0, transparent=True)
        plt.close(fig)
    
        fig, ax = plt.subplots(1, 2, figsize=(7, 5), dpi=100)
        ax[0].imshow(real_imgs[i])
        ax[1].imshow(pil_images[i])
        plt.tight_layout()
        plt.savefig(f'{save_path}/compare_dino-ldm_{args.attribute1}_{i}.png', dpi=100, bbox_inches='tight', pad_inches=0, transparent=True)
        plt.close(fig)   

    """
    for i, img in enumerate(pil_images):
        image_name = f'cfg_d_{args.cfg_tar}_skip_{args.skip}_{i}.png'
        img.save(os.path.join(save_path, image_name))
    """

if __name__ == '__main__':
    main()