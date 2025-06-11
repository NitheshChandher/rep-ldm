import torch
from torch import autocast
from tqdm.auto import tqdm
from diffusers import DDPMScheduler, AutoencoderKL
from timm.models.vision_transformer import VisionTransformer
import os
from PIL import Image
import numpy as np
import random
import pandas as pd

def vit_small(pretrained, progress, **kwargs):
    patch_size = kwargs.get("patch_size", 8)
    model = VisionTransformer(
        img_size=224, patch_size=patch_size, embed_dim=384, num_heads=6, num_classes=0
    )
    if pretrained:
        pretrained_url = f"https://github.com/lunit-io/benchmark-ssl-pathology/releases/download/pretrained-weights/dino_vit_small_patch8_ep200.torch"
        verbose = model.load_state_dict(
            torch.hub.load_state_dict_from_url(pretrained_url, progress=progress)
        )
        print(verbose)
    return model

def mean_attribute(directory_path, attr_file, attribute_name):
    # Load attribute file
    attr_df = pd.read_csv(attr_file, sep=",", header=0, index_col=0)
    
    if attribute_name not in attr_df.columns:
        raise ValueError(f"Attribute '{attribute_name}' not found in attribute file.")
    
    # Split dataset into two groups
    selected_files = attr_df[attr_df[attribute_name] == 1].index.tolist()
    remaining_files = attr_df[attr_df[attribute_name] != 1].index.tolist()  # Everything else (-1 or 0)
    
    selected_files = [file.replace('.jpg', '') for file in selected_files]
    remaining_files = [file.replace('.jpg', '') for file in remaining_files]

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

def concatenate_attribute_npy(directory_path, attr_file, attribute_name):
    attr_df = pd.read_csv(attr_file, sep=",", header=0, index_col=0)
    if attribute_name not in attr_df.columns:
        raise ValueError(f"Attribute '{attribute_name}' not found in attribute file.")

    selected_files = attr_df[attr_df[attribute_name] == 1].index.tolist()
    selected_files = [file.replace('.jpg', '') for file in selected_files]
    npy_files = [f"{file}.npy" for file in selected_files]

    arrays = []
    for file in npy_files:
        file_path = os.path.join(directory_path, file)
        if os.path.exists(file_path):
            data = np.load(file_path)
            arrays.append(data)

    if not arrays:
        raise ValueError(f"No valid .npy files found for attribute '{attribute_name}'.")

    return np.stack(arrays, axis=0)
 
def concatenate_npy_files(folder_path):
    arrays = []
    for file in sorted(f for f in os.listdir(folder_path) if f.endswith('.npy')):
        arr = np.load(os.path.join(folder_path, file))
        if arr.shape != (768,):
            raise ValueError(f"File {file} has wrong shape {arr.shape}")
        arrays.append(arr)
    return np.vstack(arrays)

def produce_images(args, encoder_hidden_states=None):

    if args.sampling == 'RandomSeed':
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
    
    else:
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        np.random.seed(42)
        random.seed(42)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #load pre-trained models
    unet = torch.load(args.model_path, map_location=device, weights_only=False)     #UNet trained using DINOv2 Representations                           
    scheduler = DDPMScheduler.from_pretrained('CompVis/stable-diffusion-v1-4', subfolder="scheduler")       #Noise Scheduler
    vae = AutoencoderKL.from_pretrained('CompVis/stable-diffusion-v1-4', subfolder="vae", revision=None)    #Pre-trained VAE
    unet = unet.to(device)

    #sampling latents from normal distribution
    if args.sampling == "Interpolation" or args.sampling == "MeanAttributeVector" or args.sampling == "MeanAttributeDifference":
        latents = torch.randn((encoder_hidden_states.shape[0],  4, args.res // 8, args.res // 8))
        
    else:
        latents = torch.randn((args.num_samples, 4, args.res // 8, args.res // 8))

    latents = latents.to(device)

    scheduler.set_timesteps(args.num_inference_steps)

    with autocast('cuda'):
        progress_bar = tqdm(range(0, args.num_inference_steps))
        progress_bar.set_description("Steps")
        for _, t in enumerate(scheduler.timesteps): #iterate over timesteps
            with torch.no_grad():
                noise_pred = unet(latents, t, encoder_hidden_states)['sample']  #predict noise added to latents at each timestep  
            latents = scheduler.step(noise_pred, t, latents)['prev_sample']
            progress_bar.update(1)
            
    del unet, scheduler
    
    latents = 1 / 0.18215 * latents # Rescale latents to be in the same range as the diffusion model

    #decode latents to generate images
    vae = vae.to(device)
    with torch.no_grad():
        decoder_output = vae.decode(latents)
        imgs = decoder_output.sample if hasattr(decoder_output, 'sample') else decoder_output
    imgs = (imgs / 2 + 0.5).clamp(0, 1)
    imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
    imgs = (imgs * 255).round().astype('uint8')
    pil_images = [Image.fromarray(image) for image in imgs]   
    return pil_images