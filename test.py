import argparse
import os
import torch
from torchvision.utils import save_image
from dataset.dataloader import load_and_prepare_dataset
from diffusers import DDPMScheduler, DDIMScheduler, AutoencoderKL
from transformers import ViTModel
from torch import autocast
from tqdm.auto import tqdm
from PIL import Image
import numpy as np
import random
import torch_fidelity
import pandas as pd

def torch_metrics(args, gen_path, alpha=None):
    model = args.model
    method = args.method
    dataset = args.dataset
    seed = args.seed
    filename = "results.csv"
    real_path = args.test_dir

    # Calculate metrics
    metrics_dict = torch_fidelity.calculate_metrics(
        input1=gen_path, 
        input2=real_path, 
        cuda=True, 
        isc=True, 
        fid=True, 
        kid=True, 
        prc=True, 
        verbose=False,
    )

    metrics_dict['model'] = model
    metrics_dict['method'] = method
    metrics_dict['dataset'] = dataset

    if args.method == 'seed-dataset':
        metrics_dict['seed'] = seed
    else:
        metrics_dict['seed'] = 'N/A'

    if args.method == 'perturbate-dataset':
        metrics_dict['noise-strength'] = alpha
    else:
        metrics_dict['noise-strength'] = 'N/A'

    metrics_dict['interpolation'] = 'N/A'
    metrics_dict['num_inference_steps'] = args.num_inference_steps
    metrics_dict['scheduler'] = args.scheduler
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame([metrics_dict])

    if os.path.exists(filename):
        # Read existing CSV
        existing_df = pd.read_csv(filename)

        # Check if this exact row already exists
        duplicate = (existing_df == metrics_df.iloc[0]).all(axis=1).any()

        if duplicate:
            print("This row already exists in the CSV. Skipping append.")
        else:
            # Append and save
            updated_df = pd.concat([existing_df, metrics_df], ignore_index=True)
            updated_df.to_csv(filename, index=False)
            print(f"Evaluation of {args.method} for {args.model} trained on {args.dataset} added!")

    else:
        # Create new CSV with this row
        metrics_df.to_csv(filename, index=False)
        print("Created new CSV file and added the evaluation metrics.")

    return None

def seed_dataset(args):
    """
    Generate images using a trained dino-ldm model and save them to disk.
    """
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.cuda.empty_cache()
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    dtype = torch.float32

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_path = os.path.join(args.save_path, args.model, args.dataset, args.method, str(args.seed))

    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
        
    unet = torch.load(args.model_path, map_location=device)
    unet.to(device)

    if args.scheduler=="ddpm":
        scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    else:
        scheduler = DDIMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
            
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=None)
    vae.requires_grad_(False)

    _,dataloader = load_and_prepare_dataset(dataset_name=args.dataset, batch_size=args.bs, img_size=(args.height, args.width),data_dir=None, rep_dir=args.rep_dir)
 
    unet.eval()
    if args.model == 'dino-ldm' or args.model == 'clip-ldm':
        for step, batch in enumerate(dataloader):
            latents = torch.randn((args.bs, 4, args.height // 8, args.width // 8))
            latents = latents.to(device)
            scheduler.set_timesteps(args.num_inference_steps)
            encoder_hidden_states = batch.to(device, dtype=dtype).unsqueeze(1)  # Add a channel dimension
            with autocast('cuda'):
                progress_bar = tqdm(range(0, args.num_inference_steps))
                progress_bar.set_description("Steps")
                for i, t in enumerate(scheduler.timesteps):
                    with torch.no_grad(): 
                        noise_pred = unet(latents, t, encoder_hidden_states)['sample']    
                    latents = scheduler.step(noise_pred, t, latents)['prev_sample']
                    progress_bar.update(1)
            latents = 1 / 0.18215 * latents
            vae = vae.to(device)
            with torch.no_grad():
                decoder_output = vae.decode(latents)
                imgs = decoder_output.sample if hasattr(decoder_output, 'sample') else decoder_output
            imgs = (imgs / 2 + 0.5).clamp(0, 1)
            imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
            imgs = (imgs * 255).round().astype('uint8')
            pil_images = [Image.fromarray(image) for image in imgs]
            for idx, pil_img in enumerate(pil_images):
                img_path = os.path.join(save_path, f"image_{step}_{idx}.png")
                pil_img.save(img_path)
                    
            del latents, encoder_hidden_states, decoder_output, imgs, pil_images, noise_pred, batch
            torch.cuda.empty_cache()
            print(f"Batch {step+1}/{len(dataloader)} is Saved!")    
    
    elif args.model == 'diffae':
        raise NotImplementedError("DiffAE model is not supported yet.")
    
    else:
        for step in range(0,len(dataloader)):
            latents = torch.randn((args.bs, 4, args.height // 8, args.width // 8))
            latents = latents.to(device)
            scheduler.set_timesteps(args.num_inference_steps)
            with autocast('cuda'):
                progress_bar = tqdm(range(0, args.num_inference_steps))
                progress_bar.set_description("Steps")
                for i, t in enumerate(scheduler.timesteps):
                    with torch.no_grad(): 
                        noise_pred = unet(latents, t)['sample']    
                    latents = scheduler.step(noise_pred, t, latents)['prev_sample']
                    progress_bar.update(1)
            latents = 1 / 0.18215 * latents
            vae = vae.to(device)
            with torch.no_grad():
                decoder_output = vae.decode(latents)
                imgs = decoder_output.sample if hasattr(decoder_output, 'sample') else decoder_output
            imgs = (imgs / 2 + 0.5).clamp(0, 1)
            imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
            imgs = (imgs * 255).round().astype('uint8')
            pil_images = [Image.fromarray(image) for image in imgs]
            for idx, pil_img in enumerate(pil_images):
                img_path = os.path.join(save_path, f"image_{step}_{idx}.png")
                pil_img.save(img_path)
                    
            del latents, decoder_output, imgs, pil_images, noise_pred,
            torch.cuda.empty_cache()
            print(f"Batch {step+1}/{len(dataloader)} is Saved!")

def perturbate_dataset(args):
    """
    Generate images using a trained dino-ldm model and save them to disk.
    """
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    random.seed(42)
    torch.cuda.empty_cache()
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    dtype = torch.float32
    alpha = np.linspace(0, 1, 5)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_path = os.path.join(args.save_path, args.model, args.dataset, args.method)
        
    unet = torch.load(args.model_path, map_location=device)
    unet.to(device)

    if args.scheduler=="ddpm":
        scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    else:
        scheduler = DDIMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
            
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=None)
    vae.requires_grad_(False)

    _,dataloader = load_and_prepare_dataset(dataset_name=args.dataset, batch_size=args.bs, img_size=(args.height, args.width),data_dir=None, rep_dir=args.rep_dir)
 
    unet.eval()
    if args.model == 'dino-ldm' or args.model == 'clip-ldm':
        for lamda in alpha:
            print(f"Generating images for alpha value: {alpha[lamda]}")
            save_path_lamda = os.path.join(save_path, str(lamda))
            if not os.path.exists(save_path_lamda):
                os.makedirs(save_path_lamda, exist_ok=True)
            for step, batch in enumerate(dataloader):
                latents = torch.randn((args.bs, 4, args.height // 8, args.width // 8))
                latents = latents.to(device)
                scheduler.set_timesteps(args.num_inference_steps)
                encoder_hidden_states = batch.to(device, dtype=dtype).unsqueeze(1)
                noise = torch.randn(encoder_hidden_states.shape, dtype=encoder_hidden_states.dtype,
                                device=encoder_hidden_states.device) * lamda
                encoder_hidden_states = encoder_hidden_states + noise

                with autocast('cuda'):
                    progress_bar = tqdm(range(0, args.num_inference_steps))
                    progress_bar.set_description("Steps")
                    for i, t in enumerate(scheduler.timesteps):
                        with torch.no_grad(): 
                            noise_pred = unet(latents, t, encoder_hidden_states)['sample']    
                        latents = scheduler.step(noise_pred, t, latents)['prev_sample']
                        progress_bar.update(1)
                latents = 1 / 0.18215 * latents
                vae = vae.to(device)
                with torch.no_grad():
                    decoder_output = vae.decode(latents)
                    imgs = decoder_output.sample if hasattr(decoder_output, 'sample') else decoder_output
                imgs = (imgs / 2 + 0.5).clamp(0, 1)
                imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
                imgs = (imgs * 255).round().astype('uint8')
                pil_images = [Image.fromarray(image) for image in imgs]
                for idx, pil_img in enumerate(pil_images):
                    img_path = os.path.join(save_path, f"image_{step}_{idx}.png")
                    pil_img.save(img_path)
                        
                del latents, encoder_hidden_states, decoder_output, imgs, pil_images, noise_pred, batch
                torch.cuda.empty_cache()
                print(f"Batch {step+1}/{len(dataloader)} is Saved!")    
    
    elif args.model == 'diffae':
        raise NotImplementedError("DiffAE model is not supported yet.")
    
    else:
        for step in range(0,len(dataloader)):
            latents = torch.randn((args.bs, 4, args.height // 8, args.width // 8))
            latents = latents.to(device)
            scheduler.set_timesteps(args.num_inference_steps)
            with autocast('cuda'):
                progress_bar = tqdm(range(0, args.num_inference_steps))
                progress_bar.set_description("Steps")
                for i, t in enumerate(scheduler.timesteps):
                    with torch.no_grad(): 
                        noise_pred = unet(latents, t)['sample']    
                    latents = scheduler.step(noise_pred, t, latents)['prev_sample']
                    progress_bar.update(1)
            latents = 1 / 0.18215 * latents
            vae = vae.to(device)
            with torch.no_grad():
                decoder_output = vae.decode(latents)
                imgs = decoder_output.sample if hasattr(decoder_output, 'sample') else decoder_output
            imgs = (imgs / 2 + 0.5).clamp(0, 1)
            imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
            imgs = (imgs * 255).round().astype('uint8')
            pil_images = [Image.fromarray(image) for image in imgs]
            for idx, pil_img in enumerate(pil_images):
                img_path = os.path.join(save_path, f"image_{step}_{idx}.png")
                pil_img.save(img_path)
                    
            del latents, decoder_output, imgs, pil_images, noise_pred,
            torch.cuda.empty_cache()
            print(f"Batch {step+1}/{len(dataloader)} is Saved!")


def parse_args():
    parser = argparse.ArgumentParser(description="Quantitative Evaluation of Generative Models")
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        required=True,
        help="Path to DINO-LDM/Unconditional LDM checkpoint",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="dino-ldm",
        required=True,
        help="Choose among dino-ldm or clip-ldm or diffae or u-ldm",
    )

    parser.add_argument(
        "--method",
        type=str,
        default="seed-dataset",
        required=True,
        help="Choose among perturbate-dataset or interpolate-dataset",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default='ffhq',
        required=True,
        help="Choose among ffhq or celeba or subset-imagenet",
    )

    parser.add_argument(
        "--test_dir",
        type=str,
        default='./data/ffhq512/test',
        required=True,
        help="Path to the test images",
    ) 

    parser.add_argument(
        "--rep_dir",
        type=str,
        default='./rep/ffhq512-dinov2',
        required=True,
        help="Path to the representation files",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        required=False,
        help="Select Seed value",
    )

    parser.add_argument(
        "--bs",
        type=int,
        default=10,
        required=False,
        help="Select the batch size for the dataloader depending on the device memory",
    )

    parser.add_argument(
        "--height",
        type=int,
        default=512,
        required=False,
        help="Height of the Image",
    )

    parser.add_argument(
        "--width",
        type=int,
        default=512,
        required=False,
        help="Width of the Image",
    )

    parser.add_argument(
        "--save_path",
        type=str,
        default="./eval_dataset",
        required=False,
        help="Path to save the generated dataset",
    )

    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=100,
        required=False,
        help="No of Inference Steps",
    )

    parser.add_argument(
        "--scheduler",
        type=str,
        default="ddpm",
        required=False,
        help="Select the scheduler",
    )

    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="CompVis/stable-diffusion-v1-4",
        required=False,
        help="Path to the pretrained model",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    if args.model not in ['dino-ldm', 'clip-ldm', 'diffae', 'baseline']:
        raise ValueError("Invalid method! Choose among dino-ldm, clip-ldm, diffae, or baseline")
    if args.method not in ['seed-dataset', 'perturbate-dataset', 'interpolate-dataset']:
        raise ValueError("Invalid method! Choose among perturbate-dataset or interpolate-dataset")
    if args.dataset not in ['ffhq', 'celeba', 'subset-imagenet']:
        raise ValueError("Invalid dataset! Choose among ffhq, celeba, or subset-imagenet")
    
    gen_path = os.path.join(args.save_path, args.model, args.dataset)
    if not os.path.exists(gen_path):
        os.makedirs(gen_path, exist_ok=True)
        print(f'Creating {args.dataset} folder in {args.save_path}/{args.model}')
    else:
        print(f'{args.dataset} folder already exists in {args.save_path}/{args.model}')
    
    if args.method == 'seed-dataset':
        seed_dataset(args)
        gen_path = os.path.join(args.save_path, args.model, args.dataset, args.method, str(args.seed))
        torch_metrics(args, gen_path)

    elif args.method == 'perturbate-dataset':
        perturbate_dataset(args)
        save_path = os.path.join(args.save_path, args.model, args.dataset, args.method, str(args.seed))
        alpha = np.linspace(0, 1, 5)
        for lamda in alpha:
            gen_path = os.path.join(save_path, str(lamda))
            torch_metrics(args, gen_path)

    elif args.method == 'interpolate-dataset':
        raise NotImplementedError("Interpolate dataset functionality is not implemented yet.")
    else:
        raise ValueError("Invalid method! Choose among seed-dataset or interpolate-dataset")
    
    
if __name__ == "__main__":
    main()