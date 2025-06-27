import os
import torch
from dataset.dataloader import load_and_prepare_dataset
from diffusers import DDPMScheduler, DDIMScheduler, AutoencoderKL
from torch import autocast
from tqdm.auto import tqdm
from PIL import Image
import numpy as np
import random

def interpolate_dataset(args):
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
    save_path = os.path.join(args.save_path, args.model, args.dataset, args.method)

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
 
    unet.eval()
    if args.model == 'dino-ldm' or args.model == 'clip-ldm' or args.model == 'diffae':
        if args.rep_dir is None:
            raise ValueError("rep_dir must be provided for dino-ldm or clip-ldm or diffae models.")

        _,dataloader = load_and_prepare_dataset(dataset_name=args.dataset, batch_size=args.bs, img_size=(args.height, args.width),data_dir=None, rep_dir=args.rep_dir)
        first_element = None
        prev_element = None
        all_interpolated = []
        for step, batch in enumerate(dataloader):
            encoder_hidden_states = batch.to(device, dtype=dtype).unsqueeze(1)  # Add a channel dimension

            if step == 0:
                first_element = encoder_hidden_states[0]  # Save for wraparound
            
            all_interpolated = []
            for i in range(args.bs):
                if i < args.bs - 1:
                    interpolated = 0.5 * encoder_hidden_states[i] + 0.5 * encoder_hidden_states[i + 1]
                else:
                    if prev_element is not None:
                        interpolated = 0.5 * encoder_hidden_states[i] + 0.5 * prev_element
                    elif step == len(dataloader) - 1:
                        interpolated = 0.5 * encoder_hidden_states[i] + 0.5 * first_element
                    else:
                        prev_element = encoder_hidden_states[i]
                        continue
                all_interpolated.append(interpolated)
            interpolated_tensor = torch.cat(all_interpolated, dim=0).unsqueeze(1)   # [N, 1, 768]
            num_interp = interpolated_tensor.size(0)
            latents = torch.randn((num_interp, 4, args.height // 8, args.width // 8))
            latents = latents.to(device)
            scheduler.set_timesteps(args.num_inference_steps)

            with autocast('cuda'):
                progress_bar = tqdm(range(0, args.num_inference_steps))
                progress_bar.set_description("Steps")
                for i, t in enumerate(scheduler.timesteps):
                    with torch.no_grad(): 
                        noise_pred = unet(latents, t, interpolated_tensor)['sample']    
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
    
    elif args.model == 'baseline':
        raise NotImplementedError("Baseline model is not supported.")

    else:
        raise ValueError(f"Model {args.model} is not supported.")