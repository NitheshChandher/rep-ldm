import argparse
import os
import torch
import numpy as np
import random
from diffusers import DDIMScheduler, AutoencoderKL
from PIL import Image
from dataset.dataloader import load_and_prepare_dataset
from utils.inversion_utils import inversion_forward_process, inversion_reverse_process
import torch.nn.functional as F
from pytorch_msssim import ssim
from lpips import LPIPS
from torch import inference_mode, autocast
from tqdm import tqdm
from torchvision import transforms


def save_image_tensor(img_tensor, path, is_real=False):
    if is_real:
        img_tensor = img_tensor.detach().cpu().float()
        if img_tensor.min() < 0:  # Images in [-1, 1] → denormalize
            img_tensor = (img_tensor + 1) / 2
        img_np = (img_tensor.permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8)
        Image.fromarray(img_np).save(path)
    else:
        img_tensor.save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device_num", type=int, default=0)
    parser.add_argument("--num_diffusion_steps", type=int, default=100)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--rep_path", type=str, default="rep/test")
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--res", type=int, default=256)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--skip", type=int, default=36)
    args = parser.parse_args()

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    random.seed(42)
    torch.cuda.empty_cache()
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    dtype = torch.float32
    device = torch.device(f"cuda:{args.device_num}" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.output_path, exist_ok=True)
    real_dir = os.path.join(args.output_path, "real")
    recon_dir = os.path.join(args.output_path, "recon")
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(recon_dir, exist_ok=True)

    _, test_loader = load_and_prepare_dataset("lsun-church", batch_size=10,
                                              img_size=(args.res, args.res),
                                              data_dir=args.data_path,
                                              rep_dir=args.rep_path, shuffle=False)

    model_id = "CompVis/stable-diffusion-v1-4"
    scheduler = DDIMScheduler.from_config(model_id, subfolder="scheduler")
    scheduler.set_timesteps(args.num_diffusion_steps)

    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae").to(device).eval()
    unet = torch.load(args.model_path, map_location=device).to(device).eval()

    # ------------------ STEP 1: GENERATE AND SAVE IMAGES ------------------
    print("Generating and saving images...")
    image_idx = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Generating Images"):
            latents = torch.randn((batch[1].shape[0], 4, args.res // 8, args.res // 8))
            latents = latents.to(device)
            real_imgs = batch[0]
            encoder_hidden_states = batch[1].unsqueeze(1).to(device)
            with autocast("cuda"), inference_mode():
                for i, t in enumerate(scheduler.timesteps):
                    with torch.no_grad(): 
                        noise_pred = unet(latents, t, encoder_hidden_states)['sample']    
                    latents = scheduler.step(noise_pred, t, latents)['prev_sample']

            x0_recon = vae.decode(1 / 0.18215 * latents).sample
            recon_imgs = (x0_recon / 2 + 0.5).clamp(0, 1)
            imgs = recon_imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
            imgs = (imgs * 255).round().astype('uint8')
            pil_images = [Image.fromarray(image) for image in imgs]

            for i in range(real_imgs.size(0)):
                real_path = os.path.join(real_dir, f"{image_idx:05d}.png")
                recon_path = os.path.join(recon_dir, f"{image_idx:05d}.png")
                save_image_tensor(real_imgs[i], real_path, is_real=True)
                save_image_tensor(pil_images[i], recon_path)
                image_idx += 1

            del real_imgs, recon_imgs, latents, noise_pred, encoder_hidden_states

    print(f"Total images saved: {image_idx}")
    del vae, unet, scheduler
    # ------------------ STEP 2: METRICS CALCULATION ------------------
    print("Calculating metrics from saved images...")

    transform = transforms.Compose([transforms.ToTensor()])
    lpips_fn = LPIPS(net='vgg').to(device).eval()

    total_mse, total_ssim, total_lpips, count = 0.0, 0.0, 0.0, 0
    all_indices = sorted([f"{i:05d}.png" for i in range(image_idx)])

    with torch.no_grad():
        for idx in tqdm(all_indices, desc="Computing Metrics"):
            real_img = transform(Image.open(os.path.join(real_dir, idx))).unsqueeze(0).to(device)
            recon_img = transform(Image.open(os.path.join(recon_dir, idx))).unsqueeze(0).to(device)

            mse = F.mse_loss(recon_img, real_img, reduction='mean')
            ssim_val = ssim(recon_img.float(), real_img.float(), data_range=1.0, size_average=True)
            lpips_val = lpips_fn(recon_img * 2 - 1, real_img * 2 - 1).mean()

            total_mse += mse.item()
            total_ssim += ssim_val.item()
            total_lpips += lpips_val.item()
            count += 1

    avg_mse = total_mse / count
    avg_ssim = total_ssim / count
    avg_lpips = total_lpips / count

    print(f"\nFinal Averages:\nMSE: {avg_mse:.6f}\nSSIM: {avg_ssim:.6f}\nLPIPS: {avg_lpips:.6f}")

    with open(os.path.join(args.output_path, "metrics.txt"), "w") as f:
        f.write(f"Average MSE: {avg_mse:.6f}\n")
        f.write(f"Average SSIM: {avg_ssim:.6f}\n")
        f.write(f"Average LPIPS: {avg_lpips:.6f}\n")


if __name__ == "__main__":
    main()
