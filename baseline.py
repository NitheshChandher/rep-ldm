import os
import math
import logging
import argparse
import pathlib
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import VQModel, AutoencoderKL, DDPMScheduler, UNet2DModel
from diffusers.optimization import get_scheduler
from utils.lr_scheduler import LambdaLinearScheduler
from tqdm.auto import tqdm
import transformers
import diffusers
import yaml
import wandb
from utils.utility import decode_img_latents, produce_latents, make_image_grid
from dataset.dataloader import load_and_prepare_dataset

global_step = 0

def setup(config):
    """
    Set up the training environment: datasets, models, optimizer, scheduler, and logging.
    """
    exp = config['trials']
    root_path = os.path.join(config["output_dir"], exp)
    log_path = os.path.join(root_path, "logs")
    model_dir = os.path.join(root_path, "model")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{config['name']}.pth")

    train_dataloader, val_dataloader = load_and_prepare_dataset(
        dataset_name=config["dataset"],
        batch_size=config["batch_size"],
        img_size=(config["resolution"], config["resolution"]),
        data_dir=pathlib.Path(config["data_dir"]),
        rep_dir=None
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=config['gradient_accumulation_steps'],
        mixed_precision="no",
        log_with=None,
        project_dir=log_path,
    )

    # Set logging verbosity
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    set_seed(config['seed'])

    # Initialize W&B (main process only)
    if accelerator.is_main_process:
        wandb.init(
            project=config.get("wandb_project", config["name"]),
            name=config.get("name", None),
            config=config,
            dir=root_path,
        )
        wandb.define_metric("global_step", hidden=True)
        wandb.define_metric("train/loss_per_step", step_metric="global_step")
        wandb.define_metric("train/loss_per_epoch", step_metric="epoch")
        wandb.define_metric("val/loss_per_epoch", step_metric="epoch")
    # Load noise scheduler and VAE
    noise_scheduler = DDPMScheduler.from_pretrained(
        config["pretrained_model_name_or_path"], subfolder="scheduler"
    )
    if config['encoder_name'] == 'KL':
        vae = AutoencoderKL.from_pretrained(
            config["pretrained_model_name_or_path"], subfolder="vae"
        )
    elif config['encoder_name'] == 'VQ':
        vae = VQModel.from_pretrained(
            config["pretrained_model_name_or_path"], subfolder="vae"
        )
    else:
        raise ValueError("Invalid Autoencoder type specified.")

    resume_path = None
    if not config.get('train_from_scratch', False):
        resume_path = config.get('resume_path')
        print(f"Resuming from {resume_path}")

    # Load or initialize UNet
    if config['train_from_scratch']:
        unet = UNet2DModel(
                sample_size=int(config["resolution"]) // 8,                
                in_channels=config['channels'],                  
                out_channels=config['channels'],    
                down_block_types=(
                    'DownBlock2D',
                    'DownBlock2D',
                    'DownBlock2D',
                    'DownBlock2D',
                    'DownBlock2D', 
                ),
                up_block_types=(
                    "UpBlock2D", 
                    "UpBlock2D", 
                    "UpBlock2D",
                    "UpBlock2D",
                    "UpBlock2D",
                ),
                block_out_channels=(192, 384, 384, 768, 768), 
                layers_per_block=2,            
                attention_head_dim=8,    
            )
    else:
        unet = torch.load(model_path, weights_only=False)
        print(f"Resuming training from checkpoint: {model_path}")

    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=config["lr"],
        betas=(config["beta_1"], config["beta_2"]),
        weight_decay=config["weight_decay"],
        eps=1e-8,
    )

    # Learning rate scheduler
    if config["lr_scheduler"] == "constant":
        lr_scheduler = get_scheduler(
            name="constant",
            optimizer=optimizer,
            num_warmup_steps=500 * config['gradient_accumulation_steps'],
            num_training_steps=max_train_steps * config['gradient_accumulation_steps'],
        )
    elif config["lr_scheduler"] == "lambdalinear":
        lr_scheduler = LambdaLinearScheduler(
            warm_up_steps=[10000],
            f_min=[0.1],
            f_max=[1.],
            f_start=[1e-6],
            cycle_lengths=[1000000],
        )
    else:
        raise ValueError(f"Unknown lr_scheduler {config['lr_scheduler']}")

    if resume_path is not None and os.path.exists(resume_path):
        checkpoint = torch.load(resume_path, map_location="cpu")
        unet.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if hasattr(lr_scheduler, "load_state_dict") and "scheduler_state_dict" in checkpoint:
            lr_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        config["resume_epoch"] = checkpoint.get("epoch", 0)
        config["prev_step"] = checkpoint.get("global_step", 0)
    else:
        config["resume_epoch"] = 0
        config["prev_step"] = 0

    vae.requires_grad_(False)
    unet.requires_grad_(True)

    # Watch the model with wandb for gradients and parameters
    if accelerator.is_main_process:
        wandb.watch(unet, log_freq=100)

    # Prepare for distributed training
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )
    vae.to(accelerator.device)

    # Recalculate steps if dataloader size changed
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / config['gradient_accumulation_steps'])
    if config.get('train_from_scratch', True): 
        max_train_steps = config['num_train_epochs'] * num_update_steps_per_epoch
    else:
        max_train_steps = (config['num_train_epochs'] - config["resume_epoch"]) * num_update_steps_per_epoch + config["prev_step"]

    return (unet, vae, accelerator, optimizer, train_dataloader, val_dataloader,
            lr_scheduler, noise_scheduler, max_train_steps)

def train_epoch(
    vae, unet, train_dataloader, accelerator, optimizer, lr_scheduler,
    noise_scheduler, progress_bar, config, epoch
):
    """
    Train the model for one epoch.
    """
    global global_step
    unet.train()
    root_path = os.path.join(config["output_dir"], config["trials"])
    checkpoint = (epoch % config["checkpoint_epoch"] == 0)
    train_loss = []

    for step, batch in enumerate(train_dataloader):
        optimizer.zero_grad()
        with accelerator.accumulate(unet):
            latents = vae.encode(batch[0].to(accelerator.device)).latent_dist.sample().detach()
            latents = latents * 0.18215
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device
            ).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            model_pred = unet(noisy_latents, timesteps).sample

            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
            accelerator.backward(loss)

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(unet.parameters(), 1.0)
                optimizer.step()
                global_step += 1
                progress_bar.update(1)
                progress_bar.set_description(f"Epoch {epoch + 1} Step {step + 1}/{len(train_dataloader)} Loss: {loss.item():.4f}")

            if config["lr_scheduler"] == "constant":
                lr_scheduler.step()
            else:
                lr_scheduler(global_step)

            if accelerator.is_main_process:
                wandb.log({"loss_per_step": loss.item()})
            
            train_loss.append(loss.detach().item())

    # Optionally log checkpoint info
    if checkpoint and accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(unet)
        pred_latents = produce_latents(
            config, unet=unwrapped_model, seed=config['seed']
        )
        pred_imgs = decode_img_latents(pred_latents, config, vae=vae)
        save_path = os.path.join(root_path, "images")
        os.makedirs(save_path, exist_ok=True)
        img_path = os.path.join(save_path, f"image-grid_epoch-{epoch}_step-{accelerator.num_processes * global_step}.png")
        grid_image = make_image_grid(pred_imgs, grid_size=(2, 2), padding=10, bg_color=(255, 255, 255))
        grid_image.save(img_path)
        wandb.log({f"train/generated_image_epoch-{epoch}": wandb.Image(grid_image, caption=f"epoch {epoch} step {global_step}"), "epoch": epoch})
    
        checkpoint_data = {
            "epoch": epoch + 1,
            "global_step": global_step,
            "model_state_dict": unwrapped_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": lr_scheduler.state_dict() if hasattr(lr_scheduler, "state_dict") else None
        }
        ckpt_path = os.path.join(root_path, "model", f"{config['name']}_epoch{epoch+1}_step{global_step}.pt")
        torch.save(checkpoint_data, ckpt_path)
        print(f"Checkpoint saved at {ckpt_path}")

    return sum(train_loss) / len(train_loss)

def eval_epoch(vae, unet, val_dataloader, noise_scheduler, accelerator):
    """
    Evaluate the model for one epoch.
    """
    unet.eval()
    val_loss = []
    for _, batch in enumerate(val_dataloader):
        with torch.no_grad():
            latents = vae.encode(batch[0].to(accelerator.device)).latent_dist.sample().detach()
            latents = latents * 0.18215
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device
            ).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            model_pred = unet(noisy_latents, timesteps).sample

            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
            val_loss.append(loss.detach().item())
    return sum(val_loss) / len(val_loss)

def objective(config):
    """
    Orchestrates the full training: setup, training loop, evaluation, logging, and checkpointing.
    """
    (unet, vae, accelerator, optimizer, train_dataloader, val_dataloader,
     lr_scheduler, noise_scheduler, max_train_steps) = setup(config)

    progress_bar = tqdm(range(max_train_steps), desc="Training Progress")
    start_epoch = config.get("resume_epoch", 0)
    final_model_path = os.path.join(
        config["output_dir"],
        config["trials"],
        "model",
        f"{config['name']}_final_full_model.pth"
    )

    for epoch in range(start_epoch, config["num_train_epochs"]):
        if accelerator.is_main_process:
            wandb.log({"epoch": epoch})
        train_loss = train_epoch(
            vae, unet, train_dataloader, accelerator, optimizer, lr_scheduler,
            noise_scheduler, progress_bar, config, epoch=epoch
        )
        if accelerator.is_main_process:
            wandb.log({"train/loss_per_epoch": train_loss, "epoch": epoch})
    
        if config.get("validation", False) and (epoch % 10 == 0):
            val_loss = eval_epoch(vae, unet, val_dataloader, noise_scheduler, accelerator)
            if accelerator.is_main_process:
                wandb.log({"val/loss_per_epoch": val_loss, "epoch": epoch})
            
    if accelerator.is_main_process:
        wandb.finish()
    unwrapped_model = accelerator.unwrap_model(unet)
    torch.save(unwrapped_model, final_model_path)
    print(f"✅ Full model saved to: {final_model_path}")

def main():
    """
    Main entry point. Parses arguments, loads config, and launches training.
    """
    parser = argparse.ArgumentParser(description="Train Representation Conditioned Latent Diffusion Model")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    objective(config)

if __name__ == "__main__":
    main()
