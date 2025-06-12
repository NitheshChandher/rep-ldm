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
from diffusers import VQModel, AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from utils.lr_scheduler import LambdaLinearScheduler
from tqdm.auto import tqdm
import transformers
import diffusers
import yaml
import wandb
from utils.utility import decode_img_latents, produce_latents
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
        rep_dir=pathlib.Path(config["rep_dir"])
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
            dir=os.path.join(root_path, config["trials"], "wandb"),
        )

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

    # Load or initialize UNet
    if config['train_from_scratch']:
        unet = UNet2DConditionModel(
            sample_size=int(config["resolution"]) // 8,
            in_channels=config['channels'],
            out_channels=config['channels'],
            cross_attention_dim=config['embed_dim']
        )
    else:
        unet = torch.load(model_path, weights_only=False)
        print(f"Resuming training from checkpoint: {model_path}")

    vae.requires_grad_(False)
    unet.requires_grad_(True)

    # Watch the model with wandb for gradients and parameters
    if accelerator.is_main_process:
        wandb.watch(unet, log_freq=100)

    # Training steps and optimizer
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / config['gradient_accumulation_steps'])
    max_train_steps = config['num_train_epochs'] * num_update_steps_per_epoch

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

    # Prepare for distributed training
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )
    vae.to(accelerator.device, dtype=torch.float32)
    unet.to(accelerator.device, dtype=torch.float32)

    # Recalculate steps if dataloader size changed
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / config['gradient_accumulation_steps'])
    max_train_steps = config['num_train_epochs'] * num_update_steps_per_epoch
    config["num_train_epochs"] = math.ceil(max_train_steps / num_update_steps_per_epoch)

    print(f"Total batch size: {config['batch_size'] * accelerator.num_processes * config['gradient_accumulation_steps']}")
    print(f"Gradient accumulation steps: {config['gradient_accumulation_steps']}")
    print(f"Number of processes: {accelerator.num_processes}")
    print(f"Number of training steps: {max_train_steps}")
    print(f"Number of epochs: {config['num_train_epochs']}")
    print(f"Number of update steps per epoch: {num_update_steps_per_epoch}")

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
    prev_step = 0 if config["train_from_scratch"] else int(config.get("prev_step", 0))
    train_loss = []
    dtype = torch.float32
    root_path = os.path.join(config["output_dir"], config["trials"])
    if epoch % config["checkpoint_epoch"] == 0:
        checkpoint = True
    else:
        checkpoint = False

    for step, batch in enumerate(train_dataloader):
        optimizer.zero_grad()
        with accelerator.accumulate(unet):
            latents = vae.encode(batch[0].to(accelerator.device, dtype=dtype)).latent_dist.sample().detach()
            latents = latents * 0.18215
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device
            ).long()
            encoder_hidden_states = batch[1].to(accelerator.device, dtype=dtype).unsqueeze(1)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

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
                progress_bar.update(1)
                global_step += 1

            optimizer.step()
            if config["lr_scheduler"] == "constant":
                lr_scheduler.step()
            else:
                lr_scheduler(global_step)
            logs = {"train/loss_per_step": loss.detach().item(), "step": global_step}
            progress_bar.set_postfix(**logs)
            if accelerator.is_main_process:
                wandb.log(logs, step=global_step)
            train_loss.append(loss.detach().item())

    # Optionally log checkpoint info
    if checkpoint and accelerator.is_main_process:
        num_samples = config["num_samples"]
        encoder_hidden_states = encoder_hidden_states[:num_samples, :, :]
        unwrapped_model = accelerator.unwrap_model(unet)
        pred_latents = produce_latents(
            config, encoder_hidden_states=encoder_hidden_states, unet=unwrapped_model, seed=config['seed']
        )
        pred_imgs = decode_img_latents(pred_latents, config, vae=vae)
        save_path = os.path.join(root_path, "images")
        os.makedirs(save_path, exist_ok=True)
        for idx, pil_img in enumerate(pred_imgs):
            img_path = os.path.join(save_path, f"image_{idx}_epoch-{epoch}_step-{accelerator.num_processes * global_step + prev_step}.png")
            pil_img.save(img_path)
            wandb.log({f"train/generated_image_{idx}": wandb.Image(pil_img, caption=f"epoch {epoch} step {global_step}")}, step=global_step)
            break

        checkpoint_path = os.path.join(root_path, "model")
        os.makedirs(checkpoint_path, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_path, f"{config['name']}_epoch-{epoch}_step-{accelerator.num_processes * global_step + prev_step}.pth")
        torch.save(unwrapped_model, checkpoint_path)
        del unwrapped_model

    return train_loss

def eval_epoch(vae, unet, val_dataloader, noise_scheduler, accelerator):
    """
    Evaluate the model for one epoch.
    """
    dtype = torch.float32
    unet.eval()
    val_loss = []
    for _, batch in enumerate(val_dataloader):
        with torch.no_grad():
            latents = vae.encode(batch[0].to(accelerator.device, dtype=dtype)).latent_dist.sample().detach()
            latents = latents * 0.18215
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device
            ).long()
            encoder_hidden_states = batch[1].to(accelerator.device, dtype=dtype).unsqueeze(1)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
            val_loss.append(loss.detach().item())
    return val_loss

def objective(config):
    """
    Orchestrates the full training: setup, training loop, evaluation, logging, and checkpointing.
    """
    (unet, vae, accelerator, optimizer, train_dataloader, val_dataloader,
     lr_scheduler, noise_scheduler, max_train_steps) = setup(config)

    progress_bar = tqdm(range(max_train_steps), desc="Training Progress")
    for epoch in range(config["num_train_epochs"]):
        train_loss = train_epoch(
            vae, unet, train_dataloader, accelerator, optimizer, lr_scheduler,
            noise_scheduler, progress_bar, config, epoch=epoch
        )
        if config["validation"] is True:
            if epoch % 10 == 0:
                val_loss = eval_epoch(vae, unet, val_dataloader, noise_scheduler, accelerator)
                if accelerator.is_main_process:
                    wandb.log({
                        "train/loss_epoch": sum(train_loss) / len(train_loss),
                        "val/loss_epoch": sum(val_loss) / len(val_loss),
                        "epoch": epoch
                    }, step=epoch)
            else:    
                if accelerator.is_main_process:
                    wandb.log({
                        "train/loss_epoch": sum(train_loss) / len(train_loss),
                        "epoch": epoch
                    }, step=epoch)
        else:
            if accelerator.is_main_process:
                    wandb.log({
                        "train/loss_epoch": sum(train_loss) / len(train_loss),
                        "epoch": epoch
                    }, step=epoch)
    if accelerator.is_main_process:
        wandb.finish()
    print("Training complete.")

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
