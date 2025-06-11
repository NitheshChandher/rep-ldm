import sys
import argparse
import logging
import math
import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.tensorboard import SummaryWriter
import math
import diffusers
import transformers
import yaml
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import VQModel, AutoencoderKL, DDPMScheduler, UNet2DModel
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm
import tensorflow as tf
sys.path.append('.')
from support import np_tile_imgs, save_tiled_imgs, save_progress, save_image, load_data, get_full_repo_name, visualize_and_save_latent
from sample import decode_img_latents, produce_latents
from dataset import load_and_prepare_dataset
from lr_scheduler import LambdaLinearScheduler
from PIL import Image
tf.config.experimental.set_visible_devices([], "GPU")
import pathlib
sys.path.append('.')
global_step = 0
TF_ENABLE_ONEDNN_OPTS=0

def setup(conf):
    """
    This function is used to setup the training environment. It loads the datasets, initializes the models, optimizers, and schedulers,
    and prepares the training environment.

    Args:
    - conf: The configuration for the experiment.(/config/uldm.yaml)

    Returns:
    - unet: Unconditional U-Net model for training.
    - vae: Pre-trained VAE model from CompVis/stable-diffusion-v1-4.
    - accelerator: Accelerator object for distributed training.
    - optimizer: Optimizer for training the model.
    - train_dataloader: training dataloader for model training.
    - val_dataloader: validation dataloader for model evaluation.
    - lr_scheduler: Learning rate scheduler for optimizer.
    - noise_scheduler: Noise scheduler for adding noise to latents.
    - writer: Tensorboard SummaryWriter for logging.
    - logger: Accelerate logger for logging.
    - max_train_steps: Maximum number of training steps.
    """
    
    exp = conf['trials']
    root_path = conf["output_dir"] + exp
    log_path =  root_path + "/logs"
    ptmodel_path = conf["pretrained_model_name_or_path"]
    data_dir = pathlib.Path(conf["data_dir"])

    seed = conf['seed']
    dataset_name = conf["dataset"]   
    size = conf["resolution"]

    lr = conf["lr"]
    lr_name = conf["lr_scheduler"]
    weight_decay = conf["weight_decay"]
    beta_1 = conf["beta_1"]
    beta_2 = conf["beta_2"]
    bs = conf["batch_size"] 
    gradient_accumulation_steps = conf['gradient_accumulation_steps']
    epochs = conf['num_train_epochs']
    encoder_name = conf['encoder_name']
    channels = conf['channels']
    
    if root_path is not None:
        os.makedirs(root_path, exist_ok=True)

    writer = SummaryWriter(log_path)

    train_dataloader, val_dataloader = load_and_prepare_dataset(dataset_name=dataset_name, batch_size=bs, img_size=(size, size),data_dir=data_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps,
        mixed_precision="no",
        log_with="tensorboard",
        project_dir=log_path,
    )

    logger = get_logger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if accelerator.is_local_main_process:
    #    datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
        
    else:
    #    datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    set_seed(seed)

    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(
        ptmodel_path, subfolder="scheduler")

    if encoder_name =='KL':
        vae = AutoencoderKL.from_pretrained(
            ptmodel_path, subfolder="vae", revision=None)
            
        unet = UNet2DModel(
                sample_size=int(size)// 8,                
                in_channels=channels,                  
                out_channels=channels,    
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
        vae = VQModel.from_pretrained(ptmodel_path, subfolder="vae", revision=None)
        unet = UNet2DModel(
                sample_size=int(size)// 8,
                in_channels=channels,
                out_channels=channels,
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
        
        
    #Freeze vae and unet
    vae.requires_grad_(False)
    unet.requires_grad_(True)

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / gradient_accumulation_steps)

    max_train_steps = epochs * num_update_steps_per_epoch
    overrode_max_train_steps = True

    print(f"No of update steps per epoch before accelerator: {num_update_steps_per_epoch}, Max Train Steps: {max_train_steps}")

    # Initialize the optimizer
    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=lr,
        betas=(beta_1, beta_2),
        weight_decay=weight_decay,
        eps=1e-08,
        )
    
    if lr_name == "constant":
        lr_scheduler = get_scheduler(
            name=lr_name,
            optimizer=optimizer,
            num_warmup_steps=500 * gradient_accumulation_steps,
            num_training_steps=max_train_steps * gradient_accumulation_steps,
            )
    
    elif lr_name == "lambdalinear":
        warm_up_steps = [10000]
        f_min = [ 1.]
        f_max = [1.]
        f_start = [1e-6]
        cycle_lengths = [10000000000000]

        lr_scheduler = LambdaLinearScheduler(
                    warm_up_steps=warm_up_steps,
                    f_min=f_min,
                    f_max=f_max,
                    f_start=f_start,
                    cycle_lengths=cycle_lengths,
                    )
    else:
        raise ValueError(f"Unknown lr_scheduler {lr_name}")
    
    # Prepare everything with our `accelerator`.
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
        )   
    weight_dtype = torch.float32

    # Move vae and unet to device and cast to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / gradient_accumulation_steps)
    if overrode_max_train_steps:
        max_train_steps = epochs * num_update_steps_per_epoch

    # Afterwards we recalculate our number of training epochs
    epochs = math.ceil(
        max_train_steps / num_update_steps_per_epoch)
    conf["num_train_epochs"] = epochs

    print(f"No of update steps per epoch after accelerator: {num_update_steps_per_epoch}, Max Train Steps: {max_train_steps}")

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers(f"{exp}")
        
    # Train!
    total_batch_size = bs * \
        accelerator.num_processes * gradient_accumulation_steps

    print(f"Total batch size: {total_batch_size}")
    print(f"Gradient accumulation steps: {gradient_accumulation_steps}")
    print(f"Number of processes: {accelerator.num_processes}")
    print(f"Number of training steps: {max_train_steps}")
    print(f"Number of epochs: {epochs}")
    print(f"Number of update steps per epoch: {num_update_steps_per_epoch}")

    return unet, vae, accelerator, optimizer, train_dataloader, val_dataloader, lr_scheduler, noise_scheduler, writer, logger, max_train_steps

def train_epoch(
        vae,
        unet,
        train_dataloader,
        accelerator,
        optimizer,
        lr_scheduler,
        noise_scheduler,
        progress_bar,
        logger,
        config,
        max_train_steps,
    ):

    """
    This function is used to train the model for one epoch. It loops through the training dataloader and updates the model parameters
    using the optimizer and scheduler.

    Args:
    - vae: Pre-trained VAE model from CompVis/stable-diffusion-v1-4.
    - unet: Unconditional U-Net model for training.
    - train_dataloader: training dataloader for model training.
    - accelerator: Accelerator object for distributed training.
    - optimizer: Optimizer for training the model.
    - lr_scheduler: Learning rate scheduler for optimizer.
    - noise_scheduler: Noise scheduler for adding noise to latents.
    - progress_bar: Progress bar for tracking the training progress.
    - logger: Accelerate logger for logging.
    - config: The configuration for the experiment.(/config/uldm.yaml)
    - max_train_steps: Maximum number of training steps.

    Returns:
    - train_loss: List of training loss for each step in the epoch.
    """

    root_path = config["output_dir"] + config["trials"]
    m_name = config['name']
    checkpoint_step = config["checkpoint_step"]
    save_image_step = config["save_image_step"]
    dtype = torch.float32
    global global_step
    unet.train()
    train_loss = []
    for _, batch in enumerate(train_dataloader):
        optimizer.zero_grad()  # Zero the gradients
        batch=batch[0]
        with accelerator.accumulate(unet):
            latents = vae.encode(batch.to(accelerator.device, dtype=dtype)).latent_dist.sample().detach()
            latents = latents * 0.18215
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
            timesteps = timesteps.long()
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
                progress_bar.update(1)
                global_step += 1
            optimizer.step()
            #lr_scheduler.step()
            lr_scheduler(global_step)
            logs = {"loss_per_step": loss.detach().item()}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            train_loss.append(loss.detach().item())

        if global_step % int(save_image_step) == 0:
            unwrapped_model = accelerator.unwrap_model(unet)
            pred_latents = produce_latents(config, unet=unwrapped_model, seed=42)
            pred_imgs = decode_img_latents(pred_latents,config)
            save_path = root_path + "/images"
            os.makedirs(save_path, exist_ok=True)
            for idx, pil_img in enumerate(pred_imgs):
                curr_step = accelerator.num_processes*global_step 
                img_path = os.path.join(save_path,f"image_{idx}_step-{curr_step}.png")
                pil_img.save(img_path)
                print(f"Saved: {img_path}")
                break
            del unwrapped_model

        if global_step == 0:
            save_path = root_path + "/images"
            os.makedirs(save_path, exist_ok=True)
            visualize_and_save_latent(latents[0], save_path=f'{save_path}/encoded_latent.png')
            img = decode_img_latents(latents, config)
            curr_step = accelerator.num_processes*global_step
            img_path = os.path.join(save_path, f"encoded_image_step-{curr_step}.png")
            img[0].save(img_path)
            print(f"Saved: {img_path}")

        if global_step % int(checkpoint_step) == 0:
            save_path = root_path + "/model"
            os.makedirs(save_path, exist_ok=True)
            curr_step = accelerator.num_processes*global_step
            save_path = os.path.join(save_path, f"{m_name}_step-{curr_step}.pth")
            unwrapped_model = accelerator.unwrap_model(unet)
            torch.save(unwrapped_model,save_path)
            del unwrapped_model

    return train_loss

def eval_epoch(
        vae,
        unet,
        val_dataloader,
        noise_scheduler,
        accelerator,
    ):
    """
    This function is used to evaluate the model on the validation dataset for one epoch. It loops through the validation dataloader
    and computes the validation loss.

    Args:
    - vae: Pre-trained VAE model from CompVis/stable-diffusion-v1-4.
    - unet: Unconditional U-Net model for training.
    - val_dataloader: validation dataloader for model evaluation.
    - noise_scheduler: Noise scheduler for adding noise to latents.
    - accelerator: Accelerator object for distributed training.

    Returns:
    - val_loss: List of validation loss for each step in the epoch.
    """
    unet.eval()
    val_loss = []
    for _, batch in enumerate(val_dataloader):
        batch = batch[0]
        with torch.no_grad():
            latents = vae.encode(batch.to(accelerator.device)).latent_dist.sample().detach()
            latents = latents * 0.18215
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
            timesteps = timesteps.long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            pred = unet(noisy_latents, timesteps).sample
            
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

            loss = F.mse_loss(pred.float(), target.float(), reduction="mean")
            val_loss.append(loss.detach().item())
    return val_loss


def train(    
        vae,
        unet,
        train_dataloader,
        val_dataloader,
        accelerator,
        optimizer,
        lr_scheduler,
        noise_scheduler,
        writer,
        logger,
        config,
        max_train_steps,
    ):
    """
    This function is used to train the model for multiple epochs. 

    Args:
    - vae: Pre-trained VAE model from CompVis/stable-diffusion-v1-4.
    - unet: Unconditional U-Net model for training.
    - train_dataloader: training dataloader for model training.
    - val_dataloader: validation dataloader for model evaluation.
    - accelerator: Accelerator object for distributed training.
    - optimizer: Optimizer for training the model.
    - lr_scheduler: Learning rate scheduler for optimizer.
    - noise_scheduler: Noise scheduler for adding noise to latents.
    - writer: Tensorboard SummaryWriter for logging.
    - logger: Accelerate logger for logging.
    - config: The configuration for the experiment.(/config/uldm.yaml)
    - max_train_steps: Maximum number of training steps.

    Returns:
    - avg_val_loss: Average validation loss over all epochs.
    """

    # Only show the progress bar once on each machine.
    root_path = config["output_dir"] + config["trials"]  
    epochs = config['num_train_epochs']
    m_name = config['name']
    progress_bar = tqdm(range(global_step, max_train_steps),
                        disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    
    #Train
    for epoch in range(epochs):
            train_loss = train_epoch(
                            vae,
                            unet,
                            train_dataloader,
                            accelerator,
                            optimizer,
                            lr_scheduler,
                            noise_scheduler,
                            progress_bar,
                            logger,
                            config,
                            max_train_steps,
                        )
            val_loss = eval_epoch(
                        vae,
                        unet,
                        val_dataloader,
                        noise_scheduler,
                        accelerator,
                    )
                
            avg_train_loss = sum(train_loss)/len(train_loss)
            avg_val_loss = sum(val_loss)/len(val_loss)
            writer.add_scalars(f'result/loss', {
            'train': avg_train_loss,
            'val': avg_val_loss,
            }, epoch)
 
    accelerator.wait_for_everyone()
    save_path = root_path + "/model"
    os.makedirs(save_path, exist_ok=True)
    save_path = os.path.join(save_path, f"{m_name}.pth")
    torch.save(accelerator.unwrap_model(unet),save_path)  
    del unet, vae, accelerator 
    torch.cuda.empty_cache()
    return avg_val_loss

def objective(config):
    """
    This function is used to train the U-LDM model for the specified number of epochs

    Args:
    - config: The configuration for the experiment.(/config/uldm.yaml)

    Returns:
    - Average validation loss over all epochs.
    """

    unet, vae, accelerator, optimizer, train_dataloader, val_dataloader, lr_scheduler, noise_scheduler, writer, logger, max_train_steps = setup(config)
    m_name = config['name']
    result = train(
                vae,
                unet,
                train_dataloader,
                val_dataloader,
                accelerator,
                optimizer,
                lr_scheduler,
                noise_scheduler,
                writer,
                logger,
                config,
                max_train_steps,
                )
    del unet, vae, accelerator 
    torch.cuda.empty_cache()
    return print(f"Training {m_name} is completed! with avg_val_loss: {result}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to the config file")
    args = parser.parse_args()
    with open(args.config, "r") as file:
        config = yaml.safe_load(file)
    objective(config)

if __name__ == "__main__":
    main()