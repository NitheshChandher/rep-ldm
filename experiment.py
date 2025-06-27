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
from eval.syn_dataset import syn_dataset
from eval.perturbe_dataset import perturbe_dataset
from eval.interpolate_dataset import interpolate_dataset
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
    real_path = args.eval_dir

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
    return metrics_df

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
        default="syn-dataset",
        required=True,
        help="Choose among perturbe-dataset or interpolate-dataset",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default='ffhq',
        required=True,
        help="Choose among ffhq or celeba or subset-imagenet",
    )

    parser.add_argument(
        "--eval_dir",
        type=str,
        default='./data/ffhq512/test',
        required=True,
        help="Path to the test images",
    ) 

    parser.add_argument(
        "--rep_dir",
        type=str,
        default=None,
        required=False,
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
    if args.method not in ['syn-dataset', 'perturbate-dataset', 'interpolate-dataset']:
        raise ValueError("Invalid method! Choose among perturbate-dataset or interpolate-dataset")
    if args.dataset not in ['ffhq', 'celeba', 'celeba-hq', 'imagenet-100']:
        raise ValueError("Invalid dataset! Choose among ffhq, celeba, 'celeba-hq', or imagenet-100")
    
    gen_path = os.path.join(args.save_path, args.model, args.dataset)
    if not os.path.exists(gen_path):
        os.makedirs(gen_path, exist_ok=True)
        print(f'Creating {args.dataset} folder in {args.save_path}/{args.model}')
    else:
        print(f'{args.dataset} folder already exists in {args.save_path}/{args.model}')
    
    if args.method == 'syn-dataset':
        syn_dataset(args)
        gen_path = os.path.join(args.save_path, args.model, args.dataset, args.method, str(args.num_inference_steps))
        metric_dict = torch_metrics(args, gen_path)
        print("Metric Info:", metric_dict)

    elif args.method == 'perturbate-dataset':
        perturbe_dataset(args)
        save_path = os.path.join(args.save_path, args.model, args.dataset, args.method)
        alpha = np.linspace(0, 1, 5)
        for lamda in alpha:
            gen_path = os.path.join(save_path, str(lamda))
            metric_dict = torch_metrics(args, gen_path)
            print("Metric Info:", metric_dict)
    
    elif args.method == 'interpolate-dataset':
        interpolate_dataset(args)
        gen_path = os.path.join(args.save_path, args.model, args.dataset, args.method)
        metric_dict = torch_metrics(args, gen_path)
        print("Metric Info:", metric_dict)

    else:
        raise ValueError("Invalid method! Choose among syn-dataset or interpolate-dataset")
    
    
if __name__ == "__main__":
    main()