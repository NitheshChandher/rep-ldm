#!/bin/bash

# This script extracts FFHQ256 dataset and representations, then launches DINO-LDM training.

#SBATCH -A berzelius-2025-21
#SBATCH --mail-type=ALL
#SBATCH --mail-user=nithesh.chandher.karthikeyan@liu.se
#SBATCH --gpus 4
#SBATCH -t 3-00:00:00

# Define paths
export ZIP_DATA=/proj/dcdl/users/gabei62/rep-ldm/data/imagenet100.zip
export ZIP_REP=/proj/dcdl/users/gabei62/rep-ldm/rep/imagenet100-clip.zip
export SCRATCH_DIR=/scratch/local/rep-ldm/clip/data
export SCRATCH_REP_DIR=/scratch/local/rep-ldm/clip/rep

# Create scratch directories
mkdir -p $SCRATCH_DIR $SCRATCH_REP_DIR

# Extract ZIP files
echo "Extracting dataset ZIP..."
unzip -q $ZIP_DATA -d $SCRATCH_DIR
echo "Extracting representation ZIP..."
unzip -q $ZIP_REP -d $SCRATCH_REP_DIR

# Load Anaconda and activate environment
module load Anaconda
conda activate di

# Navigate to DINO-LDM directory
cd /proj/dcdl/users/gabei62/rep-ldm

# Start training
accelerate launch rep-ldm.py --config="configs/imagenet-100/clip.yaml"