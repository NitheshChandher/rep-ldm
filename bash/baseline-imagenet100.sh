#!/bin/bash

#SBATCH -A your-project-id         # Replace with your actual SLURM account/project
#SBATCH --mail-type=ALL
#SBATCH --mail-user=your@email.com  # Replace with your actual email
#SBATCH --gpus 2
#SBATCH -t 3-00:00:00

# Define paths
export ZIP_DATA=/path/to/your/dataset.zip        # Replace with actual dataset ZIP path
export SCRATCH_DIR=/scratch/local/project/data   # Replace with actual scratch directory

# Create scratch directories
mkdir -p $SCRATCH_DIR 

# Extract ZIP files
echo "Extracting dataset ZIP..."
unzip -q $ZIP_DATA -d $SCRATCH_DIR

# Load Anaconda and activate environment
module load Anaconda
conda activate rep-ldm      # Replace with your actual conda environment name

# Navigate to project directory
cd /path/to/your/project           # Replace with the path to your project

# Start training
accelerate launch baseline.py --config="configs/imagenet-100/baseline.yaml"  # Adjust path/config as needed
