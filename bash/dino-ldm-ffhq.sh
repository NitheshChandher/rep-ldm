#!/bin/bash

#SBATCH -A your-project-id         # Replace with your actual SLURM account/project
#SBATCH --mail-type=ALL
#SBATCH --mail-user=your@email.com  # Replace with your actual email
#SBATCH --gpus 4
#SBATCH -t 3-00:00:00

# Define paths
export ZIP_DATA=/path/to/your/dataset.zip                 # Replace with actual dataset ZIP path
export ZIP_REP=/path/to/your/representations.zip           # Replace with actual representations ZIP path
export SCRATCH_DIR=/scratch/local/project/images           # Replace with actual scratch directory for images
export SCRATCH_REP_DIR=/scratch/local/project/representations  # Replace with actual scratch directory for representations

# Create scratch directories
mkdir -p $SCRATCH_DIR $SCRATCH_REP_DIR

# Extract ZIP files
echo "Extracting dataset ZIP..."
unzip -q $ZIP_DATA -d $SCRATCH_DIR

echo "Extracting representation ZIP..."
unzip -q $ZIP_REP -d $SCRATCH_REP_DIR

# Load Anaconda and activate environment
module load Anaconda
conda activate rep-ldm      # Replace with your actual conda environment name

# Navigate to project directory
cd /path/to/your/project           # Replace with the path to your project

# Start training
accelerate launch rep-ldm.py --config="configs/ffhq/dino.yaml"  # Adjust config path as needed
