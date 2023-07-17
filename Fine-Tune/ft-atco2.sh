#!/bin/bash

#SBATCH --job-name=FT-A2
#SBATCH --partition=gpu
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=4
#SBATCH --mem=128GB
#SBATCH --account=research-ae-co

#SBATCH --mail-type=ALL
#SBATCH --mail-user=jlpmvandoorn

# Load modules:
module load 2022r2
module load openmpi
module load miniconda3
module load git
module load ffmpeg
module load cuda
module load nccl
module load git-lfs

git lfs install

# Set conda env:
unset CONDA_SHLVL
source "$(conda info --base)/etc/profile.d/conda.sh"

conda activate /home/junzisun/env/jan
deepspeed FineTuning-ATCO2.py
conda deactivate
