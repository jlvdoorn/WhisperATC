#!/bin/bash

#SBATCH --job-name=DataCreationTest
#SBATCH --partition=memory
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=750G
#SBATCH --account=education-ae-msc-ae

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

conda activate whisper
echo "Starting Job"
echo "00-Downloading data"
bash 00-Download.sh
echo "01-Preparing data"
python 01-Prepare.py
echo "--Job Finished--"
conda deactivate
