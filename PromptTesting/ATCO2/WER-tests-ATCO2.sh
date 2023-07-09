#!/bin/bash

#SBATCH --job-name=WER-prmpt
#SBATCH --partition=compute
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --account=research-ae-co

#SBATCH --mail-type=ALL
#SBATCH --mail-user=jlpmvandoorn

# Load modules:
module load 2022r2
module load openmpi
module load miniconda3
module load git
module load ffmpeg

# Set conda env:
unset CONDA_SHLVL
source "$(conda info --base)/etc/profile.d/conda.sh"

conda activate /home/junzisun/env/jan
python WER-tests-ATCO2.py
conda deactivate

