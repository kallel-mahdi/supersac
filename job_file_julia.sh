#!/bin/bash
#SBATCH -J StandardJob           # Name your job
#SBATCH --time=00:35:00           # Time limit in the form hh:mm:ss
#SBATCH -c 2                     # We want to use 2 cores
#SBATCH --mem=2G                 # Job needs 2GB of memory (default!)
#SBATCH -p standard              # Select standard partion
#SBATCH --gres=gpu:1             # We need 1 GPU
#SBATCH --tmp=5G                 # We need 5G of /tmp space
#SBATCH --export=NONE


unset SLURM_EXPORT_ENV
export WANDB_API_KEY=28996bd59f1ba2c5a8c3f2cc23d8673c327ae230
srun .venv3/bin/python run_ppo_plus.py "$@"

