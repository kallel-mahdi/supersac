#!/bin/bash -l
#SBATCH --gres=gpu:a40:1
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=16
#SBATCH --export=NONE
#SBATCH -J RL_Experiment

unset SLURM_EXPORT_ENV

export HTTPS_PROXY=http://proxy:80
export WANDB_API_KEY=28996bd59f1ba2c5a8c3f2cc23d8673c327ae230
module load python/3.12-conda cuda cudnn

# Execute the full command passed as arguments
srun "$@"

