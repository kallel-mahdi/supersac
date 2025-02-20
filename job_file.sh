#!/bin/bash -l
#SBATCH --gres=gpu:a40:1
#SBATCH --time=3:00:00
#SBATCH --cpus-per-task=16
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV

export HTTPS_PROXY=http://proxy:80
export WANDB_API_KEY=28996bd59f1ba2c5a8c3f2cc23d8673c327ae230
module load python/3.12-conda cuda cudnn
srun .venv/bin/python run_ppo_plus.py "$@"