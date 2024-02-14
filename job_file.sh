#!/bin/bash -l
#SBATCH --gres=gpu:40:1
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=1

export HTTPS_PROXY=http://proxy:80
export WANDB_API_KEY=28996bd59f1ba2c5a8c3f2cc23d8673c327ae230
module load python/3.9-anaconda cuda cudnn

srun myenv/bin/python/ run_mpd.py "$@"

    