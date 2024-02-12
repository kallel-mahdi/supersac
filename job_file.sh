#!/bin/bash -l
#SBATCH --gres=gpu:a100:1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16

export HTTPS_PROXY=http://proxy:80
export WANDB_API_KEY=28996bd59f1ba2c5a8c3f2cc23d8673c327ae230
module load python/3.9-anaconda cuda cudnn


srun ./mujoco-v4 python3 run_mpd.py "$@"

    