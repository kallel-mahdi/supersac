#!/bin/bash
# YAML Configuration Translation:
# nodes: 1                      # Default (not explicitly set)
# gpus_per_node: 1             # From --gres=gpu:1
# gres: "gpu:1"                # From --gres=gpu:1  
# cpus_per_task: 8             # From -c 8
# time: "04:00:00"             # From --time=04:00:00
# mem_gb: 8                    # From --mem=8G
# tmpfs: "5G"                  # From --tmp=5G
# account: null                # Not specified in script
# qos: null                    # Not specified in script
# partition: "standard"        # From -p standard
# export: "NONE"               # From --export=NONE

#SBATCH -J RL_Experiment         # Name your job
#SBATCH --time=08:00:00          # Time limit in the form hh:mm:ss
#SBATCH -c 8                     # We want to use 8 cores
#SBATCH --mem=8G                 # Job needs 8GB of memory
#SBATCH -p standard              # Select standard partition
#SBATCH --gres=gpu:1             # We need 1 GPU
#SBATCH --tmp=5G                 # We need 5G of /tmp space
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV
export WANDB_API_KEY=28996bd59f1ba2c5a8c3f2cc23d8673c327ae230

# Execute the full command passed as arguments
srun "$@"

