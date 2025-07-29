# PPO Parameter Sweep Guide

This guide shows how to use `run_experiments.py` for PPO parameter sweeps.

## Quick Start

```bash
# Basic parameter sweep
python run_experiments.py \
    --algorithm ppo \
    --project_name my_ppo_sweep \
    --num_seeds 3 \
    --env_names "HalfCheetah-v5" "Hopper-v5" \
    --sweep_params learning_rate:1e-4,3e-4,1e-3 clip_coef:0.1,0.2,0.3 \
    --dry_run
```

## Key Arguments

- `--algorithm ppo`: Use PPO algorithm
- `--project_name`: Name for WandB logging
- `--num_seeds`: Number of random seeds (default: 5)
- `--env_names`: List of environments (default: all MuJoCo envs)
- `--max_steps`: Maximum training steps (default: 1M)
- `--sweep_params`: Parameters to sweep in format `param:val1,val2,val3`
- `--dry_run`: Show commands without executing
- `--execution_mode`: `local` or `slurm`

## Available PPO Parameters to Sweep

From the default PPO configuration:
- `learning_rate`: Learning rate (default: 3e-4)
- `num_steps`: Steps per update (default: 2000)
- `gamma`: Discount factor (default: 0.99)
- `gae_lambda`: GAE lambda (default: 0.95)
- `num_minibatches`: Number of minibatches (default: 32)
- `update_epochs`: PPO update epochs (default: 10)
- `clip_coef`: PPO clipping coefficient (default: 0.2)
- `ent_coef`: Entropy coefficient (default: 0.0)
- `vf_coef`: Value function coefficient (default: 0.5)
- `max_grad_norm`: Gradient clipping (default: 0.5)
- `hidden_dims`: Network hidden dimensions (default: 64)
- `use_layer_norm`: Use layer normalization (default: false)
- `normalize_reward`: Normalize rewards (default: true)
- `normalize_observation`: Normalize observations (default: true)

## Common Sweep Examples

### Learning Rate and Clipping
```bash
--sweep_params learning_rate:1e-4,3e-4,1e-3 clip_coef:0.1,0.2,0.3
```

### Architecture
```bash
--sweep_params hidden_dims:64,128,256 use_layer_norm:true,false
```

### Training Dynamics
```bash
--sweep_params num_steps:1000,2000,4000 update_epochs:5,10,20
```

### Regularization
```bash
--sweep_params ent_coef:0.0,0.01,0.1 vf_coef:0.25,0.5,1.0
```

## Number of Experiments

Total experiments = (num_sweep_param_combinations) × (num_seeds) × (num_environments)

Example: 3 learning rates × 2 clip values × 5 seeds × 2 envs = 60 experiments

## Execution

Remove `--dry_run` to actually run experiments:
```bash
python run_experiments.py --algorithm ppo --project_name real_sweep --sweep_params learning_rate:3e-4,1e-3
```

For SLURM clusters:
```bash
python run_experiments.py --algorithm ppo --execution_mode slurm --job_script job_file_julia.sh --sweep_params learning_rate:1e-4,3e-4
```

## Example Output

The script will generate all parameter combinations and show:
```
Generated 8 experiment configurations
Algorithm: ppo
Environments: ['HalfCheetah-v5']
Seeds: 2
Sweep parameters: {'learning_rate': ['3e-4', '1e-3'], 'clip_coef': [0.1, 0.2]}
[1/8] python run_ppo.py --seed 121958 --env_name HalfCheetah-v5 ...
``` 