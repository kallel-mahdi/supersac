import subprocess
import itertools
import argparse
import numpy as np
import os
from typing import Dict, List, Any

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# Algorithm-specific default configurations
ALGORITHM_CONFIGS = {
    'ppo': {
        'script': 'run_ppo.py',
        'default_args': {
            'learning_rate': 3e-4,
            'num_steps': 2048,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'num_minibatches': 32,
            'update_epochs': 10,
            'clip_coef': 0.15,
            'ent_coef': 0.0,
            'vf_coef': 0.5,
            'max_grad_norm': 0.5,
            'hidden_dims': 64,
            'use_layer_norm': False,
            'normalize_reward': True,
            'normalize_observation': True,
        }
    },
        
    'ppo_ours': {
        'script': 'run_ppo.py',
        'default_args': {
            'learning_rate': 3e-4,
            'num_steps': 5000,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'num_minibatches': 32,
            'update_epochs': 20,
            'clip_coef': 0.2,
            'ent_coef': 0.0,
            'vf_coef': 0.5,
            'max_grad_norm': 0.5,
            'hidden_dims': 256,
            'use_layer_norm': True,
            'normalize_reward': True,
            'normalize_observation': True,
        }
    },
    'ppo_plus': {
        'script': 'run_ppo_plus.py',
        'default_args': {
            'gamma': 0.99,
            'entropy_coeff': 0.5,
            'num_critics': 2,
            'hidden_dims': 256,
            'temperature': 1.0,
            'clipping_ratio': 0.25,
            'gae_lambda': 0.5,
            'buffer_size': 50000,
            'policy_steps': 5000,
            'num_epochs': 10,
            'activation_fn': 'tanh',
            'use_layer_norm': True,
            'bound_actions': True,
        }
    },
    'sac': {
        'script': 'run_sac.py',
        'default_args': {
            'gamma': 0.99,
            'entropy_coeff': 0.5,
            'buffer_size': 1000_000
        }
    },
    'trpo': {
        'script': 'run_trpo.py',
        'default_args': {
            'learning_rate': 3e-4,
            'num_steps': 5120,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'num_minibatches': 32,
            'update_epochs': 10,
            'clip_coef': 0.2,
            'vf_coef': 0.5,
            'max_grad_norm': 0.5,
            'target_kl': 0.01,
            'anneal_lr': True,
            'norm_adv': True,
            'clip_vloss': True
        }
    }
}

# Common environments
DEFAULT_ENVS = [
    #"InvertedDoublePendulum-v5", 
    "Hopper-v5", 
    "Walker2d-v5", 
    "HalfCheetah-v5", 
    "Ant-v5", 
    "Humanoid-v5"
]

def create_parser():
    parser = argparse.ArgumentParser(description='Run RL experiments with multiple algorithms')
    
    # Core experiment settings
    parser.add_argument('--algorithm', type=str, choices=['ppo', 'ppo_plus', 'sac', 'ppo_ours', 'trpo'], 
                       required=True, help='Algorithm to run')
    parser.add_argument('--project_name', type=str, default='experiment_sweep',
                       help='WandB project name')
    parser.add_argument('--num_seeds', type=int, default=5,
                       help='Number of random seeds to generate')
    parser.add_argument('--base_seed', type=int, default=42,
                       help='Base seed for random seed generation')
    
    # Environment settings
    parser.add_argument('--env_names', nargs='+', default=DEFAULT_ENVS,
                       help='List of environment names')
    parser.add_argument('--max_steps', type=int, default=1000000,
                       help='Maximum training steps')
    
    # Execution settings
    parser.add_argument('--execution_mode', type=str, choices=['local', 'slurm'], 
                       default='local', help='How to execute jobs')
    parser.add_argument('--job_script', type=str, default='job_file_julia.sh',
                       help='SLURM job script to use')
    parser.add_argument('--dry_run', action='store_true',
                       help='Print commands without executing')
    
    # Parameter sweep settings
    parser.add_argument('--sweep_params', type=str, nargs='+', default=[],
                       help='Parameters to sweep (format: param_name:val1,val2,val3)')
    
    return parser

def parse_sweep_params(sweep_params: List[str]) -> Dict[str, List[Any]]:
    """Parse sweep parameters from command line format"""
    sweep_dict = {}
    for param in sweep_params:
        if ':' not in param:
            continue
        param_name, values_str = param.split(':', 1)
        values = []
        for val in values_str.split(','):
            val = val.strip()
            # Try to convert to appropriate type
            try:
                if '.' in val:
                    values.append(float(val))
                else:
                    values.append(int(val))
            except ValueError:
                if val.lower() in ['true', 'false']:
                    values.append(val.lower() == 'true')
                else:
                    values.append(val)
        sweep_dict[param_name] = values
    return sweep_dict

def generate_configs(algorithm: str, args: argparse.Namespace, sweep_params: Dict[str, List[Any]]):
    """Generate all parameter combinations"""
    
    # Get base configuration
    base_config = ALGORITHM_CONFIGS[algorithm]['default_args'].copy()
    
    # Generate seeds
    np.random.seed(args.base_seed)
    seeds = list(np.random.randint(0, 1e6, args.num_seeds))
    
    # Setup parameter combinations
    param_names = ['seed', 'env_name', 'project_name', 'max_steps']
    param_values = [seeds, args.env_names, [args.project_name], [args.max_steps]]
    
    # Add base algorithm parameters
    for param, value in base_config.items():
        if param not in sweep_params:
            param_names.append(param)
            param_values.append([value])
    
    # Add sweep parameters
    for param, values in sweep_params.items():
        param_names.append(param)
        param_values.append(values)
    
    # Generate all combinations
    configs = []
    for combination in itertools.product(*param_values):
        config = dict(zip(param_names, combination))
        configs.append(config)
    
    return configs

def format_command(algorithm: str, config: Dict[str, Any], execution_mode: str, job_script: str = None):
    """Format the execution command"""
    script = ALGORITHM_CONFIGS[algorithm]['script']
    
    # Determine virtual environment based on algorithm
    if algorithm in ['ppo', 'ppo_ours', 'trpo']:
        venv_path = '.venv'
    else:
        venv_path = '.venv_jax'
    
    # Build argument string
    args_str = []
    for param, value in config.items():
        if isinstance(value, bool):
            args_str.append(f'--{param} {str(value).lower()}')
        else:
            args_str.append(f'--{param} {value}')
    
    args_string = ' '.join(args_str)
    
    if execution_mode == 'slurm':
        command = f'sbatch {job_script} {venv_path}/bin/python {script} {args_string}'
    else:
        command = f'{venv_path}/bin/python {script} {args_string}'
    
    return command

def main():
    parser = create_parser()
    args = parser.parse_args()
    
    # Parse sweep parameters
    sweep_params = parse_sweep_params(args.sweep_params)
    
    # Generate configurations
    configs = generate_configs(args.algorithm, args, sweep_params)
    
    print(f"Generated {len(configs)} experiment configurations")
    print(f"Algorithm: {args.algorithm}")
    print(f"Environments: {args.env_names}")
    print(f"Seeds: {args.num_seeds}")
    if sweep_params:
        print(f"Sweep parameters: {sweep_params}")
    
    # Execute experiments
    for i, config in enumerate(configs):
        command = format_command(args.algorithm, config, args.execution_mode, args.job_script)
        
        if args.dry_run:
            print(f"[{i+1}/{len(configs)}] {command}")
        else:
            print(f"[{i+1}/{len(configs)}] Executing: {config['env_name']}, seed={config['seed']}")
            if args.execution_mode == 'slurm':
                subprocess.call(command + ' >./null 2>&1 &', shell=True)
            else:
                subprocess.call(command, shell=True)

if __name__ == "__main__":
    main()
