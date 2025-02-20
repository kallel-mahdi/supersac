import subprocess
import itertools
import argparse
import numpy as np


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42) 
parser.add_argument('--env_name', type=str, default="Hopper-v5") 
parser.add_argument('--project_name', type=str, default="RLC_PPO_OURS") 
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--max_steps', type=int, default=1_000_000) 
parser.add_argument('--normalize_reward', type=str2bool, default=True) 
parser.add_argument('--normalize_observation', type=str2bool, default=True) 
parser.add_argument('--full_batch', type=str2bool, default=False)
parser.add_argument('--gae_lambda', type=float, default=0.95, help='the lambda for the general advantage estimation')
# New arguments
parser.add_argument('--use_layer_norm', type=str2bool, default=False, help="whether to use layer normalization")
parser.add_argument('--hidden_dims', type=int, default=64, help="list of hidden dimensions")
parser.add_argument('--anneal_lr', type=str2bool, default=True, help="whether to anneal the learning rate")
args = parser.parse_args()
##############################

np.random.seed(42)
seeds = list(np.random.randint(0, 1e6, 5))
configs = itertools.product(
    seeds,
    #["InvertedDoublePendulum-v5", "Hopper-v5", "Walker2d-v5", "HalfCheetah-v5", "Ant-v5", "Humanoid-v5"],
    ["Hopper-v5","Walker2d-v5"],
    [args.project_name],
    [args.gamma],
    [args.max_steps],
    [args.normalize_reward],
    [args.full_batch],
    [args.gae_lambda],
    [args.use_layer_norm],
    [args.hidden_dims],
    [args.anneal_lr],
    [args.normalize_observation],
)
            
for cfg in configs:

    # Add random time pause
    # time.sleep(random.uniform(0.1,3))

    command = (
        f'sbatch job_file_ppo.sh '
        #f'python run_ppo.py '
        f'--seed {cfg[0]} --env_name {cfg[1]} --project_name {cfg[2]} '
        f'--gamma {cfg[3]} --max_steps {cfg[4]} --normalize_reward {cfg[5]} '
        f'--full_batch {cfg[6]} --gae_lambda {cfg[7]} '
        f'--use_layer_norm {cfg[8]} --hidden_dims {cfg[9]} '
        f'--anneal_lr {cfg[10]} '
        f'--normalize_observation {cfg[11]} '
        f'>./null 2>&1 &'
    )
    
    print(command)
    subprocess.call(command, shell=True)
