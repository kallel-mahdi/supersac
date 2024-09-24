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
parser.add_argument('--seed',type=int,default=42) 
parser.add_argument('--env_name',type=str,default="Hopper-v5") 
parser.add_argument('--project_name',type=str,default="iclr_ppo_diagnosis") 
parser.add_argument('--gamma',type=float,default=0.99)
parser.add_argument('--max_steps',type=int,default=1_000_000) 
parser.add_argument('--normalize_reward',type=bool,default=True) 
parser.add_argument('--full_batch',type=str2bool, default=True)
parser.add_argument('--gae_lambda', type=float, default=0.95, help='the lambda for the general advantage estimation')
args = parser.parse_args()
##############################

np.random.seed(42)
seeds = list(np.random.randint(0,1e6,5))
configs = itertools.product(seeds,[args.env_name],[args.project_name],
                            [args.gamma],[args.max_steps],[args.normalize_reward],[args.full_batch],[args.gae_lambda])
            
for cfg in configs :
    
    import time
    import random

    # Add random time pause
    #time.sleep(random.uniform(0.1,3))

    command = f'sbatch job_file_ppo.sh\
    --seed  {cfg[0]} --env_name {cfg[1]} --project_name {cfg[2]} \
    --gamma {cfg[3]} --max_steps {cfg[4]} --normalize_reward {cfg[5]} --full_batch {cfg[6]} --gae_lambda {cfg[7]}\
    >./null 2>&1 & '
    
    print(command)

    subprocess.call(command,shell=True)
    