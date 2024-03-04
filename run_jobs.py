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
parser.add_argument('--env_name',type=str,default="Walker2d-v4") 
parser.add_argument('--project_name',type=str,default="Walker2d_nips") 
parser.add_argument('--gamma',type=float,default=0.99)
parser.add_argument('--max_steps',type=int,default=1_000_000) 
parser.add_argument('--num_rollouts',type=int,default=5) 
parser.add_argument('--num_critics',type=int,default=5) 
parser.add_argument('--adaptive_critics',type=str2bool,default=True) 
parser.add_argument('--discount_entropy',type=str2bool,default=True) 
parser.add_argument('--discount_actor',type=str2bool,default=True) 

args = parser.parse_args()
##############################

np.random.seed(42)
seeds = list(np.random.randint(0,1e6,5))


configs = itertools.product(seeds,[args.env_name],[args.project_name],
                            [args.gamma],[args.max_steps],[5,8],
                            [args.num_critics],[args.adaptive_critics],[args.backup_entropy],[True,False])
            
            
for cfg in configs :
    
    import time
    import random

    # Add random time pause
    time.sleep(random.uniform(0.1,3))

    command = f'sbatch job_file.sh\
    --seed  {cfg[0]} --env_name {cfg[1]} --project_name {cfg[2]} \
    --gamma {cfg[3]} --max_steps {cfg[4]} --num_rollouts {cfg[5]} \
    --num_critics {cfg[6]} --adaptive_critics {cfg[7]} --discount_entropy {cfg[8]} --discount_actor {cfg[9]} \
    >./null 2>&1 & '
    
    print(command)

    subprocess.call(command,shell=True)
    