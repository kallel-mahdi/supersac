import subprocess
import itertools
import argparse
import numpy as np
from jaxrl_m.utils import str2bool

parser = argparse.ArgumentParser()
parser.add_argument('--seed',type=int,default=42) 
parser.add_argument('--env_name',type=str,default="HalfCheetah-v4") 
parser.add_argument('--project_name',type=str,default="delete") 
parser.add_argument('--max_steps',type=int,default=1e6)
parser.add_argument('--num_critics',type=int,default=5) 
parser.add_argument('--num_rollouts',type=int,default=5) 
parser.add_argument('--backup_entropy',type=str2bool,default=True) 
parser.add_argument('--discount_actor',type=str2bool,default=True) 

args = parser.parse_args()
##############################

np.random.seed(42)
seeds = list(np.random.randint(0,1e6,5))

configs = itertools.product(seeds,[args.env_name],[args.project_name],
                            [args.num_critics],[args.num_rollouts],[args.backup_entropy],[args.discount_actor])
            
for cfg in configs :
    
    command = f'sbatch job_file.sh --seed {cfg[0]} --env_name {cfg[1]} --project_name {cfg[2]} --num_critics {cfg[3]} --num_rollouts {cfg[4]} --backup_entropy {cfg[5]} --discount_actor {cfg[6]} >./null 2>&1 & '
    
    print(command)

    subprocess.call(command,shell=True)
    