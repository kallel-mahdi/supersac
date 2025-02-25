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
parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
parser.add_argument('--project_name', type=str, default="DMC_DOG")
args = parser.parse_args()
##############################

np.random.seed(42)
seeds = list(np.random.randint(0,1e6,5))
configs = itertools.product(seeds,["walk","stand","trot","run"],[args.gamma],[args.project_name])
            
for cfg in configs :
    
    command = f'sbatch job_file_sac.sh\
    --seed  {cfg[0]} --env_name {cfg[1]} --gamma {cfg[2]} --project_name {cfg[3]}\
    >./null 2>&1 & '
    
    print(command)

    subprocess.call(command,shell=True)
    