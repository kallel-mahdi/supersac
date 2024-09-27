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
parser.add_argument('--state_dim',type=int,default=4) 
parser.add_argument('--a_dim',type=int,default=2) 

args = parser.parse_args()
##############################

np.random.seed(42)
seeds = list(np.random.randint(0,1e6,10))
configs = itertools.product(seeds,[args.state_dim],[args.a_dim])
            
for cfg in configs :
    
    import time
    import random

    # Add random time pause
    #time.sleep(random.uniform(0.1,3))

    command = f'sbatch job_file_lqr.sh\
    --seed  {cfg[0]} --state_dim {cfg[1]} --a_dim {cfg[2]} \
    
    >./null 2>&1 & '
    
    print(command)

    subprocess.call(command,shell=True)
    