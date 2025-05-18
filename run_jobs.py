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
parser.add_argument('--env_name',type=str,default="Walker2d-v5")
parser.add_argument('--project_name',type=str,default="DMC_DOG") 

parser.add_argument('--activation_fn',type=str,default="tanh") 
parser.add_argument('--discount_entropy',type=str2bool,default=False)


parser.add_argument('--on_policy_data',type=str2bool,default=False) 
parser.add_argument('--min_target',type=str2bool,default=False) 
parser.add_argument('--use_layer_norm',type=str2bool,default=True) 
parser.add_argument('--bound_actions',type=str2bool,default=True) 
parser.add_argument('--temperature',type=float,default=1.)
parser.add_argument('--num_critics',type=int,default=2)
parser.add_argument('--gae_lambda',type=float,default=0.5) 

parser.add_argument('--max_episode_steps',type=int,default=1000)
parser.add_argument('--clipping_ratio',type=float,default=0.25)
parser.add_argument('--gamma',type=float,default=0.99)

parser.add_argument('--policy_steps',type=int,default=5000) 
parser.add_argument('--buffer_size',type=int,default=50_000) 

args = parser.parse_args()

##############################

np.random.seed(args.seed)
seeds = list(np.random.randint(0,1e6,5))
configs = itertools.product(seeds,["Hopper-v5","Walker2d-v5","HalfCheetah-v5","Ant-v5"],[args.project_name],
                            [args.gamma],[args.temperature],[args.policy_steps],
                            [args.num_critics],[args.activation_fn],[args.discount_entropy],[args.gae_lambda],[args.buffer_size],
                            [args.max_episode_steps],[args.use_layer_norm],[args.bound_actions],[args.on_policy_data],[args.clipping_ratio],[args.min_target])
            
for cfg in configs :
    
    import time
    import random

    # Add random time pause
    # time.sleep(random.uniform(0.1,3))

    command = f'sbatch job_file.sh\
    --seed  {cfg[0]} --env_name {cfg[1]} --project_name {cfg[2]} \
    --gamma {cfg[3]} --temperature {cfg[4]} --policy_steps {cfg[5]} \
    --num_critics {cfg[6]} --activation_fn {cfg[7]} --discount_entropy {cfg[8]} \
    --gae_lambda {cfg[9]} --buffer_size {cfg[10]} --max_episode_steps {cfg[11]} \
    --use_layer_norm {cfg[12]} --bound_actions {cfg[13]} --on_policy_data {cfg[14]} --clipping_ratio {cfg[15]} --min_target {cfg[16]}\
    >./null 2>&1 & '
    
    print(command)

    subprocess.call(command,shell=True)
    
