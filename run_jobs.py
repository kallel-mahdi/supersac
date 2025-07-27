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
parser.add_argument('--project_name',type=str,default="AAAI_ERLANGEN") 
parser.add_argument('--activation_fn',type=str,default="tanh") 



parser.add_argument('--on_policy_data',type=str2bool,default=False) 
parser.add_argument('--min_target',type=str2bool,default=False) 
parser.add_argument('--use_layer_norm',type=str2bool,default=True) 
parser.add_argument('--bound_actions',type=str2bool,default=True) 
parser.add_argument('--temperature',type=float,default=1.)
parser.add_argument('--num_critics',type=int,default=2)
parser.add_argument('--gae_lambda',type=float,default=0.5) 

parser.add_argument('--max_episode_steps',type=int,default=1000)
parser.add_argument('--clipping_ratio',type=float,default=0.2)
parser.add_argument('--gamma',type=float,default=0.99)

parser.add_argument('--policy_steps',type=int,default=5000) 
parser.add_argument('--buffer_size',type=int,default=100_000) 
parser.add_argument('--spo_loss',type=str2bool,default=False)

args = parser.parse_args()

##############################

np.random.seed(args.seed)
seeds = list(np.random.randint(0,1e6,10))
configs = itertools.product(seeds,["InvertedPendulum-v5","Hopper-v5","Walker2d-v5","HalfCheetah-v5","Ant-v5","Humanoid-v5"],[args.project_name],
                            [args.gamma],[args.temperature],[args.policy_steps],
                            [args.num_critics],[args.activation_fn],[args.gae_lambda],[args.buffer_size],
                           [args.use_layer_norm],[args.bound_actions],[args.on_policy_data],[args.clipping_ratio],[args.min_target],[args.spo_loss])
            
for cfg in configs :
    
    import time
    import random

    # Add random time pause
    # time.sleep(random.uniform(0.1,3))

    command = f'sbatch job_file.sh\
    --seed  {cfg[0]} --env_name {cfg[1]} --project_name {cfg[2]} \
    --gamma {cfg[3]} --temperature {cfg[4]} --policy_steps {cfg[5]} \
    --num_critics {cfg[6]} --activation_fn {cfg[7]} \
    --gae_lambda {cfg[8]} --buffer_size {cfg[9]} \
    --use_layer_norm {cfg[10]} --bound_actions {cfg[11]} --on_policy_data {cfg[12]} \
    --clipping_ratio {cfg[13]} --min_target {cfg[14]} --spo_loss {cfg[15]} \
    >./null 2>&1 & '
    
    print(command)

    subprocess.call(command,shell=True)
    
