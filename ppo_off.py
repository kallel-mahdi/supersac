# %%

import argparse
import logging
import os



import os
from collections import deque
from functools import partial

import gymnasium as gym
import jax
import numpy as np
import tqdm
import wandb
from jax import config

from jaxrl_m.dataset import ActorReplayBuffer, ReplayBuffer

from jaxrl_m.evaluation import (EpisodeMonitor, evaluate, flatten,
                                supply_rng)
from jaxrl_m.rollout import (rollout_policy, rollout_policy2)
from jaxrl_m.utils import flatten_rollouts
from jaxrl_m.wandb import default_wandb_config, get_flag_dict, setup_wandb
from jaxrl_m.ppo_plus2 import *
from jaxrl_m.utils import *
from jaxrl_m.normalize import *
from jaxrl_m.dmc import DMCGym
import random
from dm_control import suite

#logging.basicConfig(level=logging.DEBUG)  # Ignore warnings and below (INFO, WARNING, etc.)


# Set env variables
os.environ["WANDB_API_KEY"]="28996bd59f1ba2c5a8c3f2cc23d8673c327ae230"
os.environ["WANDB__SERVICE_WAIT"] = str(1800)
os.environ['PYTHONHASHSEED'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['XLA_FLAGS']='--xla_gpu_deterministic_ops=true'
config.update("jax_log_compiles", True)




##############################
parser = argparse.ArgumentParser()

parser.add_argument('--seed',type=int,default=42) 

parser.add_argument('--algo_name', type=str, default='superppo', help='the name of the RL algorithm')
parser.add_argument('--project_name',type=str,default="single_exp") 
parser.add_argument('--env_name',type=str,default="Walker2d-v5") 
parser.add_argument('--max_steps',type=int,default=1_000_000) 
parser.add_argument('--max_episode_steps',type=int,default=1000) 
parser.add_argument('--gamma',type=float,default=0.995)
parser.add_argument('--entropy_coeff',type=float,default=1.) 

parser.add_argument('--num_critics',type=int,default=5)
parser.add_argument('--hidden_dims',type=int,default=256) 
parser.add_argument('--critic_lr',type=float,default=3e-4) 
parser.add_argument('--actor_lr',type=float,default=3e-4) 
parser.add_argument('--temp_lr',type=float,default=3e-4) 
parser.add_argument('--momentum',type=float,default=0.) 
parser.add_argument('--b2',type=float,default=0.999) 
parser.add_argument('--temperature',type=float,default=1.0) 

parser.add_argument('--discount_actor',type=str2bool,default=False)
parser.add_argument('--discount_entropy',type=str2bool,default=False) 
parser.add_argument('--on_policy_data',type=str2bool,default=False)
parser.add_argument('--adaptive_critics',type=str2bool,default=False) 
parser.add_argument('--min_target',type=str2bool,default=False)
parser.add_argument('--use_layer_norm',type=str2bool,default=True)

parser.add_argument('--clipping_ratio',type=float,default=0.25) 
parser.add_argument('--gae_lambda',type=float,default=0.) 

parser.add_argument('--episode_based',type=str2bool,default=False) 
parser.add_argument('--minibatch',type=str2bool,default=True) 
parser.add_argument('--buffer_size',type=int,default=50_000) 
parser.add_argument('--policy_steps',type=int,default=5_000) 
parser.add_argument('--num_epochs',type=int,default=50) 
parser.add_argument('--batch_size',type=int,default=250)
parser.add_argument('--num_actor_updates',type=int,default=1)
parser.add_argument('--activation_fn',type=str,default='relu')
parser.add_argument('--stable_scheme',type=str2bool,default=True)
parser.add_argument('--bound_actions',type=str2bool,default=True)

args = parser.parse_args()
print(args)



random.seed(args.seed)
np.random.seed(args.seed)
jax_rng = jax.random.PRNGKey(args.seed)



#jax.config.update("jax_disable_jit", True)
#config.update("jax_debug_nans", True)
# config.update("jax_default_matmul_precision", "highest")
#config.update("jax_log_compiles", True)

def train(args):
    
    
        
    if args.env_name in ["Humanoid-v5","HumanoidStandup-v5","walk","stand","trot","run"]: args.max_steps = 5_000_000
    elif args.env_name == "InvertedDoublePendulum-v5": args.max_steps = 500_000
    if args.on_policy_data: args.buffer_size = args.policy_steps
    
    max_steps = args.max_steps
    log_interval = 20000
    n_grads = 0

    wandb_config = {
        'project': args.project_name,
        'name':None,
        'hyperparam_dict':args.__dict__,
        }
    wandb_run = setup_wandb(**wandb_config)
    
    if args.env_name in ["walk","stand","trot","run"]:
        env = DMCGym("dog",args.env_name)
        eval_env = DMCGym("dog",args.env_name)
    
    else : 
    
        env = gym.wrappers.RecordEpisodeStatistics(gym.make(args.env_name, max_episode_steps=args.max_episode_steps))
        eval_env = gym.wrappers.RecordEpisodeStatistics(gym.make(args.env_name,max_episode_steps=1000))
    
    
  
    

    example_transition = dict(
        observations=env.observation_space.sample(),
        actions=env.action_space.sample(),
        rewards=0.0,
        masks=1.0,
        truncateds = 0.0,
        next_observations=env.observation_space.sample(),
        pre_actions = env.action_space.sample(),
        discounts=1.0,
        log_probs=0.,
    )

    replay_buffer = ReplayBuffer.create(example_transition, size=int(args.buffer_size))
    actor_buffer = ActorReplayBuffer.create(example_transition, size=args.policy_steps)

    agent = create_learner(args.seed,
                        
                    observations=example_transition['observations'][None],
                    actions =example_transition['actions'][None],
                    max_steps=max_steps,
                    discount=args.gamma,
                    discount_actor=args.discount_actor,
                    min_target=args.min_target,
                    discount_entropy=args.discount_entropy,
                    adaptive_critics=args.adaptive_critics,
                    num_critics= args.num_critics,
                    entropy_coeff=args.entropy_coeff,
                    temperature=args.temperature,
                    actor_lr=args.actor_lr,
                    critic_lr=args.critic_lr,
                    temp_lr=args.temp_lr,
                    momentum=args.momentum,
                    b2=args.b2,
                    clipping_ratio=args.clipping_ratio,
                    num_actor_updates=args.num_actor_updates,
                    actor_hidden_dims=(128,128),
                    critic_hidden_dims=(128,128),
                    use_layer_norm= args.use_layer_norm,
                    gae_lambda=args.gae_lambda,
                    minibatch = args.minibatch,
                    activation_fn = args.activation_fn,
                    state_dependent_std=True,
                    tanh_squash_distribution= not args.stable_scheme and args.bound_actions,## This should be false
                    tanh_squash_actions= args.stable_scheme and args.bound_actions, ## This should be true
                    #**FLAGS.config
                    )

    exploration_metrics = dict()
    exploration_rng = jax.random.PRNGKey(0)
    i = 0
    unlogged_steps,cached_steps = 0,0
    
    rollout_fn = rollout_policy if args.episode_based else rollout_policy2

    with tqdm.tqdm(total=max_steps) as pbar:
        
        while (i < max_steps):
                
                logging.debug('policy rollout')
                if args.on_policy_data: replay_buffer = replay_buffer.reset()
                replay_buffer,actor_buffer,policy_return,undisc_policy_return,num_steps = rollout_fn(
                                                                        agent,env,exploration_rng,
                                                                        replay_buffer,actor_buffer,eval=False,
                                                                        discount = args.gamma,max_steps=args.policy_steps)
                         
                
                unlogged_steps += num_steps
                cached_steps += num_steps
                
                i+=num_steps
                pbar.update(int(num_steps))
                
                
                    
                
                for _ in range(args.num_epochs):
                    ### Get all transitions
                    transitions = replay_buffer.get_all()
                    
                    # Update agent for one epoch (JIT compiled)
                    agent, actor_update_infos = agent.update_epoch(
                        transitions, args.batch_size, exploration_rng
                    )
                    
                    # Update exploration RNG
                    exploration_rng = jax.random.split(exploration_rng)[0]
                    
                    critic_update_info = {}
                    actor_update_info = {}  # Could extract from actor_update_infos if needed
                
                update_info = {**critic_update_info, **actor_update_info}
                
                ### Log training info ###
                exploration_metrics = {f'exploration/disc_return': policy_return}
                train_metrics = {f'training/{k}': v for k, v in update_info.items()}
                train_metrics['training/undisc_return'] = undisc_policy_return
                
                wandb.log(train_metrics, step=int(i),commit=False)
                wandb.log(exploration_metrics, step=int(i),commit=False)
            
                ### Log evaluation info ###
                
                if unlogged_steps >= log_interval:
                    
                    _,_,policy_return,undisc_policy_return,num_steps = rollout_policy(
                                                                    agent,eval_env,exploration_rng,
                                                                    None,None,eval=True,
                                                                    discount = args.gamma,max_rollouts=10)
                    eval_metrics = {"policy_return": policy_return,"undisc_policy_return": undisc_policy_return}
                    print(eval_metrics)
                    
                    eval_metrics = {f'evaluation/{k}': v for k, v in eval_metrics.items()}
                    eval_metrics['n_grads']=int(n_grads)

                    eval_step = i
                    wandb.log(eval_metrics, step=int(eval_step),commit=True)
                    unlogged_steps = 0
        
    wandb_run.finish()

train(args)
#%%
