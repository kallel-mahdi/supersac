# %%

import argparse
import logging
import os

import jax.numpy as jnp


import os


import jax
import numpy as np
import tqdm
import wandb


from jaxrl_m.dataset import ActorReplayBuffer, ReplayBuffer

from jaxrl_m.rollout import rollout_policy
from jaxrl_m.wandb import setup_wandb
from jaxrl_m.ppo_plus import SuperPPOConfig, create_learner
from jaxrl_m.utils import *
from jaxrl_m.normalize import *
from cosine_distance_ppo import create_experimental_agents, evaluate_experimental_agents_gradients, compute_cosines
import random



# Set env variables
os.environ["WANDB_API_KEY"]="28996bd59f1ba2c5a8c3f2cc23d8673c327ae230"
os.environ["WANDB__SERVICE_WAIT"] = str(1800)
os.environ['PYTHONHASHSEED'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['XLA_FLAGS']='--xla_gpu_deterministic_ops=true'


parser = argparse.ArgumentParser()

parser.add_argument('--seed',type=int,default=42) 

parser.add_argument('--algo_name', type=str, default='superppo', help='the name of the RL algorithm')
parser.add_argument('--project_name',type=str,default="single_exp_off") 
parser.add_argument('--env_name',type=str,default="Ant-v5") 
parser.add_argument('--evaluate_grad',type=bool,default=True)

parser.add_argument('--max_steps',type=int,default=1_100_000) 
parser.add_argument('--gamma',type=float,default=0.99)
parser.add_argument('--entropy_coeff',type=float,default=1.) 

parser.add_argument('--num_critics',type=int,default=2)
parser.add_argument('--hidden_dims',type=int,default=256) 
parser.add_argument('--temperature',type=float,default=0.) 

parser.add_argument('--on_policy_data',type=str2bool,default=False)
parser.add_argument('--min_target',type=str2bool,default=False)
parser.add_argument('--use_layer_norm',type=str2bool,default=True)
parser.add_argument('--spo_loss',type=str2bool,default=False)

parser.add_argument('--clipping_ratio',type=float,default=0.25) 
parser.add_argument('--gae_lambda',type=float,default=0.5) 

parser.add_argument('--buffer_size',type=int,default=50_000) 
parser.add_argument('--policy_steps',type=int,default=5000) 
parser.add_argument('--num_epochs',type=int,default=10) 
parser.add_argument('--num_critic_updates',type=int,default=5000)
parser.add_argument('--activation_fn',type=str,default='tanh')
parser.add_argument('--stable_scheme',type=str2bool,default=True)
parser.add_argument('--bound_actions',type=str2bool,default=True)

args = parser.parse_args()
print(args)


random.seed(args.seed)
np.random.seed(args.seed)
jax_rng = jax.random.PRNGKey(args.seed)





def train(args):

    log_interval = 20000
    n_grads = 0

    wandb_config = {
        'project': args.project_name,
        'name':None,
        'hyperparam_dict':args.__dict__,
        }
    wandb_run = setup_wandb(**wandb_config)
    
    # Create environments using the utility function
    env, eval_env = create_environments(args.env_name)
    max_steps = args.max_steps
    if args.on_policy_data: 
        args.buffer_size = args.policy_steps
    
    
  
    

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
    reference_buffer = ReplayBuffer.create(example_transition, size=int(args.buffer_size))

    # Create configuration from command line arguments
    config = SuperPPOConfig.from_args(args)
    
    # Create agent with clean, organized config
    agent = create_learner(
        config=config,
        observations=example_transition['observations'][None],
        actions=example_transition['actions'][None]
    )
    
    experimental_agents = create_experimental_agents(config, example_transition)
    
 


    
    

    exploration_metrics = dict()
    exploration_rng = jax.random.PRNGKey(0)
    i = 0
    unlogged_steps,cached_steps = 0,0
    
    
    with tqdm.tqdm(total=max_steps) as pbar:
        
        while (i < max_steps):
                
                logging.debug('policy rollout')
                if args.on_policy_data: replay_buffer = replay_buffer.reset()
                replay_buffer,actor_buffer,policy_return,undisc_policy_return,num_steps = rollout_policy(
                                                                        agent,env,exploration_rng,
                                                                        discount = args.gamma,max_steps=args.policy_steps,
                                                                        replay_buffer=replay_buffer,actor_buffer=actor_buffer,eval=False)
                         
                
                unlogged_steps += num_steps
                cached_steps += num_steps
                
                i+=num_steps
                pbar.update(int(num_steps))
                 ### Update critics ###:
                logging.debug('update critics')
                
                
                critic_transitions = replay_buffer.get_all()
                actor_transitions = actor_buffer.get_all()
                
                for _ in range(args.num_epochs):
                    
                    agent = agent.update_critics_seq(critic_transitions)
                    critic_update_info = {}
                    
                
                if i % 5_000 == 0 and args.evaluate_grad:
                    reference_buffer,_,_,_,_ = rollout_policy(agent,env,exploration_rng,
                                                            discount = args.gamma,max_steps=50_000,
                                                            replay_buffer=reference_buffer,actor_buffer=None,eval=False)
                    
                    # For every agent in experimental_agents, change their actor to agent.actor
                    for agent_name in experimental_agents.keys():
                        experimental_agents[agent_name] = experimental_agents[agent_name].replace(actor=agent.actor)
                    
                    experimental_agents,gradient_results,bias_results = evaluate_experimental_agents_gradients(experimental_agents,actor_buffer,replay_buffer,reference_buffer,i)
                    
                    wandb.log(gradient_results, step=i)
                    wandb.log(bias_results, step=i)
                    #print("RESUUUUUUUUUULTS",gradient_results)
                    #print("RESUUUUUUUUUULTS",bias_results)
    
                
                # Evaluate experimental agents gradients  
                if i % 50_000 == 0:  # Evaluate every 50k steps
                    exp_results = evaluate_experimental_agents_gradients(
                        experimental_agents=experimental_agents,
                        actor_buffer=actor_buffer,
                        replay_buffer=replay_buffer,
                        reference_buffer=reference_buffer,
                        step_num=i,
                        critic_training_steps=5000,  # Adjust as needed
                        evaluation_batch_size=3000   # Adjust as needed
                    )
                
                
                    
                    
                
                for _ in range(args.num_epochs):
                    agent,actor_update_info = agent.update_actor_seq(actor_transitions)
                
                
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
                                                                    discount = args.gamma,max_rollouts=10,
                                                                    replay_buffer=None,actor_buffer=None,eval=True)
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
