# %%

import argparse
import logging
import os

import jax.numpy as jnp

from jaxrl_m.onsac_clean import *
from jaxrl_m.utils import *
from jaxrl_m.normalize import *

logging.basicConfig(level=logging.CRITICAL)
#jax.config.update("jax_enable_x64", True)
jax.config.update('jax_default_matmul_precision', 'float32')

# Set env variables
os.environ["WANDB_API_KEY"]="28996bd59f1ba2c5a8c3f2cc23d8673c327ae230"
os.environ['PYTHONHASHSEED'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

##############################
parser = argparse.ArgumentParser()

parser.add_argument('--seed',type=int,default=42) 

parser.add_argument('--algo_name', type=str, default='superppo', help='the name of the RL algorithm')
parser.add_argument('--project_name',type=str,default="single_exp") 

parser.add_argument('--env_name',type=str,default="Hopper-v5") 
parser.add_argument('--max_steps',type=int,default=1_000_000) 
parser.add_argument('--max_episode_steps',type=int,default=1000) 
parser.add_argument('--num_rollouts',type=int,default=4) 
parser.add_argument('--gamma',type=float,default=0.995)
parser.add_argument('--healthy_reward',type=float,default=0.5) 
parser.add_argument('--entropy_coeff',type=float,default=1.) 

parser.add_argument('--discount_actor',type=str2bool,default=True)
parser.add_argument('--min_target',type=str2bool,default=False)
parser.add_argument('--discount_entropy',type=str2bool,default=False) 
parser.add_argument('--on_policy_data',type=str2bool,default=False)
parser.add_argument('--adaptive_critics',type=str2bool,default=True) 
parser.add_argument('--num_critics',type=int,default=4)

parser.add_argument('--critic_lr',type=float,default=3e-4) 
parser.add_argument('--actor_lr',type=float,default=3e-4) 
parser.add_argument('--temp_lr',type=float,default=3e-4)
parser.add_argument('--use_layer_norm',type=str2bool,default=True)

parser.add_argument('--momentum',type=float,default=0.) 
parser.add_argument('--num_actor_updates',type=int,default=10) 
parser.add_argument('--clipping_ratio',type=float,default=0.05) 
parser.add_argument('--hidden_dims',type=int,default=256) 
parser.add_argument('--episode_based',type=str2bool,default=False) 

args = parser.parse_args()
print(args)


def train(args):
    
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
    from jaxrl_m.evaluate_critic import evaluate_many_critics
    from jaxrl_m.evaluation import (EpisodeMonitor, evaluate, flatten,
                                    supply_rng)
    from jaxrl_m.rollout import (rollout_policy, rollout_policy2)
    from jaxrl_m.utils import flatten_rollouts
    from jaxrl_m.wandb import default_wandb_config, get_flag_dict, setup_wandb
    config.update("jax_debug_nans", True)

    eval_episodes=10
    batch_size = 256
    max_steps = args.max_steps
    start_steps = 0
    log_interval = 10000
    n_grads = 0

    wandb_config = {
        'project': args.project_name,
        'name':None,
        'hyperparam_dict':args.__dict__,
        }
    wandb_run = setup_wandb(**wandb_config)
    
    ### HalfCheetah does not have healthy_reward argument
    if 'HalfCheetah' in args.env_name or 'Standup' in args.env_name:
        env = EpisodeMonitor(gym.make(args.env_name, max_episode_steps=args.max_episode_steps))
    else:
        print(f'env_name: {args.env_name}, max_episode_steps: {args.max_episode_steps}, healthy_reward: {args.healthy_reward}')
        env = gym.make(args.env_name, max_episode_steps=args.max_episode_steps, healthy_reward=args.healthy_reward)
    
    eval_env = EpisodeMonitor(gym.make(args.env_name,max_episode_steps=1000))
    

    example_transition = dict(
        observations=env.observation_space.sample(),
        actions=env.action_space.sample(),
        rewards=0.0,
        masks=1.0,
        next_observations=env.observation_space.sample(),
        pre_actions = env.action_space.sample(),
        discounts=1.0,
        log_probs=0.,
    )
    buffer_size = args.num_rollouts*args.max_episode_steps if args.on_policy_data else 100_000
    replay_buffer = ReplayBuffer.create(example_transition, size=int(buffer_size))
    actor_buffer = ActorReplayBuffer.create(example_transition, size=int(args.num_rollouts*args.max_episode_steps))

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
                    temp_lr=args.temp_lr,
                    actor_lr=args.actor_lr,
                    critic_lr=args.critic_lr,
                    momentum=args.momentum,
                    clipping_ratio=args.clipping_ratio,
                    num_actor_updates=args.num_actor_updates,
                    actor_hidden_dims=(args.hidden_dims,args.hidden_dims),
                    critic_hidden_dims=(args.hidden_dims,args.hidden_dims),
                    use_layer_norm= args.use_layer_norm,
                    #**FLAGS.config
                    )

    exploration_metrics = dict()
    obs,info = env.reset()    
    exploration_rng = jax.random.PRNGKey(0)
    i = 0
    unlogged_steps,cached_steps = 0,0
    policy_rollouts = deque([], maxlen=20)
    
    R2,bias = jnp.ones(args.num_critics)/args.num_critics,jnp.zeros(args.num_critics)
    
    rollout_fn = rollout_policy if args.episode_based else rollout_policy2
        
    with tqdm.tqdm(total=max_steps) as pbar:
        
        while (i < max_steps):
                
                logging.debug('policy rollout')
                if args.on_policy_data: replay_buffer = replay_buffer.reset()
                replay_buffer,actor_buffer,policy_rollout,policy_return,variance,undisc_policy_return,num_steps = rollout_fn(
                                                                        agent,env,exploration_rng,
                                                                        replay_buffer,actor_buffer,eval=False,
                                                                        num_rollouts=args.num_rollouts,discount = args.gamma,max_length=args.max_episode_steps)
                                                              
                policy_rollouts.append(policy_rollout)
                
                unlogged_steps += num_steps
                cached_steps += num_steps
                
                i+=num_steps
                pbar.update(int(num_steps))
                
            
                ### Update critics ###:
                logging.debug('update critics')
                transitions = replay_buffer.get_all()
                idxs = jax.random.choice(agent.rng,a=transitions['observations'].shape[0], shape=(args.num_rollouts*args.max_episode_steps,256), replace=True)
                batches = jax.vmap(lambda i: jax.tree.map(lambda x: x[i], transitions))(idxs)
                agent = agent.update_critics_seq(batches,R2)
                        
                ### Update actor ###
                actor_batch = actor_buffer.get_all()    
                
                if len(policy_rollouts)>=2 and args.adaptive_critics:
                    
                    
                    mask = jnp.zeros(( agent.config["num_critics"],))
                    mask = mask.at[0].set(1)
                    rngs = jax.random.split(agent.rng, agent.config["num_critics"])
                    critic = OriginalCritic((256,256))
                    reset = lambda rng,params : critic.init(rng,
                                                    agent.config["observations"], agent.config["actions"],False)["params"]
                    no_reset = lambda rng,params: params
            
                    f = lambda  mask,rng,params :lax.cond(mask,reset,no_reset,rng,params)
                    new_params = jax.vmap(f,in_axes=(0,0,0))(mask,rngs,agent.critic.params)
                    new_opt_state = agent.critic.tx.init(new_params)
                    new_critic = agent.critic.replace(params=new_params,opt_state=new_opt_state)
                    agent = agent.replace(critic=new_critic)
                    
                    flattened_rollouts = flatten_rollouts(policy_rollouts)
                    R2,bias = evaluate_many_critics(agent,policy_rollout.policy_return,flattened_rollouts,args.num_critics)
                    
                    R2_train_info = {'R2/max': jnp.max(R2),'R2/bias': bias[jnp.argmax(R2)],
                                    "R2/histogram": wandb.Histogram(jnp.clip(R2,a_min=-1,a_max=1)),
                                    }
                    wandb.log(R2_train_info, step=int(i),commit=False)
                    
                        
                #with jax.default_matmul_precision("float32"):
                agent, actor_update_info = agent.update_actor(actor_batch,R2)    
                    
                critic_update_info = {}
                update_info = {**critic_update_info, **actor_update_info}
                n_grads += 1
                
                ### Log training info ###
                exploration_metrics = {f'exploration/disc_return': policy_return,'training/std': jnp.sqrt(variance)}
                train_metrics = {f'training/{k}': v for k, v in update_info.items()}
                train_metrics['training/undisc_return'] = undisc_policy_return
                
                wandb.log(train_metrics, step=int(i),commit=False)
                wandb.log(exploration_metrics, step=int(i),commit=False)
            
                ### Log evaluation info ###
                
                if unlogged_steps >= log_interval:
                    
                    _,_,policy_rollout,policy_return,variance,undisc_policy_return,num_steps = rollout_policy(
                                                                    agent,eval_env,exploration_rng,
                                                                    None,None,eval=True,
                                                                    num_rollouts=10,
                                                                    discount = args.gamma,max_length=1000)
                    eval_metrics = {"policy_return": policy_return,"std": jnp.sqrt(variance),"undisc_policy_return": undisc_policy_return}

                    
                    # policy_fn = partial(supply_rng(agent.sample_action), temperature=0.)
                    # eval_metrics = evaluate(policy_fn, eval_env, num_episodes=10)
                    
                    eval_metrics = {f'evaluation/{k}': v for k, v in eval_metrics.items()}
                    eval_metrics['n_grads']=int(n_grads)

                    eval_step = i
                    wandb.log(eval_metrics, step=int(eval_step),commit=True)
                    unlogged_steps = 0
            
                if cached_steps >= int(1e6): 
                    jax.clear_caches()
                    cached_steps = 0
                    print('clearing cache')
        
    wandb_run.finish()

train(args)
#%%
