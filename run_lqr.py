from mushroom_rl.environments import LQR
from mushroom_rl.solvers.lqr import *
import jax
import jax.numpy as jnp
jax.config.update('jax_default_matmul_precision', 'float32')


def evaluate_critic(agent,test_transitions):
    
    
    K = np.array(agent.actor.params['means']['kernel'])
    noise = jnp.diag(jnp.exp(agent.actor.params['log_stds']))# For state dependant noise but i think formulation is without
    
    Q_true = []
    
    for obs,action in zip(test_transitions["observations"],test_transitions["actions"]):
        
        Q_true.append(compute_lqr_Q_gaussian_policy(obs,action,env,-K.T,noise))
    
    Q_true = jnp.array(Q_true)
    
    Q = agent.critic(test_transitions["observations"],test_transitions["actions"]).mean(axis=0)
    
    return jnp.sqrt((Q-Q_true)**2).mean(), (Q-Q_true).mean()

def compute_gradient(agent,transitions):

    K = np.array(agent.actor.params['means']['kernel'])
    noise = jnp.diag(jnp.exp(agent.actor.params['log_stds']))# For state dependant noise but i think formulation is without

    grad = np.zeros((1,np.size(K)))

    for obs,action,discount in zip(transitions["observations"],transitions["actions"],transitions["discounts"]):
        
        grad+= discount * compute_lqr_Q_gaussian_policy_gradient_K(obs,action,env,-K.T,noise)

    grad = grad/(transitions["discounts"].sum())
    
    return grad
        
    


import os
import wandb
import argparse
import itertools
import numpy as np
import jax
import jax.numpy as jnp
from jaxrl_m.common import CodeTimer
import logging
from flax.core.frozen_dict import unfreeze
logging.basicConfig(level=logging.CRITICAL)


def get_batch(i,batches):
    return  jax.tree.map(lambda x: x[i], batches)

def body(i,val):
    agent,batches = val
    return (agent.update_critics(get_batch(i,batches)),batches)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    

def none_or_str(value):
    if value == 'None':
        return None
    return value

# Set env variables
os.environ["WANDB_API_KEY"]="28996bd59f1ba2c5a8c3f2cc23d8673c327ae230"
os.environ['PYTHONHASHSEED'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

##############################
parser = argparse.ArgumentParser()

parser.add_argument('--seed',type=int,default=42) 

parser.add_argument('--algo_name', type=str, default='superppo', help='the name of the RL algorithm')
parser.add_argument('--project_name',type=str,default="gradient_exps_99") 

parser.add_argument('--env_name',type=str,default="Hopper-v5") 
parser.add_argument('--max_steps',type=int,default=1000_000) 
parser.add_argument('--max_episode_steps',type=int,default=500) 
parser.add_argument('--num_rollouts',type=int,default=5) 
parser.add_argument('--gamma',type=float,default=0.99)
parser.add_argument('--healthy_reward',type=float,default=1.) 
parser.add_argument('--entropy_coeff',type=float,default=1.) 

parser.add_argument('--discount_actor',type=str2bool,default=True)
parser.add_argument('--min_target',type=str2bool,default=False)
parser.add_argument('--discount_entropy',type=str2bool,default=True) 
parser.add_argument('--on_policy_data',type=str2bool,default=False)
parser.add_argument('--adaptive_critics',type=str2bool,default=False) 
parser.add_argument('--num_critics',type=int,default=2)

parser.add_argument('--critic_lr',type=float,default=3e-4) 
parser.add_argument('--actor_lr',type=float,default=3e-4) 
parser.add_argument('--temperature',type=float,default=0.)
parser.add_argument('--use_layer_norm',type=str2bool,default=True)

parser.add_argument('--momentum',type=float,default=0.) 
parser.add_argument('--num_actor_updates',type=int,default=5) 
parser.add_argument('--clipping_ratio',type=float,default=0.1) 
parser.add_argument('--hidden_dims',type=int,default=64) 
parser.add_argument('--episode_based',type=str2bool,default=False) 
parser.add_argument('--tanh_squash_actions',type=str2bool,default=True) 

parser.add_argument('--state_dim',type=int,default=4) 
parser.add_argument('--a_dim',type=int,default=2) 


args = parser.parse_args()


env = LQR.generate(s_dim=args.state_dim,a_dim=args.a_dim,gamma=args.gamma,episodic=True,horizon=args.max_episode_steps,random_init=True)


from jaxrl_m.onsac_clean import *

hidden_dims = ()
NUM_UPDATES = args.num_rollouts*args.max_episode_steps
data = []

import os
from functools import partial
import numpy as np
import jax
import tqdm
import gymnasium as gym


from jaxrl_m.wandb import setup_wandb, default_wandb_config, get_flag_dict
import wandb
from jaxrl_m.evaluation import supply_rng, evaluate, flatten, EpisodeMonitor
from jaxrl_m.dataset import ReplayBuffer,ActorReplayBuffer
from collections import deque
from jax import config
from jaxrl_m.utils import flatten_rollouts
from jaxrl_m.evaluate_critic import evaluate_many_critics
from jaxrl_m.rollout import rollout_policy_lqr
from jax import config
import copy 


config.update("jax_debug_nans", True)

wandb_config = {
    'project': args.project_name,
    'name':None,
    'hyperparam_dict':args.__dict__,
    }
wandb_run = setup_wandb(**wandb_config)

eval_episodes=10
batch_size = 256
max_steps = args.max_steps
start_steps = 0
log_interval = 10000
n_grads = 0


observation = jnp.ones(env._mdp_info.observation_space.shape)
action = jnp.ones(env._mdp_info.action_space.shape)


example_transition = dict(
    observations=observation,
    actions=action,
    rewards=0.0,
    masks=1.0,
    next_observations=observation,
    pre_actions = action,
    discounts=1.0,
    log_probs=0.,
)
buffer_size = args.num_rollouts*args.max_episode_steps if args.on_policy_data else 100_000
replay_buffer = ReplayBuffer.create(example_transition, size=int(buffer_size))
actor_buffer = ActorReplayBuffer.create(example_transition, size=int(args.num_rollouts*args.max_episode_steps))
test_buffer = ActorReplayBuffer.create(example_transition, size=int(args.num_rollouts*args.max_episode_steps))



args_dict = {

"seed": args.seed,
"observations":example_transition['observations'][None],
"actions":example_transition['actions'][None],
"max_steps":max_steps,
"discount":args.gamma,
"discount_actor":args.discount_actor,
"min_target":args.min_target,
"discount_entropy":args.discount_entropy,
"adaptive_critics":args.adaptive_critics,
"num_critics": args.num_critics,
"entropy_coeff":args.entropy_coeff,
"temperature":args.temperature,
"actor_lr":args.actor_lr,
"critic_lr":args.critic_lr,
"momentum":args.momentum,
"clipping_ratio":args.clipping_ratio,
"num_actor_updates":args.num_actor_updates,
"critic_hidden_dims":(args.hidden_dims,args.hidden_dims),
"actor_hidden_dims":(),
"use_layer_norm": args.use_layer_norm,
"state_dependent_std":False,
"tanh_squash_distribution":False,
"tanh_squash_actions":False,
"use_bias":False,

}

args_dict_min = copy.deepcopy(args_dict)
args_dict_no = copy.deepcopy(args_dict)
args_dict_min["min_target"]=True
args_dict_no["discount_actor"]=False
agent = create_learner(**args_dict)
agent = agent.replace(config=unfreeze(agent.config))
agent_on = create_learner(**args_dict)
agent_no = create_learner(**args_dict_no)
agent_min = create_learner(**args_dict_min)

##############




exploration_metrics = dict()
#obs,info = env.reset()    
exploration_rng = jax.random.PRNGKey(0)
i = 0
unlogged_steps,cached_steps = 0,0
policy_rollouts = deque([], maxlen=20)
warmup = True
R2,bias = jnp.ones(args.num_critics),jnp.zeros(args.num_critics)



with tqdm.tqdm(total=max_steps) as pbar:
    
    while (i < max_steps):
        with jax.log_compiles(False):
            warmup=(i < start_steps)
            
            logging.debug('policy rollout')
            replay_buffer,actor_buffer,policy_rollout,policy_return,variance,undisc_policy_return,num_steps = rollout_policy_lqr(
                                                                    agent,env,exploration_rng,
                                                                    replay_buffer,actor_buffer,eval=False,
                                                                    num_rollouts=args.num_rollouts,discount = args.gamma,max_length=args.max_episode_steps)
            
            
            
            
            _,test_buffer,_,_,_,_,_ = rollout_policy_lqr(
                                                                    agent,env,exploration_rng,
                                                                    None,test_buffer,eval=False,
                                                                    num_rollouts=args.num_rollouts,discount = args.gamma,max_length=args.max_episode_steps)
            
            
            print(f'policy_return: {policy_return}, undisc_policy_return {undisc_policy_return}')                                                              
            if not warmup : policy_rollouts.append(policy_rollout)
            unlogged_steps += num_steps
            cached_steps += num_steps
            i+=num_steps
            pbar.update(int(num_steps))
            
            if replay_buffer.size > start_steps:
            
                ### Update critics ###:
                logging.debug('update critics')
                transitions = replay_buffer.get_all()
                idxs = jax.random.choice(agent.rng,a=transitions['observations'].shape[0], shape=(NUM_UPDATES,256), replace=True)
                batches = jax.vmap(lambda i: jax.tree.map(lambda x: x[i], transitions))(idxs)
                agent = agent.update_critics_seq(batches,R2)
                agent_min = agent_min.update_critics_seq(batches,R2)
                
                
                transitions = actor_buffer.get_all()
                idxs = jax.random.choice(agent.rng,a=transitions['observations'].shape[0], shape=(NUM_UPDATES,256), replace=True)
                batches = jax.vmap(lambda i: jax.tree.map(lambda x: x[i], transitions))(idxs)
                agent_on = agent_on.update_critics_seq(batches,R2)
                agent_no = agent_no.update_critics_seq(batches,R2)

                
                a,b=evaluate_critic(agent,test_buffer.get_all())
                c,d=evaluate_critic(agent_min,test_buffer.get_all())
                e,f=evaluate_critic(agent_on,test_buffer.get_all())
                

                wandb.log({"test/normal_error":a,"test/normal_bias":b,"test/min_error":c,"test/min_bias":d,
                           "test/on_error":e,"test/on_bias":f},step=int(i))
                
                
                a,b=evaluate_critic(agent,actor_buffer.get_all())
                c,d=evaluate_critic(agent_min,actor_buffer.get_all())
                e,f=evaluate_critic(agent_on,actor_buffer.get_all())
                
                wandb.log({"train/normal_error":a,"train/normal_bias":b,"train/min_error":c,"train/min_bias":d,
                          "train/on_error":e,"train/on_bias":f},step=int(i))
                
                
                

                
                actor_batch = actor_buffer.get_all()   
                true = compute_gradient(agent,actor_batch).reshape(-1)

                 
                _, actor_update_info = agent.update_actor(actor_batch,R2)    
                estimate = actor_update_info["grads"]["means"]["kernel"]

                # _, actor_update_info = agent.update_actor_sac(actor_batch,R2)    
                # estimate_sac = actor_update_info["grads"]["means"]["kernel"]
                
                _, actor_update_info = agent_min.update_actor(actor_batch,R2)    
                estimate_min = actor_update_info["grads"]["means"]["kernel"]

                _, actor_update_info = agent_on.update_actor(actor_batch,R2)    
                estimate_on = actor_update_info["grads"]["means"]["kernel"]
                
                agent.config["discount_actor"]=False
                _, actor_update_info = agent_no.update_actor(actor_batch,R2)    
                estimate_no = actor_update_info["grads"]["means"]["kernel"]
                agent.config["discount_actor"]=True
                
                true_estimate = jnp.dot(true.flatten(),estimate.flatten())/(jnp.linalg.norm(true.flatten())*jnp.linalg.norm(estimate.flatten()))
                true_min = jnp.dot(true.flatten(),estimate_min.flatten())/(jnp.linalg.norm(true.flatten())*jnp.linalg.norm(estimate_min.flatten()))
                true_on = jnp.dot(true.flatten(),estimate_on.flatten())/(jnp.linalg.norm(true.flatten())*jnp.linalg.norm(estimate_on.flatten()))
                true_no = jnp.dot(true.flatten(),estimate_no.flatten())/(jnp.linalg.norm(true.flatten())*jnp.linalg.norm(estimate_no.flatten()))
                estimate_no = jnp.dot(estimate.flatten(),estimate_no.flatten())/(jnp.linalg.norm(estimate.flatten())*jnp.linalg.norm(estimate_no.flatten()))
                
                rslt = {"train/cosine_true_estimate":true_estimate,"train/cosine_true_min":true_min,"train/cosine_true_on":true_on,
                           "train/cosine_true_no":true_no,"train/cosine_estimate_no":estimate_no}
                wandb.log(rslt,step=int(i))
                print(rslt)
                
                critic_update_info = {}
                update_info = {**critic_update_info, **actor_update_info}
                n_grads += 1
                
                ### Update actor ###
                agent, actor_update_info = agent.update_actor(actor_batch,R2)   
                agent_min = agent_min.replace(actor=agent.actor)
                agent_on = agent_on.replace(actor=agent.actor)
                

                ### Log training info ###
                exploration_metrics = {f'exploration/disc_return': policy_return,'training/std': jnp.sqrt(variance)}
                train_metrics = {f'training/{k}': v for k, v in update_info.items()}
                train_metrics['training/undisc_return'] = undisc_policy_return
                                    
            
                if cached_steps >= int(1e6): 
                    jax.clear_caches()
                    cached_steps = 0
                    print('clearing cache')