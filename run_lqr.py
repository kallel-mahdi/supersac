import os
import argparse
import itertools
import logging
import copy
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
import tqdm
import gymnasium as gym
from flax.core.frozen_dict import unfreeze

import wandb
from mushroom_rl.environments import LQR
from mushroom_rl.solvers.lqr import *
from jaxrl_m.common import CodeTimer
from jaxrl_m.wandb import setup_wandb, default_wandb_config, get_flag_dict
from jaxrl_m.evaluation import supply_rng, evaluate, flatten, EpisodeMonitor
from jaxrl_m.dataset import ReplayBuffer, ActorReplayBuffer
from jaxrl_m.rollout import rollout_policy_lqr
from jaxrl_m.ppo_plus import create_learner



# jax.config.update('jax_default_matmul_precision', 'float32')
# jax.config.update("jax_debug_nans", True)
logging.basicConfig(level=logging.debug)
# Set env variables
os.environ["WANDB_API_KEY"] = "28996bd59f1ba2c5a8c3f2cc23d8673c327ae230"
os.environ['PYTHONHASHSEED'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
os.environ['TF_DETERMINISTIC_OPS'] = '1'

jax.config.update("jax_default_matmul_precision", "highest")


def evaluate(observations,key):
            
            actions, log_p,_ = agent.sample_actions(observations,seed=key)
            q_all = agent.critic(observations,actions)
            v = jnp.mean(q_all,axis=0)
            
            return v,log_p

# def evaluate_critic(agent, test_transitions):
#     K = np.array(agent.actor.params['means']['kernel'])
#     noise = jnp.diag(jnp.exp(agent.actor.params['log_stds']))  # For state dependent noise but I think formulation is without

#     Q_true = []
#     for obs, action in zip(test_transitions["observations"], test_transitions["actions"]):
#         Q_true.append(compute_lqr_Q_gaussian_policy(obs, action, env, -K.T, noise))

#     Q_true = jnp.array(Q_true)
#     Q = agent.critic(test_transitions["observations"], test_transitions["actions"]).mean(axis=0)

#     return jnp.sqrt((Q - Q_true) ** 2).mean(), (Q - Q_true).mean()

def evaluate_critic(agent, test_transitions,rng):
    K = np.array(agent.actor.params['means']['kernel'])
    noise = jnp.diag(jnp.exp(agent.actor.params['log_stds']))  # For state dependent noise but I think formulation is without

    Q_true,Adv_true = [],[]
    for obs, action in zip(test_transitions["observations"], test_transitions["actions"]):
        q_true = compute_lqr_Q_gaussian_policy(obs, action, env, -K.T, noise)
        v_true = compute_lqr_V_gaussian_policy(obs, env, -K.T, noise)
        Q_true.append(q_true)
        Adv_true.append(q_true - v_true)
        

    Q_true,Adv_true = jnp.array(Q_true), jnp.array(Adv_true)

    Q = agent.critic(test_transitions["observations"], test_transitions["actions"]).mean(axis=0)
    V,_ = jax.vmap(evaluate,in_axes=(None,0))(test_transitions["observations"],jax.random.split(rng,10))        
    Adv = Q - V

    return jnp.sqrt((Adv-Adv_true) ** 2).mean(), (Q - Q_true).mean()

# def compute_gradient(agent, transitions):
#     K = np.array(agent.actor.params['means']['kernel'])
#     noise = jnp.diag(jnp.exp(agent.actor.params['log_stds']))  # For state dependent noise but I think formulation is without

#     grad = np.zeros((1, np.size(K)))
#     for obs, action, discount in zip(transitions["observations"], transitions["actions"], transitions["discounts"]):
#         grad += discount * compute_lqr_Q_gaussian_policy_gradient_K(obs, action, env, -K.T, noise)

#     grad = grad / (transitions["discounts"].sum())
#     return grad

def compute_gradient(agent, transitions):
    """ This version doesn't use gradient discounting"""

    K = np.array(agent.actor.params['means']['kernel'])
    noise = jnp.diag(jnp.exp(agent.actor.params['log_stds']))  # For state dependent noise but I think formulation is without

    grad = np.zeros((1, np.size(K)))
    for obs, action, discount in zip(transitions["observations"], transitions["actions"], transitions["discounts"]):
        grad += compute_lqr_Q_gaussian_policy_gradient_K(obs, action, env, -K.T, noise)

    return grad


def get_batch(i, batches):
    return jax.tree.map(lambda x: x[i], batches)

def body(i, val):
    agent, batches = val
    return (agent.update_critics(get_batch(i, batches)), batches)

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


# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--algo_name', type=str, default='superppo', help='the name of the RL algorithm')
parser.add_argument('--episode_based', type=str2bool, default=False)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--project_name', type=str, default="RLC_GRAD_EXPS_5000")
parser.add_argument('--max_steps', type=int, default=500_000)
parser.add_argument('--max_episode_steps', type=int, default=1000)
parser.add_argument('--policy_steps', type=int, default=5000)
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--on_policy_data', type=str2bool, default=False)
parser.add_argument('--state_dim', type=int, default=2)
parser.add_argument('--a_dim', type=int, default=1)
args = parser.parse_args()

# Environment setup
env = LQR.generate(s_dim=args.state_dim, a_dim=args.a_dim, gamma=args.gamma, episodic=True, horizon=args.max_episode_steps, random_init=True)

# WandB setup
wandb_config = {
    'project': args.project_name,
    'name': None,
    'hyperparam_dict': args.__dict__,
}
wandb_run = setup_wandb(**wandb_config)

# Initialize variables
eval_episodes = 10
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
    pre_actions=action,
    discounts=1.0,
    log_probs=0.0,
    truncateds=0.0,
)

buffer_size =  args.policy_steps if args.on_policy_data else args.policy_steps * 10
replay_buffer = ReplayBuffer.create(example_transition, size=int(buffer_size))
actor_buffer = ActorReplayBuffer.create(example_transition, size=args.policy_steps)
test_buffer = ActorReplayBuffer.create(example_transition, size=args.policy_steps)

args_dict = {
    "seed": args.seed,
    "observations": example_transition['observations'][None],
    "actions": example_transition['actions'][None],
    "max_steps": max_steps,
    "discount": 0.99,
    "discount_actor": False,
    "min_target": False,
    "discount_entropy": False,
    "adaptive_critics": False,
    "num_critics": 2,
    "entropy_coeff": 1.,
    "temperature": 0.,
    "actor_lr": 3e-4,
    "critic_lr": 3e-4,
    "momentum": 0.9,
    "clipping_ratio": 0.25,
    "num_actor_updates": 1,
    "critic_hidden_dims": (256,256),
    "actor_hidden_dims": (),
    "use_layer_norm": True,
    "state_dependent_std": False,
    "use_bias": False,
    "temp_lr": 3e-4,
    "b2": 0.999,
    "gae_lambda": 0.,
    "minibatch": True,
    "activation_fn": "tanh",
    "stable_scheme": True,
    "bound_actions": False,
    "tanh_squash_distribution": False,
    "tanh_squash_actions": False,
    "store_grads": True,
}

args_dict_normal = copy.deepcopy(args_dict)
args_dict_min = copy.deepcopy(args_dict)
args_dict_no = copy.deepcopy(args_dict)
args_dict_min["min_target"] = True
args_dict_no["use_layer_norm"] = False

agent = create_learner(**args_dict)
agent_normal = create_learner(**args_dict_normal)
#agent = agent.replace(config=unfreeze(agent.config))
agent_on = create_learner(**args_dict) ## On policy
agent_no = create_learner(**args_dict_no) ## No layer norm
agent_min = create_learner(**args_dict_min) ## Minimum of two critics

# Main training loop
exploration_metrics = dict()
exploration_rng = jax.random.PRNGKey(args.seed)
i = 0
unlogged_steps, cached_steps = 0, 0
warmup = True

with tqdm.tqdm(total=max_steps) as pbar:
    while i < max_steps:
        #with jax.log_compiles(False):
            warmup = (i < start_steps)
            logging.debug('policy rollout')
            replay_buffer, actor_buffer, policy_return, undisc_policy_return, num_steps = rollout_policy_lqr(
                agent, env, exploration_rng, replay_buffer, actor_buffer, eval=False,
                num_steps=args.policy_steps, discount=args.gamma, max_length=args.max_episode_steps
            )

            _, test_buffer, _, _, _ = rollout_policy_lqr(
                agent, env, exploration_rng, None, test_buffer, eval=False,
                num_steps=args.policy_steps, discount=args.gamma, max_length=args.max_episode_steps
            )
            
            
            exploration_rng, key = jax.random.split(exploration_rng)

            print(f'policy_return: {policy_return}, undisc_policy_return {undisc_policy_return}')

            unlogged_steps += num_steps
            cached_steps += num_steps
            i += num_steps
            pbar.update(int(num_steps))

            if replay_buffer.size > start_steps:
                # Update critics
                transitions = replay_buffer.get_all()
                agent = agent.update_critics_seq2(transitions,num_updates=5000)
                
                if i % 20000 == 0:
                    
                    agent_normal = agent_normal.update_critics_seq2(transitions,num_updates=5000)
                    agent_min = agent_min.update_critics_seq2(transitions,num_updates=5000)
                    agent_no = agent_no.update_critics_seq2(transitions,num_updates=5000)

                    transitions = actor_buffer.get_all()
                    agent_on = agent_on.update_critics_seq2(transitions,num_updates=5000)
 
                    a, b = evaluate_critic(agent_normal, test_buffer.get_all(),key)
                    c, d = evaluate_critic(agent_min, test_buffer.get_all(),key)
                    e, f = evaluate_critic(agent_on, test_buffer.get_all(),key)
                    g, h = evaluate_critic(agent_no, test_buffer.get_all(),key)

                    wandb.log({"test/normal_error": a, "test/normal_bias": b,
                            "test/min_error": c, "test/min_bias": d,
                            "test/on_error": e, "test/on_bias": f,
                            "test/no_error": g, "test/no_bias": h}, step=int(i))

                    a, b = evaluate_critic(agent_normal, actor_buffer.get_all(),key)
                    c, d = evaluate_critic(agent_min, actor_buffer.get_all(),key)
                    e, f = evaluate_critic(agent_on, actor_buffer.get_all(),key)
                    g,h = evaluate_critic(agent_no, actor_buffer.get_all(),key)

                    wandb.log({"train/normal_error": a, "train/normal_bias": b,
                            "train/min_error": c, "train/min_bias": d,
                            "train/on_error": e, "train/on_bias": f,
                            "train/no_error": g, "train/no_bias": h}, step=int(i))

                    actor_batch = actor_buffer.get_all()
                    true = compute_gradient(agent, actor_batch).reshape(-1)

                    _, actor_update_info = agent_normal.update_actor(actor_batch)
                    estimate = actor_update_info["grads"]["means"]["kernel"]

                    _, actor_update_info = agent_min.update_actor(actor_batch)
                    estimate_min = actor_update_info["grads"]["means"]["kernel"]

                    _, actor_update_info = agent_on.update_actor(actor_batch)
                    estimate_on = actor_update_info["grads"]["means"]["kernel"]

                    _, actor_update_info = agent_no.update_actor(actor_batch)
                    estimate_no = actor_update_info["grads"]["means"]["kernel"]
                

                    tmp = jnp.dot(estimate.flatten(), estimate_no.flatten()) / (jnp.linalg.norm(estimate.flatten()) * jnp.linalg.norm(estimate_no.flatten()))
                    true_estimate = jnp.dot(true.flatten(), estimate.flatten()) / (jnp.linalg.norm(true.flatten()) * jnp.linalg.norm(estimate.flatten()))
                    true_min = jnp.dot(true.flatten(), estimate_min.flatten()) / (jnp.linalg.norm(true.flatten()) * jnp.linalg.norm(estimate_min.flatten()))
                    true_on = jnp.dot(true.flatten(), estimate_on.flatten()) / (jnp.linalg.norm(true.flatten()) * jnp.linalg.norm(estimate_on.flatten()))
                    true_no = jnp.dot(true.flatten(), estimate_no.flatten()) / (jnp.linalg.norm(true.flatten()) * jnp.linalg.norm(estimate_no.flatten()))
                    estimate_no = jnp.dot(estimate.flatten(), estimate_no.flatten()) / (jnp.linalg.norm(estimate.flatten()) * jnp.linalg.norm(estimate_no.flatten()))

                    rslt = {"train/cosine_true_normal": true_estimate, "train/cosine_true_min": true_min, "train/cosine_true_on": true_on,
                            "train/cosine_true_no": true_no, "train/cosine_normal_no": estimate_no}
                    wandb.log(rslt, step=int(i))
                    
                    critic_update_info = {}
                    # Log training info
                
                    update_info = {**critic_update_info, **actor_update_info}
                    train_metrics = {f'training/{k}': v for k, v in update_info.items()}
                    train_metrics['training/undisc_return'] = undisc_policy_return
                    exploration_metrics = {f'exploration/disc_return': policy_return}

                    wandb.log({**exploration_metrics, **train_metrics}, step=int(i))

                
                 
          

                # Update actor
                actor_batch = actor_buffer.get_all()
                agent, actor_update_info = agent.update_actor(actor_batch)
                agent_normal = agent_normal.replace(actor=agent.actor)
                agent_min = agent_min.replace(actor=agent.actor)
                agent_on = agent_on.replace(actor=agent.actor)
                agent_no = agent_on.replace(actor=agent.actor)
                n_grads += 1

             
                