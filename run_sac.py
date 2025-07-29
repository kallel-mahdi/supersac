"""Implementations of algorithms for continuous control."""
import functools
import argparse
from jaxrl_m.typing import *

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jaxrl_m.common import TrainState, target_update, nonpytree_field
from jaxrl_m.networks import Policy, Critic,OriginalCritic, ensemblize

import flax
import flax.linen as nn

import os
from functools import partial
import numpy as np
import jax
import tqdm
import gymnasium as gym

from jaxrl_m.wandb import setup_wandb, default_wandb_config, get_flag_dict
from jaxrl_m.evaluation import supply_rng, evaluate, flatten, EpisodeMonitor
from jaxrl_m.dataset import ReplayBuffer
from jaxrl_m.rollout import rollout_policy
from jaxrl_m.utils import *

import wandb
import random


os.environ["WANDB_API_KEY"]="28996bd59f1ba2c5a8c3f2cc23d8673c327ae230"
os.environ["WANDB__SERVICE_WAIT"] = str(1800)
os.environ['PYTHONHASHSEED'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['XLA_FLAGS']='--xla_gpu_deterministic_ops=true'

##############################
parser = argparse.ArgumentParser()

parser.add_argument('--seed',type=int,default=21) 
parser.add_argument('--project_name', type=str, default='sac_benchmark2', help='Name of the wandb project to log to')
parser.add_argument('--env_name', type=str, default='Walker2d-v5', help='Name of the gym environment to use')
parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
parser.add_argument('--algo_name', type=str, default="sac")
parser.add_argument('--entropy_coeff',type=float,default=0.5)   
parser.add_argument('--buffer_size',type=int,default=1_000_000)
parser.add_argument('--max_steps',type=int,default=None)

args = parser.parse_args()
random.seed(args.seed)
np.random.seed(args.seed)

def get_default_config():
    import ml_collections

    return ml_collections.ConfigDict({
        'actor_lr': 3e-4,
        'critic_lr': 3e-4,
        'temp_lr': 3e-4,
        'hidden_dims': (256, 256),
        'discount': 0.99,
        'tau': 0.005,
        'target_entropy': ml_collections.config_dict.placeholder(float),
        'backup_entropy': True,
    })
    


class Temperature(nn.Module):
    initial_temperature: float = 1.0

    @nn.compact
    def __call__(self) -> jnp.ndarray:
        log_temp = self.param('log_temp',
                              init_fn=lambda key: jnp.full(
                                  (), jnp.log(self.initial_temperature)))
        return jnp.exp(log_temp)

class SACAgent(flax.struct.PyTreeNode):
    rng: PRNGKey
    critic: TrainState
    target_critic: TrainState
    actor: TrainState
    temp: TrainState
    config: dict = nonpytree_field()

    @jax.jit
    def update(agent, batch: Batch):
        new_rng, curr_key, next_key = jax.random.split(agent.rng, 3)

        def critic_loss_fn(critic_params):
            next_dist = agent.actor(batch['next_observations'])
            next_actions, next_log_probs = next_dist.sample_and_log_prob(seed=next_key)

            next_q1, next_q2 = agent.target_critic(batch['next_observations'], next_actions)
            next_q = jnp.minimum(next_q1, next_q2)
            target_q = batch['rewards'] + agent.config['discount'] * batch['masks'] * next_q

            if agent.config['backup_entropy']:
                target_q = target_q - agent.config['discount'] * batch['masks'] * next_log_probs * agent.temp()
            
            q1, q2 = agent.critic(batch['observations'], batch['actions'], params=critic_params)
            critic_loss = ((q1 - target_q)**2 + (q2 - target_q)**2).mean()
            
            return critic_loss, {
                'critic_loss': critic_loss,
                'q1': q1.mean(),
            }        

        def actor_loss_fn(actor_params):
            dist = agent.actor(batch['observations'], params=actor_params)
            actions, log_probs = dist.sample_and_log_prob(seed=curr_key)
            
            q1, q2 = agent.critic(batch['observations'], actions)
            q = jnp.minimum(q1, q2)

            actor_loss = (log_probs * agent.temp() - q).mean()
            return actor_loss, {
                'actor_loss': actor_loss,
                'entropy': -1 * log_probs.mean(),
            }
        
        def temp_loss_fn(temp_params, entropy, target_entropy):
            temperature = agent.temp(params=temp_params)
            temp_loss = (temperature * (entropy - target_entropy)).mean()
            return temp_loss, {
                'temp_loss': temp_loss,
                'temperature': temperature,
            }
        
        new_critic, critic_info = agent.critic.apply_loss_fn(loss_fn=critic_loss_fn, has_aux=True)
        new_target_critic = target_update(agent.critic, agent.target_critic, agent.config['target_update_rate'])
        new_actor, actor_info = agent.actor.apply_loss_fn(loss_fn=actor_loss_fn, has_aux=True)

        temp_loss_fn = functools.partial(temp_loss_fn, entropy=actor_info['entropy'], target_entropy=agent.config['target_entropy'])
        new_temp, temp_info = agent.temp.apply_loss_fn(loss_fn=temp_loss_fn, has_aux=True)

        return agent.replace(rng=new_rng, critic=new_critic, target_critic=new_target_critic, actor=new_actor, temp=new_temp), {
            **critic_info, **actor_info, **temp_info}

    @jax.jit
    def sample_actions(agent,   
                       observations: np.ndarray,
                       seed: PRNGKey,
                       random = bool,
                       temperature: float = 1.0,
                       ) -> jnp.ndarray:
        actions = agent.actor(observations, temperature=temperature).sample(seed=seed)
        
        return actions
    
      
    @jax.jit
    def deterministic_action(agent,   
                       observations: np.ndarray,
                       ) -> jnp.ndarray:
        
        ### random always true
        seed = jax.random.PRNGKey(0)
        dist = agent.actor(observations, temperature=0.)
        actions,pre_log_ps = dist.sample_and_log_prob(seed=seed)
       
        return actions



def create_learner(
                 seed: int,
                 observations: jnp.ndarray,
                 actions: jnp.ndarray,
                 actor_lr: float = 3e-4,
                 critic_lr: float = 3e-4,
                 temp_lr: float = 3e-4,
                 hidden_dims: Sequence[int] = (256, 256),
                 discount: float = 0.99,
                 tau: float = 0.005,
                 target_entropy: float = None,
                 backup_entropy: bool = True,
            **kwargs):

        print('Extra kwargs:', kwargs)

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key = jax.random.split(rng, 3)

        action_dim = actions.shape[-1]
        actor_def = Policy(hidden_dims, action_dim=action_dim,use_layer_norm=False,activations=nn.relu,
            log_std_min=-10.0, tanh_squash_distribution=True, final_fc_init_scale=1.0)

        actor_params = actor_def.init(actor_key, observations)['params']
        actor = TrainState.create(actor_def, actor_params, tx=optax.adam(learning_rate=actor_lr))

        critic_def = ensemblize(OriginalCritic, num_qs=2)(hidden_dims,use_layer_norm=False,activations=nn.relu,)
        critic_params = critic_def.init(critic_key, observations, actions)['params']
        critic = TrainState.create(critic_def, critic_params, tx=optax.adam(learning_rate=critic_lr))
        target_critic = TrainState.create(critic_def, critic_params)

        temp_def = Temperature()
        temp_params = temp_def.init(rng)['params']
        temp = TrainState.create(temp_def, temp_params, tx=optax.adam(learning_rate=temp_lr))

        if target_entropy is None:
            target_entropy = - args.entropy_coeff * action_dim

        config = flax.core.FrozenDict(dict(
            discount=discount,
            target_update_rate=tau,
            target_entropy=target_entropy,
            backup_entropy=backup_entropy,            
        ))

        return SACAgent(rng, critic=critic, target_critic=target_critic, actor=actor, temp=temp, config=config)



def train():

    #FLAGS = flags.FLAGS
    seed=args.seed
    eval_episodes=10
    batch_size = 256
    
    start_steps = int(1e4)                     
    log_interval = 20000
    eval_interval = 10000

 
    wandb_config = {
        'project': args.project_name,
        'name':None,
        'hyperparam_dict':args.__dict__,
        }
    wandb_run = setup_wandb(**wandb_config)

    env, eval_env = create_environments(args.env_name)
    
    max_steps = get_max_steps_for_env(args.env_name) if args.max_steps is None else args.max_steps
   

    example_transition = dict(
        observations=env.observation_space.sample(),
        actions=env.action_space.sample(),
        rewards=0.0,
        masks=1.0,
        next_observations=env.observation_space.sample(),
    )

    replay_buffer = ReplayBuffer.create(example_transition, size=int(args.buffer_size))
    placeholder = ReplayBuffer.create(example_transition, size=int(args.buffer_size))

    agent = create_learner(args.seed,
                    example_transition['observations'][None],
                    example_transition['actions'][None],
                    max_steps=max_steps,
                    discount = args.gamma,
                    #**FLAGS.config
                    )

    exploration_metrics = dict()
    obs,info = env.reset()    
    exploration_rng = jax.random.PRNGKey(args.seed)

    for i in tqdm.tqdm(range(1, max_steps + 1),
                        smoothing=0.1,
                        dynamic_ncols=True):

        if i < start_steps:
            action = env.action_space.sample()
        else:
            exploration_rng, key = jax.random.split(exploration_rng)
            action = agent.sample_actions(obs, seed=key)

        #next_obs, reward, done, info = env.step(action)
        next_obs, reward, done, truncated, info = env.step(action)
        
        mask = float(not done or 'TimeLimit.truncated' in info)
        
        replay_buffer.add_transition(dict(
            observations=obs,
            actions=action,
            rewards=reward,
            masks=mask,
            next_observations=next_obs,
        ))
        obs = next_obs

        if (done or truncated):
            exploration_metrics = {f'exploration/{k}': v for k, v in flatten(info).items()}
            obs,info= env.reset()

        if replay_buffer.size < start_steps:
            continue

        batch = replay_buffer.sample(batch_size)  
        
        
        agent, update_info = agent.update(batch)

        if i % log_interval == 0:
            train_metrics = {f'training/{k}': v for k, v in update_info.items()}
            wandb.log(train_metrics, step=i)
            wandb.log(exploration_metrics, step=i)
            exploration_metrics = dict()

        if i % eval_interval == 0:
            
            
            policy_fn = partial(supply_rng(agent.sample_actions), temperature=0.0)
            eval_info = evaluate(policy_fn, eval_env, num_episodes=eval_episodes)
            eval_metrics = {f'evaluation/{k}': v for k, v in eval_info.items()}
            
            _,_,policy_return,undisc_policy_return,num_steps = rollout_policy(
                                                            agent,eval_env,exploration_rng,
                                                            discount = agent.config["discount"],max_rollouts=10,
                                                            replay_buffer=None,actor_buffer=None,eval=True)
            eval_metrics = {"policy_return": policy_return,"undisc_policy_return": undisc_policy_return}
            print(eval_metrics)
            
            eval_metrics = {f'evaluation/{k}': v for k, v in eval_metrics.items()}
            
        
            wandb.log(eval_metrics, step=int(i),commit=True)


        
    wandb_run.finish()

train()