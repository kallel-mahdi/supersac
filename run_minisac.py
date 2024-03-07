# %%

import os
import wandb
import argparse
import itertools
import numpy as np
import jax
import jax.numpy as jnp


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
parser.add_argument('--env_name',type=str,default="HalfCheetah-v5") 
parser.add_argument('--project_name',type=str,default="delete") 
parser.add_argument('--gamma',type=float,default=0.995)
parser.add_argument('--max_steps',type=int,default=1e6) 
parser.add_argument('--num_rollouts',type=int,default=5) 
parser.add_argument('--num_critics',type=int,default=5) 
parser.add_argument('--adaptive_critics',type=str2bool,default=True) 
parser.add_argument('--discount_entropy',type=str2bool,default=False) 
parser.add_argument('--discount_actor',type=str2bool,default=True) 
parser.add_argument('--max_episode_steps',type=int,default=1000) 
parser.add_argument('--entropy_coeff',type=float,default=1.) 

args = parser.parse_args()

hidden_dims = (256,256)
# cfg = itertools.product([args.seed],[args.env_name],[args.project_name],[args.algo_name],
#                         [args.learning_rate],[args.lengthscale_bound],
#                         [args.reset_critic],[args.aggregation])

# print(cfg)

# %%
"""Implementations of algorithms for continuous control."""
import functools
from jaxrl_m.typing import *

import jax
import jax.lax as lax
import jax.numpy as jnp
import numpy as np
import optax
from jaxrl_m.common import TrainState, target_update, nonpytree_field
from jaxrl_m.networks import Policy, Critic, ensemblize

import flax
import flax.linen as nn
from functools import partial



"""Implementations of algorithms for continuous control."""
import functools
from jaxrl_m.typing import *

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jaxrl_m.common import TrainState, target_update, nonpytree_field
from jaxrl_m.networks import Policy, Critic,OriginalCritic, ensemblize

import flax
import flax.linen as nn

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

    @partial(jax.jit,static_argnames=('num_steps',))
    def update_critic(agent, transitions:Batch,idxs:np.ndarray,num_steps,R2:np.ndarray):
        new_rng, curr_key, next_key = jax.random.split(agent.rng, 3)

        def one_update(agent,batch:Batch):
            
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
        
        def get_batch(transitions,idx):
                return jax.tree_map(lambda x : x[idx],transitions)
            
        def loop_body(i, args):
                return one_update(*args,get_batch(transitions,idxs[i]))
            
        #get_batch = lambda transitions,idx : jax.tree_map(lambda x : x[idx],transitions)
            
        agent,new_critic = jax.lax.fori_loop(0, num_steps,loop_body,(agent,))
        
        return agent.replace(rng=new_rng, critic=new_critic)
        
        
    
    @jax.jit
    def update_actor(agent,batch:Batch):
        
        new_rng, curr_key, next_key = jax.random.split(agent.rng, 3)

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
        
        # new_critic, critic_info = agent.critic.apply_loss_fn(loss_fn=critic_loss_fn, has_aux=True)
        # new_target_critic = target_update(agent.critic, agent.target_critic, agent.config['target_update_rate'])
        new_actor, actor_info = agent.actor.apply_loss_fn(loss_fn=actor_loss_fn, has_aux=True)

        temp_loss_fn = functools.partial(temp_loss_fn, entropy=actor_info['entropy'], target_entropy=agent.config['target_entropy'])
        new_temp, temp_info = agent.temp.apply_loss_fn(loss_fn=temp_loss_fn, has_aux=True)

        return agent.replace(rng=new_rng, actor=new_actor, temp=new_temp), {
             **actor_info, **temp_info}

    @jax.jit
    def sample_actions(agent,   
                       observations: np.ndarray,
                       seed: PRNGKey,
                       random = bool,
                       temperature: float = 1.0,
                       ) -> jnp.ndarray:
        actions = agent.actor(observations, temperature=temperature).sample(seed=seed)
        
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
        actor_def = Policy(hidden_dims, action_dim=action_dim, 
            log_std_min=-10.0, state_dependent_std=True, tanh_squash_distribution=True, final_fc_init_scale=1.0)

        actor_params = actor_def.init(actor_key, observations)['params']
        actor = TrainState.create(actor_def, actor_params, tx=optax.adam(learning_rate=actor_lr))

        critic_def = ensemblize(OriginalCritic, num_qs=2)(hidden_dims)
        critic_params = critic_def.init(critic_key, observations, actions)['params']
        critic = TrainState.create(critic_def, critic_params, tx=optax.adam(learning_rate=critic_lr))
        target_critic = TrainState.create(critic_def, critic_params)

        temp_def = Temperature()
        temp_params = temp_def.init(rng)['params']
        temp = TrainState.create(temp_def, temp_params, tx=optax.adam(learning_rate=temp_lr))

        if target_entropy is None:
            #target_entropy = -0.5 * action_dim
            target_entropy = - action_dim

        config = flax.core.FrozenDict(dict(
            discount=discount,
            target_update_rate=tau,
            target_entropy=target_entropy,
            backup_entropy=backup_entropy,            
        ))

        return SACAgent(rng, critic=critic, target_critic=target_critic, actor=actor, temp=temp, config=config)

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

def train(args):
    
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
    # config.update("jax_debug_nans", True)
    # config.update("jax_enable_x64", True)
            
    from jaxrl_m.rollout import rollout_policy2,rollout_policy
    from jax import config
    config.update("jax_debug_nans", True)

    eval_episodes=10
    batch_size = 256
    max_steps = args.max_steps
    start_steps = 10000                 
    log_interval = 5000

    wandb_config = {
        'project': args.project_name,
        'name':None,
        'hyperparam_dict':args.__dict__,
        }
    

    env = EpisodeMonitor(gym.make(args.env_name,max_episode_steps=args.max_episode_steps))
    eval_env = EpisodeMonitor(gym.make(args.env_name))
    wandb_run = setup_wandb(**wandb_config)

    example_transition = dict(
        observations=env.observation_space.sample(),
        actions=env.action_space.sample(),
        rewards=0.0,
        masks=1.0,
        next_observations=env.observation_space.sample(),
        discounts=1.0,
    )

    replay_buffer = ReplayBuffer.create(example_transition, size=int(200_000))
    actor_buffer = ActorReplayBuffer.create(example_transition, size=int(10e3))

    agent = create_learner(args.seed,
                        
                    observations=example_transition['observations'][None],
                    actions =example_transition['actions'][None],
                    max_steps=max_steps,
                    discount=args.gamma,
                    discount_actor=args.discount_actor,
                    discount_entropy=args.discount_entropy,
                    num_critics= args.num_critics,
                    entropy_coeff=args.entropy_coeff,
                    hidden_dims=hidden_dims,
                    #**FLAGS.config
                    )

    # agent.update_actor = jax.jit(agent.update_actor)
    # agent.update_many_critics = jax.jit(agent.update_many_critics,static_argnames=('num_steps',))
    
    exploration_metrics = dict()
    obs,info = env.reset()    
    exploration_rng = jax.random.PRNGKey(0)
    i = 0
    unlogged_steps = 0
    policy_rollouts = deque([], maxlen=10)
    warmup = True
    R2,bias = jnp.ones(args.num_critics),jnp.zeros(args.num_critics)

    with tqdm.tqdm(total=max_steps) as pbar:
        
        
        while (i < max_steps):

            warmup=(i < start_steps)
            replay_buffer,actor_buffer,policy_rollout,policy_return,variance,undisc_policy_return,num_steps = rollout_policy(
                                                                    agent,env,exploration_rng,
                                                                    replay_buffer,actor_buffer,warmup=warmup,
                                                                    num_rollouts=args.num_rollouts,random=False,
                                                                    discount = args.gamma,
                                                                    )
            
            if not warmup : policy_rollouts.append(policy_rollout)
            unlogged_steps += num_steps
            i+=num_steps
            pbar.update(num_steps)
                
            if replay_buffer.size > start_steps and len(policy_rollouts)>0:
                
                        
               
            
                ### Update critics ###
                transitions = replay_buffer.get_all()
                tmp = partial(jax.random.choice,a=replay_buffer.size, shape=(5000,256), replace=True)
                idxs = jax.vmap(tmp)(jax.random.split(agent.rng, args.num_critics))
                
                
                agent, critic_update_info = agent.update_critic(transitions,idxs,5000,R2)

                ### Update critic weights ## 
                if len(policy_rollouts)>=10 and args.adaptive_critics:   
                
                    flattened_rollouts = flatten_rollouts(policy_rollouts)
                    R2,bias = evaluate_many_critics(agent,policy_rollout.policy_return,flattened_rollouts)

                ### Update actor ###
                actor_batch = actor_buffer.get_all()
                with jax.log_compiles(True):
            
                    agent, actor_update_info = agent.update_actor(actor_batch,R2)    
                update_info = {**critic_update_info, **actor_update_info}
                
                ### Log training info ###
                exploration_metrics = {f'exploration/disc_return': policy_return,'training/std': jnp.sqrt(variance)}
                train_metrics = {f'training/{k}': v for k, v in update_info.items()}
                train_metrics['training/undisc_return'] = undisc_policy_return
                
                observations = actor_batch['observations']
                dist = agent.actor(observations)

                list = []
                curr_key = agent.rng
                for _ in range(10):
                    actions, _ = dist.sample_and_log_prob(seed=curr_key)
                    list.append(actions)
                    curr_key,_ = jax.random.split(curr_key)


                tmp = jnp.stack(list)
                train_metrics['training/noise'] = jnp.std(tmp,axis=0).mean()
                
                wandb.log(train_metrics, step=int(i),commit=False)
                wandb.log(exploration_metrics, step=int(i),commit=False)
                if len(policy_rollouts)>=10:
                    
                    R2_train_info = {'R2/max': jnp.max(R2),'R2/bias': bias[jnp.argmax(R2)],
                                    "R2/histogram": wandb.Histogram(jnp.clip(R2,a_min=-1,a_max=1)),
                                    }
                    wandb.log(R2_train_info, step=int(i),commit=False)
                
                ### Log evaluation info ###
                if unlogged_steps >= log_interval:
                    
                
                    # _,_,_,disc_policy_return,_,undisc_policy_return_e,_ = rollout_policy(agent,eval_env,exploration_rng,
                    #                                                     None,None,warmup=False,
                    #                                                     num_rollouts=10,random=False,discount=args.gamma,
                    #                                                 )
                    # eval_info = {'disc_policy_return': disc_policy_return,'undisc_policy_return': undisc_policy_return_e}
                    # eval_metrics = {f'evaluation/{k}': v for k, v in eval_info.items()}
                    ######################################################################################################
                    policy_fn = partial(supply_rng(agent.sample_actions), temperature=0.0)
                    eval_info = evaluate(policy_fn, eval_env, num_episodes=eval_episodes)
                    eval_metrics = {f'evaluation/{k}': v for k, v in eval_info.items()}
                    wandb.log(eval_metrics, step=int(i),commit=True)
                
                    
                    unlogged_steps = 0
    
    wandb_run.finish()
    

train(args)
#%%
