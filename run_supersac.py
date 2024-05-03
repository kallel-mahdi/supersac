# %%

import os
import wandb
import argparse
import itertools
import numpy as np
import jax
import jax.numpy as jnp
from jaxrl_m.common import CodeTimer
import logging
import envpool
logging.basicConfig(level=logging.CRITICAL)


def get_batch(i,batches):
    return  jax.tree_map(lambda x: x[i], batches)

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
parser.add_argument('--algo_name', type=str, default='sac', help='the name of the RL algorithm')
parser.add_argument('--seed',type=int,default=42) 
parser.add_argument('--env_name',type=str,default="Walker2d-v5") 
parser.add_argument('--project_name',type=str,default="delete") 
parser.add_argument('--gamma',type=float,default=0.995)
parser.add_argument('--max_steps',type=int,default=2_000_000) 
parser.add_argument('--num_rollouts',type=int,default=5) 
parser.add_argument('--num_critics',type=int,default=5)     
parser.add_argument('--discount_actor',type=str2bool,default=True)
parser.add_argument('--discount_entropy',type=str2bool,default=True) 
parser.add_argument('--use_momentum',type=str2bool,default=False) 
parser.add_argument('--adaptive_critics',type=str2bool,default=False) 
parser.add_argument('--max_episode_steps',type=int,default=1000) 
parser.add_argument('--entropy_coeff',type=float,default=1.) 
parser.add_argument('--actor_lr',type=float,default=3e-4) 
parser.add_argument('--temp_lr',type=float,default=3e-4)
parser.add_argument('--healthy_reward',type=float,default=0.5) 


args = parser.parse_args()
NUM_UPDATES = args.max_episode_steps*args.num_rollouts
hidden_dims = (256,256)
# cfg = itertools.product([args.seed],[args.env_name],[args.project_name],[args.algo_name],
#                         [args.learning_rate],[args.lengthscale_bound],
#                         [args.reset_critic],[args.aggregation])

# print(cfg)


NUM_UPDATES = args.max_episode_steps*args.num_rollouts
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
from jaxrl_m.networks import Policy, Critic,OriginalCritic, ensemblize

import flax
import flax.linen as nn
from functools import partial



class Temperature(nn.Module):
    initial_temperature: float = 1e-6

    
    @nn.compact
    def __call__(self) -> jnp.ndarray:
        log_temp = self.param('log_temp',
                              init_fn=lambda key: jnp.full(
                                  (), self.initial_temperature))
        return jnp.abs(log_temp)


class SACAgent(flax.struct.PyTreeNode):
    rng: PRNGKey
    critic: TrainState
    target_critic: TrainState
    actor: TrainState
    temp: TrainState
    config: dict = nonpytree_field()

    #@jax.jit
    def update_critics(agent,batch: Batch):
        
        new_rng, curr_key, next_key = jax.random.split(agent.rng, 3)

        def update_one_critic(critic):
                            
                def critic_loss_fn(critic_params):
                        
                        
                        next_dist = agent.actor(batch['next_observations'])
                        next_actions, next_log_probs = next_dist.sample_and_log_prob(seed=next_key)
                        next_q  = agent.critic(batch['next_observations'], next_actions,params=critic_params)
                        
                        target_q = batch['rewards'] + agent.config['discount'] * batch['masks'] * next_q
                        target_q = target_q - agent.config['discount'] * batch['masks'] * next_log_probs * agent.temp()
                        target_q = jax.lax.stop_gradient(target_q)
                        
                        q = agent.critic(batch['observations'], batch['actions'],params=critic_params)
                        critic_loss = ((target_q-q)**2).mean() 
                        
                        return critic_loss, {
                        'critic_loss': critic_loss,
                        'q1': q.mean(),
                    }  
                
                new_critic, critic_info = critic.apply_loss_fn(loss_fn=critic_loss_fn, has_aux=True)
                
                return new_critic,critic_info


        new_critics,critic_info = jax.vmap(update_one_critic)(agent.critic)
        agent = agent.replace(rng=new_rng,critic=new_critics)
        
        return agent
    
    @jax.jit
    def update_critics_seq(agent,batches,R2):
       
        new_critic_params = agent.critic.params
        ### Reset optimizers 
        new_opt_state = jax.vmap(agent.critic.tx.init)(new_critic_params)
        new_critics = agent.critic.replace(params=new_critic_params,opt_state=new_opt_state)
        agent = agent.replace(critic=new_critics)
        ### Train critic sequentially
        agent,batches = jax.lax.fori_loop(0,NUM_UPDATES,body,(agent,batches))
        
        return agent

    @jax.jit
    def update_actor(agent, batch: Batch,R2):
        new_rng, curr_key, next_key = jax.random.split(agent.rng, 3)

        def actor_loss_fn(actor_params,R2):
            observations = jnp.repeat(batch['observations'], 10, axis=0)
            discounts = jnp.repeat(batch['discounts'], 10, axis=0)
            masks = jnp.int32(jnp.repeat(batch['masks'], 10, axis=0))

            dist = agent.actor(observations, params=actor_params)
            actions, log_probs = dist.sample_and_log_prob(seed=curr_key)
            call_one_critic = lambda observations,actions,params: agent.critic(observations,actions,params=params)
            #q_r_all,q_e_all = jax.vmap(call_one_critic,in_axes=(None,None,0))(observations, actions,agent.critic.params)##critic_update_info
            q_all = jax.vmap(call_one_critic,in_axes=(None,None,0))(observations, actions,agent.critic.params)##critic_update_info
            #q_all = q_all.squeeze()
            q_weights = jax.nn.softmax(R2,axis=0)
            q = jnp.sum(q_weights.reshape(-1,1)*q_all,axis=0)
            print('aaaaa',q_all.shape,q_weights.shape)
            print('qqqqqqqqqqqqqq',q.shape)

            
            ### Pad Q and logits because actor buffer is padded ###
            q = masks *q
            log_probs = masks * log_probs
            
            if agent.config['discount_actor']:
                actor_loss = (discounts*(log_probs * agent.temp() - q)).sum()/(discounts.sum())
            else :
                actor_loss = (log_probs * agent.temp() - q).sum()/(masks.sum())
            
            if agent.config['discount_entropy']:
                entropy = -1 * (discounts*log_probs).sum()/(discounts.sum())
            else : 
                entropy = -1 * log_probs.sum()/(masks.sum())
            
            return actor_loss, {
                'actor_loss': actor_loss,
                'entropy': entropy,
            }
        
        
        def temp_loss_fn(temp_params, entropy, target_entropy):
            temperature = agent.temp(params=temp_params)
            entropy_diff = entropy-target_entropy
            temp_loss = (temperature * entropy_diff).mean()
            return temp_loss, {
                'temp_loss': temp_loss,
                'temperature': temperature,
                'entropy_diff': entropy_diff,
            }

        
        new_actor, actor_info = agent.actor.apply_loss_fn(actor_loss_fn,True,R2)
        new_temp, temp_info = agent.temp.apply_loss_fn(temp_loss_fn,True,actor_info['entropy'], agent.config['target_entropy'])
        new_temp.params["log_temp"]=jnp.clip(new_temp.params["log_temp"],1e-6,1)
        
        return agent.replace(rng=new_rng, actor=new_actor,temp=new_temp), {**actor_info,**temp_info}
        
        

    @jax.jit
    def sample_actions(agent,   
                       observations: np.ndarray,
                       seed: PRNGKey,
                       temperature: float = 1.0,
                       ) -> jnp.ndarray:
        
        ### random always true
        actions = agent.actor(observations, temperature=temperature).sample(seed=seed)
        
        
        return actions


def create_learner(
                seed: int,
                observations: jnp.ndarray,
                actions: jnp.ndarray,
                discount: float,
                num_critics: int,
                discount_actor ,
                discount_entropy,
                adaptive_critics,
                entropy_coeff,
                use_momentum,
                
                actor_lr: float = 3e-4,
                critic_lr: float = 3e-4,
                temp_lr: float =1e-3,## Test
                hidden_dims: Sequence[int] = (256, 256),
                target_entropy: float = None,
            **kwargs):

        print('Extra kwargs:', kwargs)

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key = jax.random.split(rng, 3)

        action_dim = actions.shape[-1]
        actor_def = Policy(hidden_dims, action_dim=action_dim,
            state_dependent_std=True, tanh_squash_distribution=True)

        critic_def = OriginalCritic(hidden_dims)
        critic_keys  = jax.random.split(critic_key, num_critics)
        critic_params = jax.vmap(critic_def.init,in_axes=(0,None,None))(critic_keys, observations, actions)['params']
        critics = jax.vmap(TrainState.create,in_axes=(None,0,None))(critic_def,critic_params,optax.adam(learning_rate=critic_lr))

        actor_params = actor_def.init(actor_key, observations)['params']
        #actor = TrainState.create(actor_def, actor_params, tx=optax.adam(learning_rate=actor_lr))
        
        temp_def = Temperature()
        temp_params = temp_def.init(rng)['params']
        #temp = TrainState.create(temp_def, temp_params, tx=optax.adam(learning_rate=temp_lr))
        
        if use_momentum:
            temp = TrainState.create(temp_def, temp_params, tx=optax.adam(learning_rate=temp_lr))
            actor = TrainState.create(actor_def, actor_params, tx=optax.adam(learning_rate=actor_lr))
            
        else:
            temp = TrainState.create(temp_def, temp_params, tx=optax.rmsprop(learning_rate=temp_lr))
            actor = TrainState.create(actor_def, actor_params, tx=optax.rmsprop(learning_rate=actor_lr))
        
            
        if target_entropy is None:
            target_entropy = -entropy_coeff*action_dim
        config = flax.core.FrozenDict(dict(
            discount=discount,
            target_entropy=target_entropy,
            observations=observations,
            actions=actions,  
            num_critics = num_critics, 
            discount_actor = discount_actor, 
            discount_entropy = discount_entropy,
            adaptive_critics = adaptive_critics,
            #critic_def = critic_def,    
        ))

        return SACAgent(rng, critic=critics, target_critic=critics, actor=actor, temp=temp, config=config)



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
    from jaxrl_m.rollout import rollout_policy2,rollout_policy,rollout_policy_parallel
    from jax import config
    config.update("jax_debug_nans", True)

    eval_episodes=10
    batch_size = 256
    max_steps = args.max_steps
    start_steps = 10000
    log_interval = 10000
    n_grads = 0

    wandb_config = {
        'project': args.project_name,
        'name':None,
        'hyperparam_dict':args.__dict__,
        }
    wandb_run = setup_wandb(**wandb_config)
    
    
    ### HalfCheetah does not have healthy_reward argument
    #f 'HalfCheetah' in args.env_name:

    if 'HalfCheetah' in args.env_name or 'Pendulum' in args.env_name:
        env = EpisodeMonitor(gym.make(args.env_name, max_episode_steps=args.max_episode_steps))
    else:
        print(f'env_name: {args.env_name}, max_episode_steps: {args.max_episode_steps}, healthy_reward: {args.healthy_reward}')
        env = EpisodeMonitor(gym.make(args.env_name, max_episode_steps=args.max_episode_steps, healthy_reward=args.healthy_reward))
    
    eval_env = EpisodeMonitor(gym.make(args.env_name,max_episode_steps=1000))
    

    example_transition = dict(
        observations=env.observation_space.sample(),
        actions=env.action_space.sample(),
        rewards=0.0,
        masks=1.0,
        next_observations=env.observation_space.sample(),
        discounts=1.0,
    )

    replay_buffer = ReplayBuffer.create(example_transition, size=int(100_000))
    actor_buffer = ActorReplayBuffer.create(example_transition, size=int(args.num_rollouts*args.max_episode_steps))

    agent = create_learner(args.seed,
                        
                    observations=example_transition['observations'][None],
                    actions =example_transition['actions'][None],
                    max_steps=max_steps,
                    discount=args.gamma,
                    discount_actor=args.discount_actor,
                    discount_entropy=args.discount_entropy,
                    adaptive_critics=args.adaptive_critics,
                    num_critics= args.num_critics,
                    entropy_coeff=args.entropy_coeff,
                    temp_lr=args.temp_lr,
                    actor_lr=args.actor_lr,
                    use_momentum=args.use_momentum,
                    hidden_dims=hidden_dims,
                    #**FLAGS.config
                    )

    exploration_metrics = dict()
    obs,info = env.reset()    
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
                replay_buffer,actor_buffer,policy_rollout,policy_return,variance,undisc_policy_return,num_steps = rollout_policy(
                                                                        agent,env,exploration_rng,
                                                                        replay_buffer,actor_buffer,warmup=warmup,
                                                                        num_rollouts=args.num_rollouts,discount = args.gamma,max_length=args.max_episode_steps)
                                                              
                if not warmup : policy_rollouts.append(policy_rollout)
                unlogged_steps += num_steps
                cached_steps += num_steps
                
                i+=num_steps
                pbar.update(int(num_steps))
                
                if replay_buffer.size >= start_steps:
                
                    ### Update critics ###:
                    logging.debug('update critics')
                    transitions = replay_buffer.get_all()
                    idxs = jax.random.choice(agent.rng,a=transitions['observations'].shape[0], shape=(NUM_UPDATES,256), replace=True)
                    batches = jax.vmap(lambda i: jax.tree_map(lambda x: x[i], transitions))(idxs)
                    agent = agent.update_critics_seq(batches,R2)
                        
                    ### Update critic weights ## 
                    logging.debug('update critic weights')
                    if len(policy_rollouts)>=20 and agent.config["adaptive_critics"]:   
                    
                        flattened_rollouts = flatten_rollouts(policy_rollouts)
                        R2,bias = evaluate_many_critics(agent,policy_rollout.policy_return,flattened_rollouts,args.num_critics)
                        R2_train_info = {'R2/max': jnp.max(R2),'R2/bias': bias[jnp.argmax(R2)],
                                        "R2/histogram": wandb.Histogram(jnp.clip(R2,a_min=-1,a_max=1)),
                                        }
                        wandb.log(R2_train_info, step=int(i),commit=False)
                    
                    ### Update actor ###
                    actor_batch = actor_buffer.get_all()    
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
                        
                        # _,_,policy_rollout,policy_return,variance,undisc_policy_return,num_steps = rollout_policy(
                        #                                                 agent,eval_env,exploration_rng,
                        #                                                 None,None,warmup=False,
                        #                                                 num_rollouts=10,random=True,
                        #                                                 discount = args.gamma,max_length=1000)
                        
                        policy_fn = partial(supply_rng(agent.sample_actions), temperature=1.)
                        eval_metrics = evaluate(policy_fn, eval_env, num_episodes=10)

                        #eval_metrics = {"policy_return": policy_return,"std": jnp.sqrt(variance),"undisc_policy_return": undisc_policy_return}
                        eval_metrics = {f'evaluation/{k}': v for k, v in eval_metrics.items()}
                        eval_metrics['n_grads']=int(n_grads)
                        wandb.log(eval_metrics, step=int(i),commit=True)
                        unlogged_steps = 0
                
                    if cached_steps >= int(1e6): 
                        jax.clear_caches()
                        cached_steps = 0
                        print('clearing cache')
            
    wandb_run.finish()

train(args)
#%%
