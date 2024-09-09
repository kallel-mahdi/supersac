import jax
from flax import struct
import chex
import numpy as np 
import jax.numpy as jnp





from typing import Callable

import gymnasium as gym





@struct.dataclass
class PolicyRollout:
    
    policy_params : chex.Array    
    num_rollouts : chex.Array 
    policy_return : chex.Array
    variance : chex.Array
    observations : chex.Array
    disc_masks : chex.Array
    #policy_entropy : chex.Array = jnp.Array(0.,dtype=jnp.float32)
    
    

from jaxrl_m.rollout import PolicyRollout
import jax
import jax.numpy as jnp



def rollout_policy(agent,env,exploration_rng,
                   replay_buffer=None,actor_buffer=None,
                   eval=False,num_rollouts=5,discount=0.99,max_length=500):
    
    if actor_buffer is not None:
        actor_buffer = actor_buffer.reset()
    obs,_ = env.reset()  
    n_steps,n_rollouts,episode_step,disc,mask = 0,0,0,1.,1.
    max_steps = num_rollouts*max_length
    observations,disc_masks,rewards = np.zeros((max_steps,obs.shape[0])),np.zeros((max_steps,)),np.zeros((max_steps,))
    policy_returns = np.zeros((num_rollouts,))
    
    while n_rollouts < num_rollouts:
        
        if eval:
            action = agent.deterministic_action(obs)
            log_p,pre_action = 0.,action
        else:
            exploration_rng, key = jax.random.split(exploration_rng)
            action,log_p,pre_action = agent.sample_actions(obs,seed=exploration_rng)
            
            
        
        next_obs, reward, done, truncated, info = env.step(action)
        
        mask = float(not done)

        transition = dict(observations=obs,actions=action,
            rewards=reward,masks=mask,next_observations=next_obs,discounts=disc,
            log_probs=log_p,pre_actions=pre_action)
        
        if replay_buffer is not None:
            replay_buffer.add_transition(transition)
        
        if actor_buffer is not None:
            actor_buffer.add_transition(transition)
    
        observations[max_length*n_rollouts+episode_step] = obs
        disc_masks[max_length*n_rollouts+episode_step] = disc
        rewards[max_length*n_rollouts+episode_step] = reward
        
        obs = next_obs
        disc *= (discount*mask)
        episode_step += 1
        n_steps += 1
        
        if (done or truncated) :
            policy_returns[n_rollouts] = (disc_masks[max_length*n_rollouts:max_length*(n_rollouts+1)]*rewards[max_length*n_rollouts:max_length*(n_rollouts+1)]).sum()
            obs,_= env.reset()
            n_rollouts += 1
            episode_step = 0
            disc,mask = 1.,1.
            
            
    policy_return = policy_returns.mean()
    variance = policy_returns.var()
    undisc_policy_return = rewards.sum()/num_rollouts
    policy_rollout = PolicyRollout( policy_params=agent.actor.params,
                                    policy_return=policy_return,
                                    variance=variance,
                                    observations=observations,
                                    disc_masks=disc_masks,
                                    num_rollouts=jnp.array(num_rollouts))
    
    return replay_buffer,actor_buffer,policy_rollout,policy_return,variance,undisc_policy_return,n_steps



def rollout_policy_ppo(agent,env,num_rollouts=5,discount=0.99,max_length=500):
    
    
    returns = []
    obs,_ = env.reset()  
    n_rollouts = 0
    
    while n_rollouts < num_rollouts:
        
    
        action = agent.deterministic_action(obs)
        log_p,pre_action = 0.,action
    
        
        next_obs, reward, done, truncated, info = env.step(action)
        
        obs = next_obs
    
        if "episode" in info.keys():
                
                returns.append(info["episode"]["r"])
                n_rollouts+=1
    
    return np.mean(np.array(returns))

def rollout_policy2(agent,env,exploration_rng,
                   replay_buffer=None,actor_buffer=None,
                   eval=False,num_rollouts=5,discount=0.99,max_length=500):
    
    
    
    #print(f'discount {discount}')
    actor_buffer = actor_buffer.reset()
    obs,_ = env.reset()  
    n_steps,n_rollouts,episode_step,disc,mask = 0,0,0,1.,1.
    max_steps = num_rollouts*max_length
    observations,disc_masks,rewards = np.zeros((max_steps,obs.shape[0])),np.zeros((max_steps,)),np.zeros((max_steps,))
    
    policy_returns,undisc_returns = [],[]
    policy_return,undisc_return = 0.,0.
    while n_steps < max_steps:
        
        if eval:
            #action = env.action_space.sample()
            action = agent.deterministic_action(obs)
            log_p,pre_action = 0.,action
        else:
            exploration_rng, key = jax.random.split(exploration_rng)
            action,log_p,pre_action = agent.sample_actions(obs,seed=exploration_rng)
            
        
        next_obs, reward, done, truncated, info = env.step(action)
        
        policy_return += reward * disc
        undisc_return += reward
        
        mask = float(not done)

        transition = dict(observations=obs,actions=action,
            rewards=reward,masks=mask,next_observations=next_obs,discounts=disc,
            log_probs=log_p,pre_actions=pre_action)

        replay_buffer.add_transition(transition)
        actor_buffer.add_transition(transition)
    
        observations[n_steps] = obs
        disc_masks[n_steps] = disc
        rewards[n_steps] = reward
         
        obs = next_obs
        disc *= (discount*mask)
        episode_step += 1
        n_steps += 1
        
        if (done or truncated) :
            #policy_returns(disc_masks[max_length*n_rollouts:max_length*(n_rollouts+1)]*rewards[max_length*n_rollouts:max_length*(n_rollouts+1)]).sum()
            policy_returns.append(policy_return)
            undisc_returns.append(undisc_return)
            policy_return,undisc_return = 0.,0.
            obs,_= env.reset()
            n_rollouts += 1
            episode_step = 0
            disc,mask = 1.,1.
            

    policy_returns = np.array(policy_returns)
    policy_return = policy_returns.mean()
    variance = policy_returns.var()
    undisc_return = np.array(undisc_returns).mean()
   
    
    #policy_rollout = None
    
    policy_rollout = PolicyRollout( policy_params=agent.actor.params,
                                    policy_return=policy_return,
                                    variance=variance,
                                    observations=observations,
                                    disc_masks=disc_masks,
                                    num_rollouts=jnp.array(num_rollouts))
    
    return replay_buffer,actor_buffer,policy_rollout,policy_return,variance,undisc_return,n_steps



def rollout_policy_lqr(agent,env,exploration_rng,
                   replay_buffer=None,actor_buffer=None,
                   warmup=False,num_rollouts=5,discount=0.99,max_length=500):
    
    if actor_buffer is not None:
        actor_buffer = actor_buffer.reset()
    obs = env.reset()  
    n_steps,n_rollouts,episode_step,disc,mask = 0,0,0,1.,1.

    max_steps = num_rollouts*max_length
    observations,disc_masks,rewards = np.zeros((max_steps,obs.shape[0])),np.zeros((max_steps,)),np.zeros((max_steps,))
    policy_returns = np.zeros((num_rollouts,))
    
    while n_rollouts < num_rollouts:
        
        
        if warmup:
            action = env.action_space.sample()
        else:
            exploration_rng, key = jax.random.split(exploration_rng)
            action = agent.sample_actions(obs,seed=exploration_rng)
        
        next_obs, reward, done, info = env.step(action)
        
        mask = float(not done)

        transition = dict(observations=obs,actions=action,
            rewards=reward,masks=mask,next_observations=next_obs,discounts=disc)
        
        if replay_buffer is not None:
            replay_buffer.add_transition(transition)
        
        if actor_buffer is not None:
            actor_buffer.add_transition(transition)

        #print("n_rollouts",n_rollouts,"done",done)
    
        observations[max_length*n_rollouts+episode_step] = obs
        disc_masks[max_length*n_rollouts+episode_step] = disc
        rewards[max_length*n_rollouts+episode_step] = reward
        
        obs = next_obs
        disc *= (discount*mask)
        episode_step += 1
        n_steps += 1
        
        if done or n_steps%500==0:
            policy_returns[n_rollouts] = (disc_masks[max_length*n_rollouts:max_length*(n_rollouts+1)]*rewards[max_length*n_rollouts:max_length*(n_rollouts+1)]).sum()
            obs = env.reset()
            n_rollouts += 1
            episode_step = 0
            disc,mask = 1.,1.
          
            
            
    policy_return = policy_returns.mean()
    variance = policy_returns.var()
    undisc_policy_return = rewards.sum()/num_rollouts
    policy_rollout = PolicyRollout( policy_params=agent.actor.params,
                                    policy_return=policy_return,
                                    variance=variance,
                                    observations=observations,
                                    disc_masks=disc_masks,
                                    num_rollouts=jnp.array(num_rollouts))
    
    return replay_buffer,actor_buffer,policy_rollout,policy_return,variance,undisc_policy_return,n_steps



