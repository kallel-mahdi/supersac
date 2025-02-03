import jax
from flax import struct
import chex
import numpy as np 
import jax.numpy as jnp

def rollout_policy(agent,env,exploration_rng,
                   replay_buffer=None,actor_buffer=None,
                   eval=False,discount=0.99,max_rollouts=10):
    
    
    
    
    
    
    n_steps,n_rollouts,disc,mask = 0,0,1.,1.
    policy_returns,returns = [],[]
    policy_return,undisc_return = 0.,0.
    
    if actor_buffer is not None: actor_buffer = actor_buffer.reset()
    
    obs,_ = env.reset()  
    
    while n_rollouts < max_rollouts:
        
        if eval:            
            action = agent.deterministic_action(obs)
            log_p,pre_action = 0.,action
        else:
            exploration_rng, key = jax.random.split(exploration_rng)
            action,log_p,pre_action = agent.sample_actions(obs,seed=exploration_rng)
            
        
        next_obs, reward, done, truncated, info = env.step(action)
        
        if "episode" in info.keys():
            
                returns.append(info["episode"]["r"])
                n_rollouts+=1
    
        policy_return += reward * disc
        
        mask = float(not done)

        transition = dict(observations=obs,actions=action,
            rewards=reward,masks=mask,truncateds=truncated,next_observations=next_obs,discounts=disc,
            log_probs=log_p,pre_actions=pre_action)

        if replay_buffer is not None:
            replay_buffer.add_transition(transition)
        
        if actor_buffer is not None:
            actor_buffer.add_transition(transition)
    
        obs = next_obs
        disc *= (discount*mask)
        n_steps += 1
        
        if (done or truncated) :
            policy_returns.append(policy_return)
            policy_return = 0.
            obs,_= env.reset()
            disc,mask = 1.,1.
            
    policy_returns = np.array(policy_returns)
    policy_return = policy_returns.mean()
    undisc_return = np.array(returns).mean()
    
  
    return replay_buffer,actor_buffer,policy_return,undisc_return,n_steps


def rollout_policy2(agent,env,exploration_rng,
                   replay_buffer=None,actor_buffer=None,
                   eval=False,discount=0.99,max_steps=5120):
    
    
    
    
    
    
    n_steps,n_rollouts,disc,mask = 0,0,1.,1.
    policy_returns,returns = [],[]
    policy_return,undisc_return = 0.,0.
    
    if actor_buffer is not None: actor_buffer = actor_buffer.reset()
    
    obs,_ = env.reset()  
    
    while n_steps < max_steps:
        
        if eval:            
            action = agent.deterministic_action(obs)
            log_p,pre_action = 0.,action
        else:
            exploration_rng, key = jax.random.split(exploration_rng)
            action,log_p,pre_action = agent.sample_actions(obs,seed=exploration_rng)
            
        
        next_obs, reward, done, truncated, info = env.step(action)
        
        if "episode" in info.keys():
            
                returns.append(info["episode"]["r"])
                n_rollouts+=1
    
        policy_return += reward * disc
        
        mask = float(not done)

        transition = dict(observations=obs,actions=action,
            rewards=reward,masks=mask,truncateds=truncated,next_observations=next_obs,discounts=disc,
            log_probs=log_p,pre_actions=pre_action)

        if replay_buffer is not None:
            replay_buffer.add_transition(transition)
        
        if actor_buffer is not None:
            actor_buffer.add_transition(transition)
    
        obs = next_obs
        disc *= (discount*mask)
        n_steps += 1
        
        if (done or truncated) :
            policy_returns.append(policy_return)
            policy_return = 0.
            obs,_= env.reset()
            disc,mask = 1.,1.
            
    policy_returns = np.array(policy_returns)
    policy_return = policy_returns.mean()
    undisc_return = np.array(returns).mean()
    
  
    return replay_buffer,actor_buffer,policy_return,undisc_return,n_steps



def rollout_policy_lqr(agent,env,exploration_rng,
                   replay_buffer=None,actor_buffer=None,
                   eval=False,num_rollouts=5,discount=0.99,max_length=500):
    
    if actor_buffer is not None:
        actor_buffer = actor_buffer.reset()
    obs = env.reset()  
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
            
        #next_obs, reward, done, truncated, info = env.step(action)
        next_obs, reward, done, info = env.step(action)
        
        mask = float(not done)

        transition = dict(observations=obs,actions=action,
            rewards=reward,masks=mask,next_observations=next_obs,discounts=disc,
            log_probs=log_p,pre_actions=pre_action)
        
        if replay_buffer is not None:
            replay_buffer.add_transition(transition)
        
        if actor_buffer is not None:
            actor_buffer.add_transition(transition)

       
        
        obs = next_obs
        disc *= (discount*mask)
        episode_step += 1
        n_steps += 1
        
        if done or n_steps%max_length==0:
            policy_returns[n_rollouts] = (disc_masks[max_length*n_rollouts:max_length*(n_rollouts+1)]*rewards[max_length*n_rollouts:max_length*(n_rollouts+1)]).sum()
            obs = env.reset()
            n_rollouts += 1
            episode_step = 0
            disc,mask = 1.,1.
          
            
            
    policy_return = policy_returns.mean()
    undisc_policy_return = rewards.sum()/num_rollouts
   
    
    return replay_buffer,actor_buffer,policy_return,undisc_policy_return,n_steps




def rollout_policy_ppo(agent,env,num_rollouts=5,discount=0.99,max_length=500):
    
    
    policy_returns,returns = [],[]
    obs,_ = env.reset()  
    n_rollouts = 0
    
    while n_rollouts < num_rollouts:
        
    
        action = agent.deterministic_action(obs)
        
        next_obs, reward, done, truncated, info = env.step(action)
        
        obs = next_obs
    
        if "episode" in info.keys():
                
                returns.append(info["episode"]["r"])
                n_rollouts+=1
        
        if (done or truncated):
        
            obs,_= env.reset()
    
    return np.mean(np.array(returns))
