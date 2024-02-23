import jax
from flax import struct
import chex
import numpy as np 
import jax.numpy as jnp


@struct.dataclass
class PolicyRollout:
    
    policy_params : chex.Array    
    num_rollouts : chex.Array 
    policy_return : chex.Array
    variance : chex.Array
    observations : chex.Array
    disc_masks : chex.Array
    #policy_entropy : chex.Array = jnp.Array(0.,dtype=jnp.float32)
    
    

def rollout_policy(agent,env,exploration_rng,
                   replay_buffer,actor_buffer,
                   warmup=False,num_rollouts=5,random=False):
    
    
    actor_buffer = actor_buffer.reset()
    obs,_ = env.reset()  
    n_steps,n_rollouts,episode_step,disc,mask = 0,0,0,1.,1.
    max_steps = num_rollouts*1000
    observations,disc_masks,rewards = np.zeros((max_steps,obs.shape[0])),np.zeros((max_steps,)),np.zeros((max_steps,))
    policy_returns = np.zeros((num_rollouts,))
    
    while n_rollouts < num_rollouts:
        
        if warmup:
            action = env.action_space.sample()
        else:
            exploration_rng, key = jax.random.split(exploration_rng)
            action = agent.sample_actions(obs,seed=exploration_rng,random=random)
        
        next_obs, reward, done, truncated, info = env.step(action)
        
        mask = float(not done)

        transition = dict(observations=obs,actions=action,
            rewards=reward,masks=mask,next_observations=next_obs,discounts=disc)
        
        replay_buffer.add_transition(transition)
        actor_buffer.add_transition(transition)
    
        observations[1000*n_rollouts+episode_step] = obs
        disc_masks[1000*n_rollouts+episode_step] = disc
        rewards[1000*n_rollouts+episode_step] = reward
        
        obs = next_obs
        disc *= (0.99*mask)
        episode_step += 1
        n_steps += 1
        
        if (done or truncated) :
            policy_returns[n_rollouts] = (disc_masks[1000*n_rollouts:1000*(n_rollouts+1)]*rewards[1000*n_rollouts:1000*(n_rollouts+1)]).sum()
            obs,_= env.reset()
            n_rollouts += 1
            episode_step = 0
            disc,mask = 1.,1.
            

    policy_return = policy_returns.mean()
    variance = policy_returns.var()
    undisc_policy_return = (rewards).sum()/num_rollouts
    policy_rollout = PolicyRollout( policy_params=agent.actor.params,
                                    policy_return=policy_return,
                                    variance=variance,
                                    observations=observations,
                                    disc_masks=disc_masks,
                                    num_rollouts=jnp.array(num_rollouts))
    
    return replay_buffer,actor_buffer,policy_rollout,policy_return,variance,undisc_policy_return,n_steps


def rollout_policy2(agent,env,exploration_rng,
                   replay_buffer,actor_buffer,
                   warmup=False,num_rollouts=5,random=False):
    
    
    actor_buffer.reset()
    obs,_ = env.reset()  
    n_steps,n_rollouts,episode_step,disc,mask = 0,0,0,1.,1.
    old_last_done,last_done = 0,0
    max_steps = num_rollouts*1000
    observations,disc_masks,masks,rewards = np.zeros((max_steps,obs.shape[0])),np.zeros((max_steps,)),np.zeros((max_steps,)),np.zeros((max_steps,))
    policy_returns = []
    
    while n_steps < max_steps:
        
    
        exploration_rng, key = jax.random.split(exploration_rng)
        action = agent.sample_actions(obs,exploration_rng,random)
    
        next_obs, reward, done, truncated, info = env.step(action)
        mask = float(not done)

        transition = dict(observations=obs,actions=action,
            rewards=reward,masks=mask,next_observations=next_obs,discounts=disc)
        
        replay_buffer.add_transition(transition)
        actor_buffer.add_transition(transition)
    
        observations[n_steps] = obs
        disc_masks[n_steps] = disc*mask
        rewards[n_steps] = reward
        masks[n_steps] = mask
        
        obs = next_obs
        disc *= (0.99*mask)
        episode_step += 1
        n_steps += 1
        
        if (done or truncated) :
            policy_returns.append((disc_masks[last_done:last_done+episode_step]*rewards[last_done:last_done+episode_step]).sum())
            obs,_= env.reset()
            n_rollouts += 1
            episode_step = 0
            disc,mask = 1.,1.
            last_done = n_steps
            

    
    
    
    
    
    observations = observations[:last_done]
    observations = jnp.pad(observations, ((0, max_steps - len(observations)), (0, 0)), mode='constant', constant_values=0)
    
    disc_masks = disc_masks[:last_done]
    disc_masks = jnp.pad(disc_masks, (0, max_steps - len(disc_masks)), mode='constant', constant_values=0)
    
    rewards = rewards[:last_done]
    rewards = jnp.pad(rewards, (0, max_steps - len(rewards)), mode='constant', constant_values=0)
    
    policy_returns = jnp.array(policy_returns)
    
    undisc_policy_return = (masks*rewards).sum()/n_rollouts

    # policy_rollout = PolicyRollout(policy_params=agent.actor.params,
    #                                policy_return=policy_return,
    #                                observations=observations,
    #                                disc_masks=disc_masks,
    #                                num_rollouts=jnp.array(n_rollouts))

    # return replay_buffer, actor_buffer, policy_rollout, policy_return, undisc_policy_return, n_steps
 
    policy_return = policy_returns.mean()
    variance = policy_returns.var()
    #print(f'policy_returns {policy_returns} policy_return {policy_return} variance {variance} undisc_policy_return {undisc_policy_return}')
    undisc_policy_return = (rewards).sum()/n_rollouts
    policy_rollout = PolicyRollout( policy_params=agent.actor.params,
                                    policy_return=policy_return,
                                    variance=variance,
                                    observations=observations,
                                    disc_masks=disc_masks,
                                    num_rollouts=jnp.array(n_rollouts))
    
    return replay_buffer,actor_buffer,policy_rollout,policy_return,variance,undisc_policy_return,n_steps
