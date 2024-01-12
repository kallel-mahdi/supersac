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
    observations : chex.Array
    disc_masks : chex.Array
    

def rollout_policy(agent,env,exploration_rng,
                   replay_buffer,actor_buffer,
                   warmup=False,num_rollouts=5,):
    
    
    actor_buffer.reset()
    obs,_ = env.reset()  
    n_steps,n_rollouts,episode_step,disc,mask = 0,0,0,1.,1.
    max_steps = num_rollouts*1000
    observations,disc_masks,rewards = np.zeros((max_steps,obs.shape[0])),np.zeros((max_steps,)),np.zeros((max_steps,))
    
    
    while n_rollouts < num_rollouts:
        
        if warmup:
            action = env.action_space.sample()
        else:
            exploration_rng, key = jax.random.split(exploration_rng)
            action = agent.sample_actions(obs,seed=exploration_rng)
        
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
            
            obs,_= env.reset()
            n_rollouts += 1
            episode_step = 0
            disc,mask = 1.,1.

    policy_return = (disc_masks*rewards).sum()/num_rollouts
    undisc_policy_return = (rewards).sum()/num_rollouts
    policy_rollout = PolicyRollout(policy_params=agent.actor.params,
                                   policy_return=policy_return,
                                   observations=observations,
                                   disc_masks=disc_masks,
                                    num_rollouts=jnp.array(num_rollouts))
    
    return replay_buffer,actor_buffer,policy_rollout,policy_return,undisc_policy_return,n_steps