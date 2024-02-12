
from jaxrl_m.rollout import PolicyRollout
import jax
import jax.numpy as jnp
from functools import partial

def f(anc_agent,obs,actor_params,critic_params,temp_critic_params,seed):

    dist = anc_agent.actor(obs, params=actor_params)
    actions, _ = dist.sample_and_log_prob(seed=seed)
    q = anc_agent.critic(obs, actions,params=critic_params)
    q_temp = anc_agent.temp_critic(obs, actions,params=temp_critic_params)
    
    q = q + anc_agent.temp()*q_temp
    
    return q
    
    
@jax.jit
def estimate_return(acq_rollout,
                    anc_agent,anc_critic_params,anc_temp_critic_params,anc_return,seed):
    
    acq_obs = jnp.repeat(acq_rollout.observations,10,axis=0)
    acq_disc_masks = jnp.repeat(acq_rollout.disc_masks,10,axis=0)
    
    
    acq_actor = acq_rollout.policy_params
    acq_return = acq_rollout.policy_return
    anc_actor = anc_agent.actor.params
    
    acq_q = f(anc_agent,acq_obs,acq_actor,anc_critic_params,anc_temp_critic_params,seed)
    anc_q = f(anc_agent,acq_obs,anc_actor,anc_critic_params,anc_temp_critic_params,seed)
    
    adv = ((acq_q - anc_q)*acq_disc_masks).sum()/(acq_rollout.num_rollouts *10)
    acq_return_pred = anc_return + adv
    
    return acq_return_pred,acq_return


@jax.jit
def evaluate_one_critic(anc_critic_params,temp_critic_params,
                        anc_agent,
                        anc_return,policy_rollouts,seed):
    
    predict_rollout =  partial(estimate_return,
                   anc_agent=anc_agent,
                   anc_critic_params =anc_critic_params,
                   anc_temp_critic_params = temp_critic_params,
                   anc_return = anc_return,seed=seed)
    y_pred,y = jax.vmap(predict_rollout)(policy_rollouts)
    a2 = ((y-y_pred)**2).sum()
    b2=jnp.clip(((y-y.mean())**2).sum(),1e-8)
    R2 = 1-(a2/b2)  
    bias = (y_pred-y).mean()
    
    return R2,bias

#@jax.jit
import jax.tree_util

def evaluate_many_critics(anc_agent, anc_return, policy_rollouts):
    
    seed = anc_agent.rng
    anc_critic_params = anc_agent.critic.params
    
    #num_critics = len(jax.tree_util.tree_leaves(anc_critic_params))
    num_critics = 5 ###HOTFIX
    #print(f'num_critics: {num_critics}')
    tmp = partial(evaluate_one_critic,
                anc_agent=anc_agent,
                anc_return=anc_return,
                policy_rollouts=policy_rollouts, seed=seed)

    R2_l, bias_l = [], []
    for i in range(num_critics):
        critic_params = jax.tree_map(lambda x: x[i], anc_critic_params)
        temp_critic_params = jax.tree_map(lambda x: x[i], anc_agent.temp_critic.params)
        R2, bias = tmp(critic_params,temp_critic_params)
        R2_l.append(R2)
        bias_l.append(bias)
    
    R2 = jnp.vstack(R2_l)
    bias = jnp.vstack(bias_l)
    
    return R2, bias
