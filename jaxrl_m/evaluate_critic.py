
from jaxrl_m.rollout import PolicyRollout
import jax
import jax.numpy as jnp
from functools import partial

def f(anc_agent,obs,actor_params,critic_params,seed):

    dist = anc_agent.actor(obs, params=actor_params)
    actions, _ = dist.sample_and_log_prob(seed=seed)
    q,_ = anc_agent.critic(obs, actions,params=critic_params)
   
    return q
    
    
@jax.jit
def estimate_return(acq_rollout,
                    anc_agent,anc_critic_params,anc_return,seed):
    
    acq_obs = jnp.repeat(acq_rollout.observations,10,axis=0)
    acq_disc_masks = jnp.repeat(acq_rollout.disc_masks,10,axis=0)
    
    
    acq_actor = acq_rollout.policy_params
    acq_return = acq_rollout.policy_return
    anc_actor = anc_agent.actor.params
    
    acq_q = f(anc_agent,acq_obs,acq_actor,anc_critic_params,seed)
    anc_q = f(anc_agent,acq_obs,anc_actor,anc_critic_params,seed)
    
    adv = ((acq_q - anc_q)*acq_disc_masks).sum()/(acq_rollout.num_rollouts *10)
    acq_return_pred = anc_return + adv
    
    return acq_return_pred,acq_return


@jax.jit
def evaluate_one_critic(anc_critic_params,
                        anc_agent,
                        anc_return,policy_rollouts,seed):
    
    predict_rollout =  partial(estimate_return,
                   anc_agent=anc_agent,
                   anc_critic_params =anc_critic_params,
                   anc_return = anc_return,seed=seed)
    y_pred,y = jax.vmap(predict_rollout)(policy_rollouts)
    variances = policy_rollouts.variance
    weights = 1/variances
    a2 = (weights * ((y-y_pred)**2)).sum()
    b2 = (weights * ((y-y.mean())**2)).sum()
    b2=jnp.clip(b2,1e-8)
    R2 = 1-(a2/b2)  
    bias = (y_pred-y).mean()
    
    return R2,bias

#@jax.jit
import jax.tree_util

def evaluate_many_critics(anc_agent, anc_return, policy_rollouts):
    
    seed = anc_agent.rng
    anc_critic_params = anc_agent.critic.params
    num_critics = 5 ###HOTFIX
    tmp = partial(evaluate_one_critic,
                anc_agent=anc_agent,
                anc_return=anc_return,
                policy_rollouts=policy_rollouts, seed=seed)

    R2_l, bias_l = [], []
    for i in range(num_critics):
        critic_params = jax.tree_map(lambda x: x[i], anc_critic_params)
        R2, bias = tmp(critic_params)
        R2_l.append(R2)
        bias_l.append(bias)
    
    R2 = jnp.vstack(R2_l)
    bias = jnp.vstack(bias_l)
    
    return R2, bias
