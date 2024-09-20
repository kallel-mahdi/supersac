
from jaxrl_m.rollout import PolicyRollout
import jax
import jax.numpy as jnp
from functools import partial
import jax.tree_util
from jaxrl_m.networks import OriginalCritic

def f(anc_agent,obs,actor_params,critic_params,seed):


    params = {'params':critic_params}
    #actions, log_p,_ = anc_agent.sample_actions(obs,seed=seed)
    
    
    #q = anc_agent.critic(obs, actions,params=critic_params)
    
    dist = anc_agent.actor(obs,params=actor_params)
    pre_actions,_ = dist.sample_and_log_prob(seed=seed)

    if anc_agent.config["tanh_squash_actions"]:
        actions = jax.nn.tanh(pre_actions)

    else : 
        actions = pre_actions
    
    q = OriginalCritic(hidden_dims=(256,256)).apply(params,obs, actions)
   
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



#@jax.jit
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
    b2=jnp.clip(b2,1e-6)
    R2 = 1-(a2/b2)  
    bias = (y_pred-y).mean()
    
    return R2,bias

#@partial(jax.jit, static_argnums=(3,))
def evaluate_many_critics(anc_agent, anc_return, policy_rollouts,num_critics):
    
    seed = anc_agent.rng
    anc_critic_params = anc_agent.critic.params
    tmp = partial(evaluate_one_critic,
                anc_agent=anc_agent,
                anc_return=anc_return,
                policy_rollouts=policy_rollouts, seed=seed)

    ### Evaluating over all critics causes O.O.M error
    ### We do it sequentially as it's not a bottleneck
    R2_l, bias_l = [], []
    for i in range(num_critics):
        critic_params = jax.tree.map(lambda x: x[i], anc_critic_params)
        R2, bias = tmp(critic_params)
        R2_l.append(R2)
        bias_l.append(bias)
    
    R2 = jnp.vstack(R2_l)
    bias = jnp.vstack(bias_l)
    
    #R2, bias = jax.vmap(tmp)(anc_critic_params)
    
    return R2, bias
