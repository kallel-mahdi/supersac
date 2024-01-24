import functools
import jax
import jax.numpy as jnp
from functools import partial


def compute_q(anc_agent,obs,actor_params,critic_params):

    actions = anc_agent.actor(obs, params=actor_params)
    q = anc_agent.critic(obs, actions,False,params=critic_params)
   
    return q

def estimate_return(acq_rollout,
                    anc_agent,anc_critic_params,anc_return):
    
    acq_obs = acq_rollout.observations
    acq_masks = acq_rollout.disc_masks
    acq_return = acq_rollout.policy_return
  
    anc_actor_params = anc_agent.actor.params
    acq_actor_params = acq_rollout.policy_params
    
    anc_q = compute_q(anc_agent,acq_obs,anc_actor_params,anc_critic_params)
    acq_q = compute_q(anc_agent,acq_obs,acq_actor_params,anc_critic_params)
    
    acq_adv = ((acq_q - anc_q)*acq_masks).sum()/acq_rollout.num_rollouts
    acq_return_pred = anc_return + acq_adv
  
    
    return acq_return_pred,acq_adv,acq_return


def evaluate_critic(anc_critic_params,anc_agent,
                    anc_return,policy_rollouts):

    
    tmp =  partial(estimate_return,
                   anc_agent=anc_agent,
                   anc_critic_params =anc_critic_params,
                   anc_return = anc_return)
    y_pred,adv,y = jax.vmap(tmp)(policy_rollouts)
    a2 = ((y-y_pred)**2).sum()
    b2=jnp.clip(((y-y.mean())**2).sum(),1e-8)
    R2 = 1-(a2/b2)  
    bias = (y_pred-y).mean()
    ### Upperbound for offline RL ###
    anc_return_e = (y-adv).mean()
    tmp =  partial(estimate_return,
                   anc_agent=anc_agent,
                   anc_critic_params =anc_critic_params,
                   anc_return = anc_return_e)
    y_pred,_,y = jax.vmap(tmp)(policy_rollouts)
    a2 = jnp.clip(((y-y_pred)**2),a_min=1e-4).sum()
    b2=((y-y.mean())**2).sum()
    R2_bound = 1-(a2/b2)  

    return R2,bias,R2_bound,anc_return_e

@jax.jit
def train_evaluation(anc_agent,anc_return,policy_rollouts):
    
    anc_critic_params = anc_agent.critic.params
    R2,bias,R2_bound,anc_return_e = jax.vmap(evaluate_critic,in_axes=(0,None,None,None))(anc_critic_params,anc_agent,anc_return,policy_rollouts)
    
    return R2,bias,R2_bound,anc_return_e


@jax.jit
def test_evaluation(anc_agent,anc_critic_params,
                    anc_return,anc_return_e,
                    policy_rollouts):
    
    R2_test_bound,_,_,_ = jax.vmap(evaluate_critic,in_axes=(0,None,0,None))(anc_critic_params,anc_agent,anc_return_e,policy_rollouts)
    R2_test,_,_,_ = jax.vmap(evaluate_critic,in_axes=(0,None,None,None))(anc_critic_params,anc_agent,anc_return,policy_rollouts)
    
    return R2_test,R2_test_bound




def merge(x,y):

    return jax.tree_map(lambda x,y : jnp.vstack([x,y]),x,y)

def flatten_rollouts(policy_rollouts):
    
    n_policies = len(policy_rollouts)
    #print(f'len_policy_rollouts: {n_policies}')
    if n_policies == 1:
        policy_rollouts.append(policy_rollouts[0]) ## HOTFIX
        n_policies = 2
    
    merged_rollouts = functools.reduce(merge, policy_rollouts)
    merged_rollouts = jax.tree_map(lambda x:jnp.stack(jnp.split(x,n_policies,axis=0)),merged_rollouts)
    
    def reshape_tree(tree, reference_tree,n_policies):
        def reshape_fn(x, reference_x):
            return jnp.reshape(x, (n_policies,*reference_x.shape))
        
        return jax.tree_map(reshape_fn, tree, reference_tree)
    
    merged_rollouts = reshape_tree(merged_rollouts,policy_rollouts[0],n_policies)
    
    return merged_rollouts

def split_rollouts(flattened_rollouts,MAX_SIZE):
    
    key = jax.random.PRNGKey(0)
    max = flattened_rollouts.policy_return.shape[0]
    size = jnp.minimum(max,MAX_SIZE)

    idxs = jax.random.choice(key,a=max, shape=(size,), replace=False)   
    train_idxs = idxs[:int(0.8*size)]
    test_idxs = idxs[int(0.8*size):]
    train_rollouts = jax.tree_map(lambda x : x[train_idxs],flattened_rollouts)
    test_rollouts = jax.tree_map(lambda x : x[test_idxs],flattened_rollouts)
    
    return train_rollouts,test_rollouts


def measure_action_distance(agent,new_params,old_params,observations):
    
    a_new = agent.actor(observations,params=new_params)
    a_old = agent.actor(observations,params=old_params)
    
    return jnp.linalg.norm(a_new-a_old,axis=-1).mean()
