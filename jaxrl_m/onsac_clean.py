
import jax.random
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax

from jaxrl_m.common import TrainState, nonpytree_field
from jaxrl_m.networks import OriginalCritic, Policy,ensemblize
from jaxrl_m.typing import *
import jax.lax as lax

def get_batch(i,batches):
    return  jax.tree.map(lambda x: x[i], batches)

def body(i,val):
    agent,batches = val
    return (agent.update_critics(get_batch(i,batches)),batches)

class Temperature(nn.Module):
    
    initial_temperature: float = -3.912 ## (log(0.02))
    
    
    @nn.compact
    def __call__(self) -> jnp.ndarray:
        log_temp = self.param('log_temp',
                              init_fn=lambda key: jnp.full(
                                  (), self.initial_temperature))
        return jnp.exp(log_temp)


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
                        
                        next_actions,next_log_probs,_ = agent.sample_actions(batch["next_observations"],seed=next_key)
                        
                        next_q  = agent.critic(batch['next_observations'], next_actions,params=critic_params)
                        
                        target_q = batch['rewards'] + agent.config['discount'] * batch['masks'] * next_q
                        ### Add entropy
                        target_q = target_q - agent.config['discount'] * batch['masks'] * next_log_probs * agent.temp()
                        target_q = jax.lax.stop_gradient(target_q)
                        
                        
                        if agent.config['min_target']:
                            target_q = jnp.min(target_q,axis=0) ### The change of shape should not pose problem
                           
                        q = agent.critic(batch['observations'], batch['actions'],params=critic_params)
                        critic_loss = ((q-target_q)**2).mean()
                        
                        return critic_loss, {
                        'critic_loss': critic_loss,
                        'q1': q.mean(),
                    }  
                
                new_critic, critic_info = critic.apply_loss_fn(loss_fn=critic_loss_fn, has_aux=True)
                
                return new_critic,critic_info


        new_critics,critic_info = update_one_critic(agent.critic)
        agent = agent.replace(rng=new_rng,critic=new_critics)
        
        return agent
    
    @jax.jit
    def update_critics_seq(agent,batches,R2):
       
    
        ### Train critic 
        size = batches["observations"].shape[0]
        agent,batches = jax.lax.fori_loop(0,size,body,(agent,batches))
        
        return agent

    
    @jax.jit
    def update_actor(agent, batch: Batch,R2):
       
        def temp_loss_fn(temp_params, entropy, target_entropy):
            temperature = agent.temp(params=temp_params)
            entropy_diff = entropy-target_entropy
            temp_loss = (temperature * entropy_diff).mean()
            return temp_loss, {
                'temp_loss': temp_loss,
                'temperature': temperature,
                'entropy_diff': entropy_diff,
            }
        
        def actor_loss_fn(
                actor_params,
                adv,
        ):
            
            ### Compute probability of old actions under new policy
            discounts,masks,logp = batch["discounts"],batch["masks"],batch["log_probs"]
            dist = agent.actor(batch["observations"],params=actor_params)
            pre_actions = batch["pre_actions"]
            pre_log_probs = dist.log_prob(pre_actions)
            
            if agent.config["tanh_squash_actions"]:
                new_logp = pre_log_probs - jnp.sum(2 * (jnp.log(2) - pre_actions - jax.nn.softplus(-2 * pre_actions)), axis=-1)
            
            else : 
                new_logp = pre_log_probs
            
            
            logratio = new_logp - logp
            ratio = jnp.exp(logratio)

            # Calculate how much policy is changing
            approx_kl = ((ratio - 1) - logratio).mean()

            # Policy loss
            clip_coef = agent.config["clipping_ratio"] ##default 0.2 
            actor_loss1 = masks*adv * ratio
            actor_loss2 = masks*adv * jnp.clip(ratio, 1 - clip_coef, 1 + clip_coef)

            if agent.config['discount_actor']:
                actor_loss = -jnp.minimum(discounts*actor_loss1,discounts*actor_loss2).sum()/(discounts.sum())
            else : 
                actor_loss = -jnp.minimum(actor_loss1,actor_loss2).mean()
                
            ### Pad Q and logits because actor buffer is padded ###
            logp = masks * new_logp
            
            if agent.config['discount_entropy']:
                entropy = -1 * (discounts*logp).sum()/(discounts.sum())
            else : 
                entropy = -1 * (masks*logp).sum()/(masks.sum())
            
            return actor_loss, {
                'actor_loss': actor_loss,
                'entropy': entropy,
                'approx_kl':approx_kl
            }
            
     
        
        new_rng, curr_key, next_key = jax.random.split(agent.rng, 3)

        # R2 = R2.reshape(-1,1)
        # R2 = jax.nn.softmax(R2,axis=0)

        observations = batch["observations"]
        
        
        j = 10
        qs,logps = jnp.zeros((observations.shape[0],)),jnp.zeros((observations.shape[0],))
      
        ### Compute value for the fixed states
        
        for i in range(j):
            
            curr_key,_ = jax.random.split(curr_key)
            actions, log_p,_ = agent.sample_actions(observations,seed=curr_key)
            q_all = agent.critic(observations,actions)
            q = jnp.mean(q_all,axis=0)
            qs+=q
            logps+=log_p
                    
        v = qs/j
        h = -(logps/j)
        
        ### Compute advantage for the fixed states AND actions
        q_all = agent.critic(batch["observations"],batch["actions"])
        q = jnp.mean(q_all,axis=0)
        adv = q-v + agent.temp()*(-batch["log_probs"]+h)### This one worked
        
        for i in range(agent.config["num_actor_updates"]):
            
            new_actor, actor_info = agent.actor.apply_loss_fn(actor_loss_fn,True,adv)
            new_temp, temp_info = agent.temp.apply_loss_fn(temp_loss_fn,True,actor_info['entropy'], agent.config['target_entropy'])
            agent = agent.replace(rng=new_rng, actor=new_actor,temp=new_temp)
        
        return agent, {**actor_info,**temp_info}
        
        
    @jax.jit
    def sample_actions(agent,   
                       observations: np.ndarray,
                       seed: PRNGKey,
                       temperature: float = 1.0,
                       ) -> jnp.ndarray:
        
        ### random always true
        dist = agent.actor(observations, temperature=temperature)
        pre_actions,pre_log_ps = dist.sample_and_log_prob(seed=seed)
        
        if agent.config["tanh_squash_actions"]:
            actions = jax.nn.tanh(pre_actions)
            log_ps = pre_log_ps - jnp.sum(2 * (jnp.log(2) - pre_actions - jax.nn.softplus(-2 * pre_actions)), axis=-1)        
        
        else : 
            log_ps = pre_log_ps
            actions = pre_actions
            
        return actions,log_ps,pre_actions
    
    
    @jax.jit
    def deterministic_action(agent,   
                       observations: np.ndarray,
                       ) -> jnp.ndarray:
        
        ### random always true
        seed = jax.random.PRNGKey(0)
        dist = agent.actor(observations, temperature=0.)
        pre_actions,pre_log_ps = dist.sample_and_log_prob(seed=seed)
        if agent.config["tanh_squash_actions"]:
            actions = jax.nn.tanh(pre_actions)
        
        else :
            actions = pre_actions
        
        return actions


def create_learner(
                seed: int,
                observations: jnp.ndarray,
                actions: jnp.ndarray,
                discount: float,
                num_critics: int,
                discount_actor ,
                min_target,
                discount_entropy,
                adaptive_critics,
                entropy_coeff,
                momentum,
                actor_lr,
                critic_lr,
                temp_lr,
                num_actor_updates,
                clipping_ratio,
                actor_hidden_dims: Sequence[int],
                critic_hidden_dims: Sequence[int],
                use_layer_norm : bool,
                target_entropy: float = None,
                state_dependent_std=True,
                tanh_squash_distribution=False,
                tanh_squash_actions=True,
                use_bias=True,
                
            **kwargs):

        print('Extra kwargs:', kwargs)

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key = jax.random.split(rng, 3)

        action_dim = actions.shape[-1]
        actor_def = Policy(actor_hidden_dims, action_dim=action_dim,use_bias=use_bias,
            state_dependent_std=state_dependent_std, tanh_squash_distribution=tanh_squash_distribution,use_layer_norm=use_layer_norm)

        critic_def = ensemblize(OriginalCritic,num_critics)(hidden_dims=critic_hidden_dims)
        critic_params = critic_def.init(critic_key, observations, actions)['params']
        critic = TrainState.create(critic_def, critic_params, tx=optax.adam(learning_rate=critic_lr))

        actor_params = actor_def.init(actor_key, observations)['params']
        temp_def = Temperature()
        temp_params = temp_def.init(rng)['params']
        
        tx = optax.chain(
            optax.clip_by_global_norm(0.5),
            optax.adam(learning_rate=actor_lr,b1=momentum),
        )
        temp = TrainState.create(temp_def, temp_params, tx=optax.adam(learning_rate=temp_lr,b1=momentum))
        actor = TrainState.create(actor_def, actor_params, tx=tx)
            
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
            num_actor_updates = num_actor_updates,
            clipping_ratio = clipping_ratio,
            min_target = min_target,
            tanh_squash_actions=tanh_squash_actions,
            
        ))

        return SACAgent(rng, critic=critic, target_critic=critic, actor=actor, temp=temp, config=config)

