
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax

from jaxrl_m.common import TrainState, nonpytree_field
from jaxrl_m.networks import OriginalCritic, Policy
from jaxrl_m.typing import *


def get_batch(i,batches):
    return  jax.tree_map(lambda x: x[i], batches)

def body(i,val):
    agent,batches = val
    return (agent.update_critics(get_batch(i,batches)),batches)

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
        agent,batches = jax.lax.fori_loop(0,2500,body,(agent,batches))
        
        return agent

    @jax.jit
    def update_actor(agent, batch: Batch,R2):

        new_rng, curr_key, next_key = jax.random.split(agent.rng, 3)

        def actor_loss_fn(actor_params,R2):
            # observations = jnp.repeat(batch['observations'], 10, axis=0)
            # discounts = jnp.repeat(batch['discounts'], 10, axis=0)
            # masks = jnp.int32(jnp.repeat(batch['masks'], 10, axis=0))

            observations = batch['observations']
            discounts = batch['discounts']
            masks = batch['masks']

            dist = agent.actor(observations, params=actor_params)
            actions, log_probs = dist.sample_and_log_prob(seed=curr_key)
            #log_probs = dist.log_prob(actions)
            call_one_critic = lambda observations,actions,params: agent.critic(observations,actions,params=params)
            q_all = jax.vmap(call_one_critic,in_axes=(None,None,0))(observations, actions,agent.critic.params)##critic_update_info
            
            q_weights = jax.nn.softmax(R2,axis=0)
            q = jnp.sum(q_weights.reshape(-1,1)*q_all,axis=0)

            
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
        temp_def = Temperature()
        temp_params = temp_def.init(rng)['params']
        
        
        if use_momentum:
            temp = TrainState.create(temp_def, temp_params, tx=optax.adam(learning_rate=temp_lr))
            actor = TrainState.create(actor_def, actor_params, tx=optax.adam(learning_rate=actor_lr))
            
        else:
            temp = TrainState.create(temp_def, temp_params, tx=optax.adam(learning_rate=temp_lr,b1=0))
            actor = TrainState.create(actor_def, actor_params, tx=optax.adam(learning_rate=actor_lr,b1=0))
            # temp = TrainState.create(temp_def, temp_params, tx=optax.rmsprop(learning_rate=temp_lr))
            # actor = TrainState.create(actor_def, actor_params, tx=optax.rmsprop(learning_rate=actor_lr))
            
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
            
        ))

        return SACAgent(rng, critic=critics, target_critic=critics, actor=actor, temp=temp, config=config)

