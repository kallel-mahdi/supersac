
import jax.random
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax

from jaxrl_m.common import nonpytree_field
from jaxrl_m.typing import *
from flax.training.train_state import TrainState
from jaxrl_m.networks import OriginalCritic, Policy,ensemblize
import jax.lax as lax
from jaxrl_m.utils import *

def get_batch(i,batches):
    return  jax.tree.map(lambda x: x[i], batches)


def body(i,val):
    agent,batches = val
    return (agent.update_critics(get_batch(i,batches)),batches)

class Temperature(nn.Module):
    initial_temperature: float = 1.
  
    @nn.compact
    def __call__(self) -> jnp.ndarray:

        log_temp = self.param('log_temp',
                    init_fn=lambda key: jnp.full(
                        (), jnp.log(self.initial_temperature) if self.initial_temperature != 0 else -jnp.inf))
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
                        
                        next_q  = agent.critic.apply_fn({'params': critic_params},batch['next_observations'], next_actions)
                        
                        target_q = batch['rewards'] + agent.config['discount'] * batch['masks'] * next_q
                        ### Add entropy
                        target_q = target_q - agent.config['discount'] * batch['masks'] * next_log_probs * agent.temp.apply_fn({'params': agent.temp.params})
                        target_q = jax.lax.stop_gradient(target_q)
                     
                        q = agent.critic.apply_fn({'params': critic_params},batch['observations'], batch['actions'])
                        critic_loss = ((q-target_q)**2).mean()
                        
                        return critic_loss, {
                        'critic_loss': critic_loss,
                        'q1': q.mean(),
                    }  
                        
                grads,info = jax.grad(critic_loss_fn,has_aux=True)(critic.params)
                new_critic = critic.apply_gradients(grads=grads)
                
                return new_critic,info
                
                return new_critic,critic_info


        new_critics,critic_info = update_one_critic(agent.critic)
        agent = agent.replace(rng=new_rng,critic=new_critics)
        
        return agent

    #@partial(jax.jit,static_argnames=("num_updates",))
    
    @jax.jit
    def update_critics_seq(agent,transitions,num_updates=0 ):
                
        n_batches = transitions['observations'].shape[0]//250

        indexes = jnp.arange(transitions['observations'].shape[0])
        indexes = jax.random.permutation(agent.rng, indexes)
        batch_size = indexes.shape[0] // n_batches
        idxs = indexes[:batch_size * n_batches].reshape((n_batches, batch_size))

        ### Make sure we're doing at least 100 updates per epoch
        ### This is to maintain some fairness between the off-policy and on-policy critics

        #if n_batches <100 or num_updates is not None:

        if n_batches < 100:
            #n_batches = jnp.maximum(100,num_updates)
            n_batches = 100
            idxs = jax.random.choice(agent.rng, a=transitions['observations'].shape[0], shape=(n_batches, 250), replace=True)

        batches = jax.vmap(lambda i: jax.tree.map(lambda x: x[i], transitions))(idxs)
        agent,batches = jax.lax.fori_loop(0,n_batches,body,(agent,batches))
        
        return agent
    


    
    @jax.jit
    def update_actor(agent, batch: Batch):
        
   
  
        def actor_loss_fn(
                actor_params,
                adv,
                batch,
        ):
            
            ### Compute probability of old actions under new policy
            

            
            
            discounts,masks,logp = batch["discounts"],batch["masks"],batch["log_probs"]
                    
            dist = agent.actor.apply_fn({'params': actor_params},batch["observations"])
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

            actor_loss = -jnp.minimum(actor_loss1,actor_loss2).mean()
                
            ### Pad Q and logits because actor buffer is padded ###
            logp = masks * new_logp
            
            entropy = -1 * (masks*logp).sum()/(masks.sum())
            
            return actor_loss, {
                'actor_loss': actor_loss,
                'entropy': entropy,
                'approx_kl':approx_kl
            }
            
        
        def temp_loss_fn(temp_params, entropy, target_entropy):
            temperature = agent.temp.apply_fn({'params': temp_params})
            temp_loss = (temperature * (entropy - target_entropy)).mean()

            ### Clip temperature to minimum value
            temp_loss = jax.lax.cond(
                        jnp.logical_and(temperature < 0.001, temp_loss > 0),
                        lambda _: 0.0,
                        lambda _: temp_loss,
                        operand=None
                        )
            
            return temp_loss, {
                'temp_loss': temp_loss,
                'temperature': temperature,
            }
            

        new_rng, curr_key, next_key = jax.random.split(agent.rng, 3)

        observations,next_observations = batch["observations"],batch["next_observations"]
        
        observations = jnp.concatenate([observations, next_observations[-1][None]], axis=0)
        
        def evaluate(observations,key):
            
            actions, log_p,_ = agent.sample_actions(observations,seed=key)
            q_all = agent.critic.apply_fn({'params': agent.critic.params},observations,actions)  
            v = jnp.mean(q_all,axis=0)
            
            return v,log_p
        
        ### Compute advantage for the fixed states AND actions
        vs,hs = jax.vmap(evaluate,in_axes=(None,0))(batch["observations"],jax.random.split(curr_key,10))        
        tmp_v,tmp_logp = jnp.mean(vs,axis=0),jnp.mean(hs,axis=0)
        q = agent.critic.apply_fn({'params': agent.critic.params},batch["observations"],batch["actions"]).mean(axis=0)
        adv = (q-agent.temp.apply_fn({'params': agent.temp.params})*batch["log_probs"]) - (tmp_v - agent.temp.apply_fn({'params': agent.temp.params}) *tmp_logp)### This one worked
        adv = adv.reshape(-1)

        grads,actor_info = jax.grad(actor_loss_fn,has_aux=True)(agent.actor.params,adv,batch)
        new_actor = agent.actor.apply_gradients(grads=grads)
        
        grads,temp_info = jax.grad(temp_loss_fn,has_aux=True)(agent.temp.params,actor_info['entropy'],agent.config['target_entropy'])
        new_temp = agent.temp.apply_gradients(grads=grads)
        agent = agent.replace(rng=new_rng, actor=new_actor,temp=new_temp)

        info = {**actor_info, **temp_info}  
     
            
        return agent,info
                    

    @jax.jit
    def sample_actions(agent,   
                       observations: np.ndarray,
                       seed: PRNGKey,
                       temperature: float = 1.0,
                       params=None,
                       ) -> jnp.ndarray:
        
        if params is None:
            params = agent.actor.params
        ### random always true
        dist = agent.actor.apply_fn({'params': params},observations, temperature=temperature)
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
        dist = agent.actor.apply_fn({'params': agent.actor.params},observations, temperature=0.)
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
                min_target,
                entropy_coeff,
                momentum,
                b2,
                actor_lr,
                critic_lr,
                temp_lr,
                temperature,
                num_actor_updates,
                clipping_ratio,
                actor_hidden_dims: Sequence[int],
                critic_hidden_dims: Sequence[int],
                activation_fn: str,
                gae_lambda : float,
                use_layer_norm : bool,
                minibatch : bool = False,
                target_entropy: float = None,
                state_dependent_std=True,
                tanh_squash_distribution=False,## This should be false
                tanh_squash_actions=True, ## This should be true
                store_grads = False,
                use_bias = True,
                
           
                
                
            **kwargs):

        print('Extra kwargs:', kwargs)

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key = jax.random.split(rng, 3)

        activations = nn.relu if activation_fn == 'relu' else nn.tanh
        #final_fc_init_scale = 1. if activation_fn == 'relu' else 1e-2
        final_fc_init_scale = 1e-2

        action_dim = actions.shape[-1]
        actor_def = Policy(actor_hidden_dims, action_dim=action_dim,activations=activations,final_fc_init_scale=final_fc_init_scale,
            state_dependent_std=state_dependent_std, tanh_squash_distribution=tanh_squash_distribution,use_layer_norm=use_layer_norm,use_bias=use_bias)

        critic_def = ensemblize(OriginalCritic,num_critics)(hidden_dims=critic_hidden_dims,use_layer_norm=use_layer_norm,activations=activations)
        critic_params = critic_def.init(critic_key, observations, actions)['params']
        critic = TrainState.create(apply_fn=critic_def.apply, params=critic_params, tx=optax.adam(learning_rate=critic_lr))
     
        actor_params = actor_def.init(actor_key, observations)['params']
        temp_def = Temperature(temperature)
        temp_params = temp_def.init(rng)['params']
        
        tx = optax.chain(
            optax.clip_by_global_norm(0.5), ## This is necessary to avoid exploding gradients due to numerical instabilities.
            optax.adam(learning_rate=actor_lr,b1=momentum,b2=b2),
        )
        actor = TrainState.create(apply_fn=actor_def.apply, params=actor_params, tx=tx)
        temp = TrainState.create(apply_fn=temp_def.apply, params=temp_params, tx=optax.adam(learning_rate=temp_lr,b1=momentum,b2=b2)) ##placeholder
            
        if target_entropy is None:

            target_entropy = -entropy_coeff*action_dim

        config = flax.core.FrozenDict(dict(
            discount=discount,
            target_entropy=target_entropy,
            observations=observations,
            actions=actions,  
            num_critics = num_critics, 
            num_actor_updates = num_actor_updates,  
            clipping_ratio = clipping_ratio,
            min_target = min_target,
            tanh_squash_actions=tanh_squash_actions,
            gae_lambda=gae_lambda,
            minibatch=minibatch,
            store_grads=store_grads,
      
        ))

        return SACAgent(rng, critic=critic, target_critic=critic, actor=actor, temp=temp, config=config)



