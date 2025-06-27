
import jax.random
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax

from jaxrl_m.common import TrainState, nonpytree_field
from jaxrl_m.networks import OriginalCritic,OriginalV, Policy,ensemblize
from jaxrl_m.typing import *
import jax.lax as lax
import chex

def get_batch(i,batches):
    return  jax.tree.map(lambda x: x[i], batches)

def body(i,val):
    agent,batches = val
    return (agent.update_critics(get_batch(i,batches)),batches)

def body2(i,val):
    agent,batches = val
    return (agent.update_vs(get_batch(i,batches)),batches)


class Temperature(nn.Module):
    initial_temperature: float = 1.
    min_temperature: float = 0.01

    @nn.compact
    def __call__(self) -> jnp.ndarray:
        log_temp = self.param('log_temp',
                                init_fn=lambda key: jnp.full(
                                    (), jnp.log(self.initial_temperature)))
        return jnp.exp(log_temp)

class SACAgent(flax.struct.PyTreeNode):
    rng: PRNGKey
    critic: TrainState
    target_critic: TrainState
    actor: TrainState
    temp: TrainState
    config: dict = nonpytree_field()
    
    
    def update_vs(agent,batch: Batch):
    
        new_rng, curr_key, next_key = jax.random.split(agent.rng, 3)

        def update_one_v(target_critic):
                            
                def critic_loss_fn(critic_params):
                        
                        next_actions,next_log_probs,_ = agent.sample_actions(batch["next_observations"],seed=next_key)
                        
                        next_v  = agent.target_critic(batch['next_observations'],params=critic_params)
                        
                        target_v = batch['rewards'] + agent.config['discount'] * batch['masks'] * next_v
                        ### Add entropy
                        target_v = target_v - agent.config['discount'] * batch['masks'] * next_log_probs * agent.temp()
                        target_v = jax.lax.stop_gradient(target_v)
                        
                        if agent.config['min_target']:
                            target_v = jnp.min(target_v,axis=0) 
                            target_v = jnp.repeat(target_v.reshape(1,-1),2,axis=0) ## make sure to keep same shape
                            
                        v = agent.target_critic(batch['observations'],params=critic_params)
                        critic_loss = ((v-target_v)**2).mean()
                        
                        return critic_loss, {
                        'critic_loss': critic_loss,
                        'q1': v.mean(),
                    }  
                
                new_critic, critic_info = target_critic.apply_loss_fn(loss_fn=critic_loss_fn, has_aux=True)
                
                return new_critic,critic_info


        new_critics,critic_info = update_one_v(agent.target_critic)
        agent = agent.replace(rng=new_rng,target_critic=new_critics)
        
        return agent

    @jax.jit
    def update_critics_seq(agent,transitions,n_batches=128):

        # n_batches = transitions['observations'].shape[0]//256

        # indexes = jnp.arange(transitions['observations'].shape[0])
        # indexes = jax.random.permutation(agent.rng, indexes)
        # batch_size = indexes.shape[0] // n_batches
        # idxs = indexes[:batch_size * n_batches].reshape((n_batches, batch_size))

        ### Or just sample  batches randomly
        n_batches = transitions['observations'].shape[0]
        n_batches = 100
        idxs = jax.random.choice(agent.rng,a=transitions['observations'].shape[0], shape=(n_batches,256), replace=True)

        batches = jax.vmap(lambda i: jax.tree.map(lambda x: x[i], transitions))(idxs)
        agent,batches = jax.lax.fori_loop(0,n_batches,body2,(agent,batches))
        
        return agent

    
    @jax.jit
    def update_actor(agent, batch: Batch):
        
      
        def compute_gae(rewards: jnp.ndarray, values: jnp.ndarray, next_values: jnp.ndarray, 
                        dones: jnp.ndarray, truncations: jnp.ndarray, gamma: float, 
                        lam: float) -> jnp.ndarray:
            # Compute deltas (TD residuals)
            
            
            deltas = rewards + gamma * next_values * (1 - dones) - values
            deltas = deltas * (1 - truncations)
            
            # Define a function to update the cumulative advantage in a scan step
            def update_advantage(cumulative_advantage, delta_and_mask):
                delta, trunc,done = delta_and_mask
                # Update advantage using GAE with truncation handling
                cumulative_advantage = delta + gamma * lam * (1-done) * (1 - trunc) * cumulative_advantage 
                return cumulative_advantage, cumulative_advantage

            # Use lax.scan to accumulate the advantages in reverse order
            # Reverse deltas and truncations for the scan (scan works forward, but we want to process backwards)
            reversed_deltas = deltas[::-1]
            reversed_truncations = truncations[::-1]
            reversed_dones = dones[::-1]
            
            # Scan will return the final cumulative advantage and the full sequence of advantages
            _, advantages = lax.scan(update_advantage, 0.0, (reversed_deltas, reversed_truncations,reversed_dones))

            # Reverse the advantages back to the original order
            advantages = advantages[::-1]
            advantages = advantages* (1 - truncations)

            return advantages,None
                
  
        def actor_loss_fn(
                actor_params,
                adv,
                batch,
                idx,
        ):
            
            ### Compute probability of old actions under new policy
            

            batch = jax.tree.map(lambda x:x[idx],batch)
            adv = adv[idx]
            
            discounts,masks,logp = batch["discounts"],batch["masks"],batch["log_probs"]
            
            #jax.debug.print("🤯 HELLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL{x} 🤯", x=discounts[:100])
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
                #jax.debug.print("🤯 HELLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL{x} 🤯", x=discounts[:100])
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
            
        
        def temp_loss_fn(temp_params, entropy, target_entropy):
            temperature = agent.temp(params=temp_params)
            temp_loss = (temperature * (entropy - target_entropy)).mean()
            temp_loss = jax.lax.cond(
                jnp.logical_and(temperature < 0.01, temp_loss > 0),
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
            q_all = agent.target_critic(observations)
            v = jnp.mean(q_all,axis=0)
            
            return v,log_p
        
        if agent.config['gae_lambda'] > 0.:
        
            vs,hs = jax.vmap(evaluate,in_axes=(None,0))(observations,jax.random.split(curr_key,10))
            
            tmp_v,tmp_logp = jnp.mean(vs,axis=0),jnp.mean(hs,axis=0)
            tmp_v -= agent.temp()*tmp_logp
            v,next_v= tmp_v[:-1],tmp_v[1:]
            
            rewards = batch["rewards"]-agent.temp()*batch["log_probs"]
            dones = jnp.bool(1-batch["masks"])
            truncations = jnp.bool(batch["truncateds"])
            
            adv,_ = compute_gae(rewards.squeeze(),v.squeeze(),next_v.squeeze(),dones.squeeze(),truncations.squeeze(),
                                gamma=agent.config['discount'],lam=agent.config['gae_lambda'])
            adv = adv.reshape(-1)

        else :   

            raise ValueError("When using value function lambda > 0, GAE must be used.")

            
        if agent.config["minibatch"]:
        
            indexes = jnp.arange(adv.shape[0])
            indexes = jax.random.permutation(new_rng, indexes)
            batch_size = 256
            num_actor_updates = adv.shape[0] // batch_size
            index_batches = jnp.split(indexes[:batch_size * num_actor_updates], num_actor_updates)
        
        else : index_batches = [jnp.arange(adv.shape[0]) for i in range(agent.config["num_actor_updates"])]
            
      
        for idx in index_batches:
            
            if not agent.config['minibatch']: idx = jnp.arange(adv.shape[0])
            
            new_actor, actor_info = agent.actor.apply_loss_fn(actor_loss_fn,True,adv,batch,idx)#adv
            new_temp, temp_info = agent.temp.apply_loss_fn(temp_loss_fn,True,actor_info['entropy'],agent.config['target_entropy'])
            

            agent = agent.replace(rng=new_rng, actor=new_actor,temp=new_temp)
            
        return agent, {**actor_info,**temp_info}

            
        
    @jax.jit
    def sample_actions(agent,   
                       observations: np.ndarray,
                       seed: PRNGKey,
                       temperature: float = 1.0,
                       params=None,
                       ) -> jnp.ndarray:
        
        ### random always true
        dist = agent.actor(observations,params=params, temperature=temperature)
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
                tanh_squash_distribution=False,
                tanh_squash_actions=True,
                use_bias=True,
                
                
                
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
        #critic_params = critic_def.init(critic_key, observations, actions)['params']
        critic_params = critic_def.init(critic_key, observations, actions)['params']
        critic = TrainState.create(critic_def, critic_params, tx=optax.adam(learning_rate=critic_lr))
          
        v_def = ensemblize(OriginalV,num_critics)(hidden_dims=critic_hidden_dims,use_layer_norm=use_layer_norm,activations=activations)
        v_params = v_def.init(critic_key, observations, actions)['params']
        v = TrainState.create(v_def, v_params, tx=optax.adam(learning_rate=critic_lr))

        actor_params = actor_def.init(actor_key, observations)['params']
        temp_def = Temperature(temperature)
        temp_params = temp_def.init(rng)['params']
        
        tx = optax.chain(
            optax.clip_by_global_norm(0.5), ## This is necessary to avoid exploding gradients due to numerical instabilities.
            optax.adam(learning_rate=actor_lr,b1=momentum,b2=b2),
        )
        actor = TrainState.create(actor_def, actor_params, tx=tx)
        #temp = TrainState.create(temp_def, temp_params, tx=optax.adam(learning_rate=temp_lr,b1=momentum,b2=b2)) ##placeholder
        temp = TrainState.create(temp_def, temp_params, tx=optax.adam(learning_rate=temp_lr,b1=momentum,b2=b2)) ##placeholder
            
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
            gae_lambda=gae_lambda,
            minibatch=minibatch,
            
        ))

        return SACAgent(rng, critic=critic, target_critic=v, actor=actor, temp=temp, config=config)

