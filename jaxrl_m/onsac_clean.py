
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
import chex

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
                                  (), jnp.log(self.initial_temperature)))
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
                            target_q = jnp.min(target_q,axis=0) 
                            target_q = jnp.repeat(target_q.reshape(1,-1),2,axis=0) ## make sure to keep same shape
                           
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
    
    #@jax.jit
    def update_critics_seq(agent,batches,R2):
        
     
        
        size = batches["observations"].shape[0]
        agent,batches = jax.lax.fori_loop(0,size,body,(agent,batches))
        
        return agent

    
    @jax.jit
    def update_actor(agent, batch: Batch,R2):
        
        

                
        def compute_gae(
            rewards,
            values,
            next_values,
            standardize_advantages: bool = False,
        ) :
           

        
                
            lambda_ = agent.config["gae_lambda"]
            r_t = rewards
            truncation_mask = batch["masks"][:-1]
            discount_t =  jnp.ones_like(r_t) * 0.995 
            
            
            
            r_t, discount_t, values,next_values, truncation_mask = jax.tree_util.tree_map(
            lambda x: jnp.expand_dims(x,axis=0), (r_t, discount_t, values,next_values, truncation_mask)
            )
            batch_size = r_t.shape[0]
            r_t, discount_t, values,next_values, truncation_mask = jax.tree_util.tree_map(
            lambda x: jnp.swapaxes(x, 0, 1), (r_t, discount_t, values,next_values, truncation_mask)
            )

            # Swap axes to make time axis the first dimension
         

            chex.assert_type([r_t, values, discount_t, truncation_mask], float)

            lambda_ = jnp.ones_like(discount_t) * lambda_  # If scalar, make into vector.

            delta_t = r_t + discount_t * next_values - values
            delta_t *= truncation_mask

            # Iterate backwards to calculate advantages.
            def _body(
                acc: chex.Array, xs: Tuple[chex.Array, chex.Array, chex.Array, chex.Array]
            ) -> Tuple[chex.Array, chex.Array]:
                deltas, discounts, lambda_, trunc_mask = xs
                acc = deltas + discounts * lambda_ * trunc_mask * acc
                return acc, acc

            #jax.debug.print(f'{delta_t.shape}, {discount_t.shape}, {lambda_.shape}, {truncation_mask.shape},')
            _, advantage_t = jax.lax.scan(
                _body,
                jnp.zeros(batch_size),
                (delta_t, discount_t, lambda_, truncation_mask),
                reverse=True,
                unroll=16,
            )

            advantage_t *= truncation_mask

            advantage_t = jax.lax.stop_gradient(advantage_t)

            if standardize_advantages:
                advantage_t = jax.nn.standardize(advantage_t)

            return advantage_t

       
      
        def actor_loss_fn(
                actor_params,
                adv,
                batch,
                idx,
        ):
            
            ### Compute probability of old actions under new policy
            
            batch = jax.tree_map(lambda x:x[idx],batch)
            adv = adv[idx]
            
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
            
        def temp_loss_fn(temp_params, entropy, target_entropy):
            temperature = agent.temp(params=temp_params)
            temp_loss = (temperature * (entropy - target_entropy)).mean()
            return temp_loss, {
                'temp_loss': temp_loss,
                'temperature': temperature,
            }
            
     
        
        new_rng, curr_key, next_key = jax.random.split(agent.rng, 3)


        observations,next_observations = batch["observations"],batch["next_observations"]
        
 
        def evaluate(observations,key):
            
            actions, log_p,_ = agent.sample_actions(observations,seed=key)
            q_all = agent.critic(observations,actions)
            v = jnp.mean(q_all,axis=0)
            
            return v,-log_p
        
        vs,hs = jax.vmap(evaluate,in_axes=(None,0))(observations,jax.random.split(curr_key,10))
        
        tmp_v,tmp_h = jnp.mean(vs,axis=0),jnp.mean(hs,axis=0)
        #tmp_v,tmp_h = agent.critic(batch["observations"],batch["actions"]).mean(axis=0),-batch["log_probs"]
        
        # v,h = tmp_v[:-1],tmp_h[:-1]
        # next_v,next_h = tmp_v[1:],tmp_h[1:]
        
        # v = v + agent.temp()*h
        # next_v = next_v + agent.temp()*next_h
        # rewards = batch["rewards"][:-1]-agent.temp()*batch["log_probs"][:-1]
        # adv = compute_gae(rewards,v,next_v).squeeze()
        
        

        ### Compute advantage for the fixed states AND actions
        
        
        q_all = agent.critic(batch["observations"],batch["actions"])
        q = jnp.mean(q_all,axis=0)
        adv = q-tmp_v + agent.temp()*(-batch["log_probs"]-tmp_h)### This one worked
        
        

     
   
        
        for i in range(agent.config["num_actor_updates"]):
     
            
            new_rng,_ = jax.random.split(new_rng)
            idx = jax.random.choice(new_rng, adv.shape[0], shape=(256,), replace=True)
            new_actor, actor_info = agent.actor.apply_loss_fn(actor_loss_fn,True,adv,batch,idx)#adv
            new_temp, temp_info = agent.temp.apply_loss_fn(temp_loss_fn,True,actor_info['entropy'],agent.config['target_entropy'])

            agent = agent.replace(rng=new_rng, actor=new_actor,temp=new_temp)
            
        return agent, {**actor_info,**temp_info}



    @jax.jit
    def update_actor_sac(agent, batch: Batch,R2):
        new_rng, curr_key, next_key = jax.random.split(agent.rng, 3)

        def actor_loss_fn(actor_params):
            
            
            actions,log_ps,_ = agent.sample_actions(batch["observations"],params=actor_params,seed=new_rng)

            q = agent.critic(batch["observations"],actions).mean(axis=0)

            q = batch["discounts"]*q
            
            actor_loss = (agent.temp()*log_ps-q).mean()
            
            return actor_loss,{}

        grads,info = jax.grad(actor_loss_fn,has_aux=True)(agent.actor.params)
        new_actor,_ = agent.actor.apply_loss_fn(loss_fn=actor_loss_fn,has_aux=True)
        actor_info = {"grads":grads}
        return agent.replace(rng=new_rng,actor=new_actor), {**actor_info}
            
        
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
                actor_lr,
                critic_lr,
                temperature,
                num_actor_updates,
                clipping_ratio,
                actor_hidden_dims: Sequence[int],
                critic_hidden_dims: Sequence[int],
                gae_lambda : float,
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
        temp_def = Temperature(temperature)
        temp_params = temp_def.init(rng)['params']
        
        tx = optax.chain(
            optax.clip_by_global_norm(0.5), ## This is necessary to avoid exploding gradients due to numerical instabilities.
            optax.adam(learning_rate=actor_lr,b1=momentum),
        )
        temp = TrainState.create(temp_def, temp_params, tx=optax.adam(learning_rate=3e-4,b1=momentum)) ##placeholder
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
            gae_lambda=gae_lambda,
            
        ))

        return SACAgent(rng, critic=critic, target_critic=critic, actor=actor, temp=temp, config=config)

