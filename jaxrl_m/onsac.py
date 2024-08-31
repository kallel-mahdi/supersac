
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax

from jaxrl_m.common import TrainState, nonpytree_field
from jaxrl_m.networks import OriginalCritic, Policy
from jaxrl_m.typing import *
import jax.lax as lax

def get_batch(i,batches):
    return  jax.tree_map(lambda x: x[i], batches)

def body(i,val):
    agent,batches = val
    return (agent.update_critics(get_batch(i,batches)),batches)

class Temperature(nn.Module):
    #initial_temperature: float = -4.605 ## (log(0.01))
    initial_temperature: float = 0.01
    
    
    @nn.compact
    def __call__(self) -> jnp.ndarray:
        log_temp = self.param('log_temp',
                              init_fn=lambda key: jnp.full(
                                  (), self.initial_temperature))
        #return jnp.exp(log_temp)
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
                        
                        next_actions,next_log_probs,_ = agent.sample_actions(batch["next_observations"],seed=next_key)
                        
                        next_q  = agent.critic(batch['next_observations'], next_actions,params=critic_params)
                        
                        target_q = batch['rewards'] + agent.config['discount'] * batch['masks'] * next_q
                        ### Add entropy
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
       
    
        # ### Reset optimizers 
        # new_critic_params = agent.critic.params
        # new_opt_state = jax.vmap(agent.critic.tx.init)(new_critic_params)
        # new_critics = agent.critic.replace(params=new_critic_params,opt_state=new_opt_state)
        # agent = agent.replace(critic=new_critics)
        ### Train critic sequentially
        agent,batches = jax.lax.fori_loop(0,2500,body,(agent,batches))
        
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
            p_log_probs = dist.log_prob(pre_actions)
            new_logp = p_log_probs - jnp.sum(2 * (jnp.log(2) - pre_actions - jax.nn.softplus(-2 * pre_actions)), axis=-1)
            
            
            logratio = new_logp - logp
            ratio = jnp.exp(logratio)

            # Calculate how much policy is changing
            approx_kl = ((ratio - 1) - logratio).mean()

            # Policy loss
            clip_coef = agent.config["clipping_ratio"] ##default 0.2 
            actor_loss1 = masks*adv * ratio
            actor_loss2 = masks*adv * jnp.clip(ratio, 1 - clip_coef, 1 + clip_coef)

            if agent.config['discount_entropy']:
                actor_loss = -jnp.minimum(discounts*actor_loss1,discounts*actor_loss2).sum()/(discounts.sum())
                #actor_loss = -(discounts*ratio*adv).sum()/(discounts.sum())
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

        R2 = R2.reshape(-1,1)
        R2 = jax.nn.softmax(R2,axis=0)

        observations = batch["observations"]
        
        
        j = 10
        qs,logps = jnp.zeros((2500,)),jnp.zeros((2500,))
        
        call_one_critic = lambda observations,actions,params: agent.critic(observations,actions,params=params)
        call_many_critics = lambda observations,actions : jax.vmap(call_one_critic,in_axes=(None,None,0))(observations, actions,agent.critic.params)
        
        ### Compute value for the fixed states
        
        for i in range(j):
            
            curr_key,_ = jax.random.split(curr_key)
            actions, log_p,_ = agent.sample_actions(observations,seed=curr_key)
            q_all = call_many_critics(observations,actions)
            q = jnp.sum(R2*q_all,axis=0)
            qs+=q
            logps+=log_p
                    
        v = qs/j
        h = -(logps/j)
        
        ### Compute advantage for the fixed states AND actions
        q_all = call_many_critics(batch["observations"],batch["actions"])
        q = jnp.sum(R2*q_all,axis=0)
        
        
        #adv = q-v - agent.temp()*batch["log_probs"]### This one worked
        adv = q-v  - agent.temp()*(batch["log_probs"]-(logps/j))### This one worked
        
        for i in range(agent.config["num_actor_updates"]):
            
            new_actor, actor_info = agent.actor.apply_loss_fn(actor_loss_fn,True,adv)
            new_temp, temp_info = agent.temp.apply_loss_fn(temp_loss_fn,True,actor_info['entropy'], agent.config['target_entropy'])
            agent = agent.replace(rng=new_rng, actor=new_actor,temp=new_temp)
            new_temp.params["log_temp"]=jnp.clip(new_temp.params["log_temp"],0.01,1)
            #agent = agent.replace(temp=new_temp)
            
        
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
        actions = jax.nn.tanh(pre_actions)
        log_ps = pre_log_ps - jnp.sum(2 * (jnp.log(2) - pre_actions - jax.nn.softplus(-2 * pre_actions)), axis=-1)        
        
        return actions,log_ps,pre_actions


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
                momentum,
                actor_lr,
                critic_lr,
                temp_lr,
                num_actor_updates,
                clipping_ratio,
                hidden_dims: Sequence[int] = (256, 256),
                target_entropy: float = None,
            **kwargs):

        print('Extra kwargs:', kwargs)

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key = jax.random.split(rng, 3)

        action_dim = actions.shape[-1]
        actor_def = Policy(hidden_dims, action_dim=action_dim,
            state_dependent_std=True, tanh_squash_distribution=False)

        critic_def = OriginalCritic(hidden_dims)
        critic_keys  = jax.random.split(critic_key, num_critics)
        critic_params = jax.vmap(critic_def.init,in_axes=(0,None,None))(critic_keys, observations, actions)['params']
        critics = jax.vmap(TrainState.create,in_axes=(None,0,None))(critic_def,critic_params,optax.adam(learning_rate=critic_lr))

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
            clipping_ratio = clipping_ratio
            
        ))

        return SACAgent(rng, critic=critics, target_critic=critics, actor=actor, temp=temp, config=config)

