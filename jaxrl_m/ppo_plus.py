
import jax.random
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from dataclasses import dataclass, field
from typing import Tuple, Optional, Union

from flax.training.train_state import TrainState
from jaxrl_m.common import nonpytree_field
from jaxrl_m.networks import OriginalCritic,OriginalV, Policy,ensemblize
from jaxrl_m.typing import *
import jax.lax as lax
from functools import partial

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
                        
                        next_q  = agent.critic.apply_fn({'params': critic_params}, batch['next_observations'], next_actions)
                        
                        target_q = batch['rewards'] + agent.config.training.discount * batch['masks'] * next_q
                        ### Add entropy
                        target_q = target_q - agent.config.training.discount * batch['masks'] * next_log_probs * agent.temp.apply_fn({'params': agent.temp.params})
                        target_q = jax.lax.stop_gradient(target_q)
                        
                        if agent.config.training.min_target:
                            target_q = jnp.min(target_q,axis=0) 
                            target_q = jnp.repeat(target_q.reshape(1,-1),2,axis=0) ## make sure to keep same shape
                           
                        q = agent.critic.apply_fn({'params': critic_params}, batch['observations'], batch['actions'])
                        critic_loss = ((q-target_q)**2).mean()
                        
                        return critic_loss, {
                        'critic_loss': critic_loss,
                        'q1': q.mean(),
                    }
                
                grads, critic_info = jax.grad(critic_loss_fn, has_aux=True)(critic.params)
                new_critic = critic.apply_gradients(grads=grads)
                
                return new_critic,critic_info


        new_critics,critic_info = update_one_critic(agent.critic)
        agent = agent.replace(rng=new_rng,critic=new_critics)
        
        return agent


    
    @partial(jax.jit,static_argnames=('n_updates',))
    def update_critics_seq(agent,transitions,n_updates=None ):
                
        batch_size = 250
        n_batches = transitions['observations'].shape[0]//batch_size
        if n_batches < 100: n_updates = 100
        
        if n_updates:
            
            ### Batches will be sampled randomly with replacement
            idxs = jax.random.choice(agent.rng, a=transitions['observations'].shape[0], shape=(n_updates, 250), replace=True)
            
        else:
            ### Go through all the transitions without replacement
            indexes = jnp.arange(transitions['observations'].shape[0])
            indexes = jax.random.permutation(agent.rng, indexes)
            idxs = indexes[:batch_size * n_batches].reshape((n_batches, batch_size))

        batches = jax.vmap(lambda i: jax.tree.map(lambda x: x[i], transitions))(idxs)
        agent,batches = jax.lax.fori_loop(0,n_batches,body,(agent,batches))
        
        return agent
    

    
    @jax.jit
    def update_actor_seq(agent, batch: Batch):
        
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
            
            masks,logp = batch["masks"],batch["log_probs"]
            dist = agent.actor.apply_fn({'params': actor_params}, batch["observations"])
            pre_actions = batch["pre_actions"]
            pre_log_probs = dist.log_prob(pre_actions)
            
            if agent.config.training.tanh_squash_actions:
                new_logp = pre_log_probs - jnp.sum(2 * (jnp.log(2) - pre_actions - jax.nn.softplus(-2 * pre_actions)), axis=-1)
            
            else : 
                new_logp = pre_log_probs
            
            
            logratio = new_logp - logp
            ratio = jnp.exp(logratio)

            # Calculate how much policy is changing
            approx_kl = ((ratio - 1) - logratio).mean()

            # Policy loss
            clip_coef = agent.config.ppo.clipping_ratio ##default 0.2 
            
            
            if agent.config.ppo.spo_loss:
                
                spo_term1 = masks * adv * ratio
                spo_term2 = masks * jnp.abs(adv) / (2 * clip_coef) * jnp.square(ratio - 1)
                actor_loss = -(spo_term1 - spo_term2).mean()
            
            
            else : 

                actor_loss1 = masks*adv * ratio
                actor_loss2 = masks*adv * jnp.clip(ratio, 1 - clip_coef, 1 + clip_coef)
                actor_loss = -jnp.minimum(actor_loss1,actor_loss2).mean()
                    
            ### Pad Q and logits because actor buffer is padded ###
            logp = masks * new_logp
            
            # Use simple entropy calculation (removed discount_entropy logic)
            entropy = -1 * (masks*logp).sum()/(masks.sum())
            
            return actor_loss, {
                'actor_loss': actor_loss,
                'entropy': entropy,
                'approx_kl':approx_kl
            }
            
        
        def temp_loss_fn(temp_params, entropy, target_entropy):
            temperature = agent.temp.apply_fn({'params': temp_params})
            temp_loss = (temperature * (entropy - target_entropy)).mean()

            ### Clip temperature to minimum value to avoid stability issues
            temp_loss = jax.lax.cond(
                        jnp.logical_and(temperature <= 0.001, temp_loss > 0),
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
            q_all = agent.critic.apply_fn({'params': agent.critic.params}, observations, actions)
            v = jnp.mean(q_all,axis=0)
            
            return v,log_p
        
        if agent.config.ppo.gae_lambda > 0.:
        
            vs,hs = jax.vmap(evaluate,in_axes=(None,0))(observations,jax.random.split(curr_key,10))
            
            tmp_v,tmp_logp = jnp.mean(vs,axis=0),jnp.mean(hs,axis=0)
            tmp_v -= agent.temp.apply_fn({'params': agent.temp.params})*tmp_logp
            v,next_v= tmp_v[:-1],tmp_v[1:]
            
            rewards = batch["rewards"]-agent.temp.apply_fn({'params': agent.temp.params})*batch["log_probs"]
            dones = jnp.bool(1-batch["masks"])
            truncations = jnp.bool(batch["truncateds"])
            
            adv,_ = compute_gae(rewards.squeeze(),v.squeeze(),next_v.squeeze(),dones.squeeze(),truncations.squeeze(),
                                gamma=agent.config.training.discount,lam=agent.config.ppo.gae_lambda)
            adv = adv.reshape(-1)

        else :   

            ### Compute advantage for the fixed states AND actions
            vs,hs = jax.vmap(evaluate,in_axes=(None,0))(batch["observations"],jax.random.split(curr_key,10))        
            tmp_v,tmp_logp = jnp.mean(vs,axis=0),jnp.mean(hs,axis=0)
            q = agent.critic.apply_fn({'params': agent.critic.params}, batch["observations"], batch["actions"]).mean(axis=0)
            adv = (q-agent.temp.apply_fn({'params': agent.temp.params})*batch["log_probs"]) - (tmp_v - agent.temp.apply_fn({'params': agent.temp.params}) *tmp_logp)### This one worked
            adv = adv.reshape(-1)
            
        
        if agent.config.ppo.store_grads:
            idx = jnp.arange(adv.shape[0])
            grads,info = jax.grad(actor_loss_fn,has_aux=True)(agent.actor.params,adv,batch,idx)
    
        indexes = jnp.arange(adv.shape[0])
        indexes = jax.random.permutation(new_rng, indexes)
        batch_size = 250
        num_actor_updates = adv.shape[0] // batch_size
        index_batches = jnp.split(indexes[:batch_size * num_actor_updates], num_actor_updates)
        
        for idx in index_batches:
            
            # Always use minibatch (removed minibatch config check)
            
            actor_grads, actor_info = jax.grad(actor_loss_fn, has_aux=True)(agent.actor.params, adv, batch, idx)
            new_actor = agent.actor.apply_gradients(grads=actor_grads)
            
            temp_grads, temp_info = jax.grad(temp_loss_fn, has_aux=True)(agent.temp.params, actor_info['entropy'], agent.config.ppo.target_entropy)
            new_temp = agent.temp.apply_gradients(grads=temp_grads)
            

            agent = agent.replace(rng=new_rng, actor=new_actor,temp=new_temp)

        info = {**actor_info, **temp_info}  
        if agent.config.ppo.store_grads:
                        info['grads'] = grads
            
        return agent,info
                    

    
    #@jax.jit
    def update_actor_old(agent, batch: Batch):
        
      
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
        ):
            """Compute actor loss for PPO.
            
            Args:
                actor_params: Parameters of the actor network
                adv: Advantage estimates
                batch: Batch of transitions
            
            Returns:
                Tuple of (actor_loss, info_dict)
            """
            discounts, masks, logp = batch["discounts"], batch["masks"], batch["log_probs"]
            
            #jax.debug.print('masks: {}', masks.sum()))
            
            # Compute probability of actions under new policy
            dist = agent.actor.apply_fn({'params': actor_params}, batch["observations"])
            pre_actions = batch["pre_actions"]
            pre_log_probs = dist.log_prob(pre_actions)
            
            # Apply tanh squashing correction if needed
            if agent.config.training.tanh_squash_actions:
                new_logp = pre_log_probs - jnp.sum(2 * (jnp.log(2) + pre_actions - jax.nn.softplus(2 * pre_actions)), axis=-1)
            else:
                new_logp = pre_log_probs
            
            # Calculate importance sampling ratio
            logratio = new_logp - logp
            ratio = jnp.exp(logratio)

            # Calculate approximate KL divergence for monitoring
            approx_kl = ((ratio - 1) - logratio).mean()

            # PPO clipped objective
            clip_coef = agent.config.ppo.clipping_ratio
            
            actor_loss1 = masks * adv * ratio
            actor_loss2 = masks * adv * jnp.clip(ratio, 1 - clip_coef, 1 + clip_coef)

            # Apply discounting if configured
            actor_loss = -jnp.minimum(actor_loss1, actor_loss2).mean()
                
            # Calculate entropy
            logp = masks * new_logp
            
            entropy = -1 * (masks * logp).sum() / (masks.sum())
            
            return actor_loss, {
                'actor_loss': actor_loss,
                'entropy': entropy,
                'approx_kl': approx_kl,
                'max_ratio':jnp.max(ratio),
                'min_ratio':jnp.min(ratio),
                'mean_ratio':jnp.mean(ratio),
                'std_ratio':jnp.std(ratio),
                
            }
            
        
        def temp_loss_fn(temp_params, entropy, target_entropy):
            """Compute temperature loss for entropy regularization.
            
            Args:
                temp_params: Temperature parameters
                entropy: Current policy entropy
                target_entropy: Target entropy value
                
            Returns:
                Tuple of (temp_loss, info_dict)
            """
            temperature = agent.temp.apply_fn({'params': temp_params})
            temp_loss = (temperature * (entropy - target_entropy)).mean()

            # Prevent temperature from going too low
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

        observations, next_observations = batch["observations"], batch["next_observations"]
        
        # Include the last next_observation to compute value for all states
        observations = jnp.concatenate([observations, next_observations[-1][None]], axis=0)
        
        def evaluate(observations, key):
            """Evaluate the value function at given observations."""
            actions, log_p, _ = agent.sample_actions(observations, seed=key)
            q_all = agent.critic.apply_fn({'params': agent.critic.params}, observations, actions)
            v = jnp.mean(q_all, axis=0)
            
            return v, log_p
        
        # Compute advantages using GAE if lambda > 0, otherwise use Q-V
        if agent.config.ppo.gae_lambda > 0.:
            # Average over multiple evaluations for stability
            vs, hs = jax.vmap(evaluate, in_axes=(None, 0))(observations, jax.random.split(curr_key, 10))
            
            tmp_v, tmp_logp = jnp.mean(vs, axis=0), jnp.mean(hs, axis=0)
            tmp_v -= agent.temp.apply_fn({'params': agent.temp.params}) * tmp_logp
            v, next_v = tmp_v[:-1], tmp_v[1:]
            
            rewards = batch["rewards"] - agent.temp.apply_fn({'params': agent.temp.params}) * batch["log_probs"]
            dones = jnp.bool(1 - batch["masks"])
            truncations = jnp.bool(batch["truncateds"])
            
            adv, _ = compute_gae(rewards.squeeze(), v.squeeze(), next_v.squeeze(), dones.squeeze(), truncations.squeeze(),
                                gamma=agent.config.training.discount, lam=agent.config.ppo.gae_lambda)
            adv = adv.reshape(-1)
        else:
            # Compute advantage as Q(s,a) - V(s)
            vs, hs = jax.vmap(evaluate, in_axes=(None, 0))(batch["observations"], jax.random.split(curr_key, 10))        
            tmp_v, tmp_logp = jnp.mean(vs, axis=0), jnp.mean(hs, axis=0)
            q = agent.critic.apply_fn({'params': agent.critic.params}, batch["observations"], batch["actions"]).mean(axis=0)
            adv = (q - agent.temp.apply_fn({'params': agent.temp.params}) * batch["log_probs"]) - (tmp_v - agent.temp.apply_fn({'params': agent.temp.params}) * tmp_logp)
            adv = adv.reshape(-1)
            
        
        if agent.config.ppo.store_grads:
            grads, info = jax.grad(actor_loss_fn, has_aux=True)(agent.actor.params, adv, batch)

        # Update actor and temperature
        actor_grads, actor_info = jax.grad(actor_loss_fn, has_aux=True)(agent.actor.params, adv, batch)
        new_actor = agent.actor.apply_gradients(grads=actor_grads)
        
        temp_grads, temp_info = jax.grad(temp_loss_fn, has_aux=True)(agent.temp.params, actor_info['entropy'], agent.config.ppo.target_entropy)
        new_temp = agent.temp.apply_gradients(grads=temp_grads)
            
        # Create updated agent
        agent = agent.replace(rng=new_rng, actor=new_actor, temp=new_temp)

        # Combine info dictionaries
        info = {**actor_info, **temp_info}  
        
        if agent.config.ppo.store_grads:
            info['grads'] = grads
            
        return agent, info

            
        
    @jax.jit
    def sample_actions(agent,   
                       observations: np.ndarray,
                       seed: PRNGKey,
                       temperature: float = 1.0,
                       params=None,
                       ) -> jnp.ndarray:
        
        ### random always true
        if params is None:
            params = agent.actor.params
        dist = agent.actor.apply_fn({'params': params}, observations, temperature=temperature)
        pre_actions,pre_log_ps = dist.sample_and_log_prob(seed=seed)
        
        if agent.config.training.tanh_squash_actions:
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
        dist = agent.actor.apply_fn({'params': agent.actor.params}, observations, temperature=0.)
        pre_actions,pre_log_ps = dist.sample_and_log_prob(seed=seed)
        if agent.config.training.tanh_squash_actions:
            actions = jax.nn.tanh(pre_actions)
        
        else :
            actions = pre_actions
        
        return actions


@dataclass
class NetworkConfig:
    """Network architecture configuration."""
    hidden_dims: int = 256
    activation_fn: str = 'tanh'
    use_layer_norm: bool = True
    final_fc_init_scale: float = 1e-2

    @property
    def actor_hidden_dims(self) -> Tuple[int, int]:
        return (self.hidden_dims, self.hidden_dims)
    
    @property 
    def critic_hidden_dims(self) -> Tuple[int, int]:
        return (self.hidden_dims, self.hidden_dims)

@dataclass
class OptimizerConfig:
    """Optimizer configuration."""
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    temp_lr: float = 3e-4
    momentum: float = 0.9
    b2: float = 0.999
    clip_grad_norm: float = 0.5

@dataclass
class PPOConfig:
    """PPO-specific configuration."""
    clipping_ratio: float = 0.25
    gae_lambda: float = 0.5
    num_actor_updates: int = 1
    entropy_coeff: float = 1.0
    temperature: float = 1.0
    store_grads: bool = False
    target_entropy: Optional[float] = None
    spo_loss: bool = False

@dataclass
class TrainingConfig:
    """Training configuration."""
    seed: int = 42
    discount: float = 0.99
    num_critics: int = 5
    
    # Training flags

    min_target: bool = False
    
    # Action space handling
    tanh_squash_distribution: bool = False
    tanh_squash_actions: bool = True

@dataclass
class SuperPPOConfig:
    """Complete configuration for SuperPPO."""
    network: NetworkConfig = field(default_factory=NetworkConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    @classmethod
    def from_args(cls, args) -> 'SuperPPOConfig':
        """Create config from argparse args."""
        return cls(
            network=NetworkConfig(
                hidden_dims=args.hidden_dims,
                activation_fn=args.activation_fn,
                use_layer_norm=args.use_layer_norm,
            ),
            optimizer=OptimizerConfig(),
            ppo=PPOConfig(
                clipping_ratio=args.clipping_ratio,
                gae_lambda=args.gae_lambda,
                entropy_coeff=args.entropy_coeff,
                temperature=args.temperature,
                spo_loss=args.spo_loss,
            ),
            training=TrainingConfig(
                seed=args.seed,
                discount=args.gamma,
                num_critics=args.num_critics,
                min_target=args.min_target,
                tanh_squash_distribution=not args.stable_scheme and args.bound_actions,
                tanh_squash_actions=args.stable_scheme and args.bound_actions
            )
        )


def create_learner(
    config: SuperPPOConfig,
    observations: jnp.ndarray,
    actions: jnp.ndarray,
):

    """
    Create a PPO learner using structured configuration.
    
    Args:
        config: SuperPPOConfig object containing all configuration
        observations: Example observations for network initialization
        actions: Example actions for network initialization
    """
    
    rng = jax.random.PRNGKey(config.training.seed)
    rng, actor_key, critic_key = jax.random.split(rng, 3)

    activations = nn.relu if config.network.activation_fn == 'relu' else nn.tanh
    final_fc_init_scale = config.network.final_fc_init_scale

    action_dim = actions.shape[-1]
    actor_def = Policy(
        config.network.actor_hidden_dims, 
        action_dim=action_dim,
        activations=activations,
        final_fc_init_scale=final_fc_init_scale,
        tanh_squash_distribution=config.training.tanh_squash_distribution,
        use_layer_norm=config.network.use_layer_norm,
    )

    critic_def = ensemblize(OriginalCritic, config.training.num_critics)(
        hidden_dims=config.network.critic_hidden_dims,
        use_layer_norm=config.network.use_layer_norm,
        activations=activations
    )
    critic_params = critic_def.init(critic_key, observations, actions)['params']
    critic = TrainState.create(
        apply_fn=critic_def.apply, 
        params=critic_params, 
        tx=optax.adam(learning_rate=config.optimizer.critic_lr)
    )
      
    v_def = ensemblize(OriginalV, config.training.num_critics)(
        hidden_dims=config.network.critic_hidden_dims,
        use_layer_norm=config.network.use_layer_norm,
        activations=activations
    )
    v_params = v_def.init(critic_key, observations, actions)['params']
    v = TrainState.create(
        apply_fn=v_def.apply, 
        params=v_params, 
        tx=optax.adam(learning_rate=config.optimizer.critic_lr)
    )

    actor_params = actor_def.init(actor_key, observations)['params']
    temp_def = Temperature(config.ppo.temperature)
    temp_params = temp_def.init(rng)['params']
    
    tx = optax.chain(
        optax.clip_by_global_norm(config.optimizer.clip_grad_norm),
        optax.adam(
            learning_rate=config.optimizer.actor_lr,
            b1=config.optimizer.momentum,
            b2=config.optimizer.b2
        ),
    )
    actor = TrainState.create(apply_fn=actor_def.apply, params=actor_params, tx=tx)
    temp = TrainState.create(
        apply_fn=temp_def.apply, 
        params=temp_params, 
        tx=optax.adam(
            learning_rate=config.optimizer.temp_lr,
            b1=config.optimizer.momentum,
            b2=config.optimizer.b2
        )
    )
        
    target_entropy = config.ppo.target_entropy
    if target_entropy is None:
        target_entropy = -config.ppo.entropy_coeff * action_dim

    if config.ppo.target_entropy is None:
        config.ppo.target_entropy = -config.ppo.entropy_coeff * action_dim


    return SACAgent(rng, critic=critic, target_critic=v, actor=actor, temp=temp, config=config)



