
from typing import Any, Callable, Dict, List, Optional, Sequence, Type, Union

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.linen.dtypes import canonicalize_dtype
from flax.linen.module import Module  # pylint: disable=g-multiple-import
from flax.linen.module import compact, merge_param
from flax.linen.normalization import (_canonicalize_axes, _compute_stats,
                                      _normalize)
from jax import lax
from jax.nn import initializers

from jaxrl_m.common import TrainState, nonpytree_field
from jaxrl_m.networks import OriginalCritic, Policy
from jaxrl_m.typing import *


def get_batch(i,batches):
    return  jax.tree_map(lambda x: x[i], batches)

def body(i,val):
    agent,batches = val
    return (agent.update_critics(get_batch(i,batches)),batches)



################ TO MOVE TO NETWORKS EVENTUALLY ##############


class BatchRenorm(Module):
  """BatchRenorm Module, implemented based on the Batch Renormalization paper (https://arxiv.org/abs/1702.03275).
  and adapted from Flax's BatchNorm implementation: 
  https://github.com/google/flax/blob/ce8a3c74d8d1f4a7d8f14b9fb84b2cc76d7f8dbf/flax/linen/normalization.py#L228


  Attributes:
    use_running_average: if True, the statistics stored in batch_stats will be
      used instead of computing the batch statistics on the input.
    axis: the feature or non-batch axis of the input.
    momentum: decay rate for the exponential moving average of the batch
      statistics.
    epsilon: a small float added to variance to avoid dividing by zero.
    dtype: the dtype of the result (default: infer from input and params).
    param_dtype: the dtype passed to parameter initializers (default: float32).
    use_bias:  if True, bias (beta) is added.
    use_scale: if True, multiply by scale (gamma). When the next layer is linear
      (also e.g. nn.relu), this can be disabled since the scaling will be done
      by the next layer.
    bias_init: initializer for bias, by default, zero.
    scale_init: initializer for scale, by default, one.
    axis_name: the axis name used to combine batch statistics from multiple
      devices. See `jax.pmap` for a description of axis names (default: None).
    axis_index_groups: groups of axis indices within that named axis
      representing subsets of devices to reduce over (default: None). For
      example, `[[0, 1], [2, 3]]` would independently batch-normalize over the
      examples on the first two and last two devices. See `jax.lax.psum` for
      more details.
    use_fast_variance: If true, use a faster, but less numerically stable,
      calculation for the variance.
  """

  use_running_average: Optional[bool] = None
  axis: int = -1
  momentum: float = 0.99
  epsilon: float = 0.001
  dtype: Optional[Dtype] = None
  param_dtype: Dtype = jnp.float32
  use_bias: bool = True
  use_scale: bool = True
  bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.zeros
  scale_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.ones
  axis_name: Optional[str] = None
  axis_index_groups: Any = None
  use_fast_variance: bool = True

  @compact
  def __call__(self, x, use_running_average: Optional[bool] = None):
    """
    Args:
      x: the input to be normalized.
      use_running_average: if true, the statistics stored in batch_stats will be
        used instead of computing the batch statistics on the input.

    Returns:
      Normalized inputs (the same shape as inputs).
    """

    use_running_average = merge_param(
        'use_running_average', self.use_running_average, use_running_average
    )
    feature_axes = _canonicalize_axes(x.ndim, self.axis)
    reduction_axes = tuple(i for i in range(x.ndim) if i not in feature_axes)
    feature_shape = [x.shape[ax] for ax in feature_axes]

    ra_mean = self.variable(
        'batch_stats',
        'mean',
        lambda s: jnp.zeros(s, jnp.float32),
        feature_shape,
    )
    ra_var = self.variable(
        'batch_stats', 'var', lambda s: jnp.ones(s, jnp.float32), feature_shape
    )

    r_max = self.variable(
        'batch_stats',
        'r_max',
        lambda s: s,
        3,
    )
    d_max = self.variable(
        'batch_stats',
        'd_max',
        lambda s: s,
        5,
    )
    steps = self.variable(
        'batch_stats',
        'steps',
        lambda s: s,
        0,
    )

    if use_running_average:
      mean, var = ra_mean.value, ra_var.value
      custom_mean = mean
      custom_var = var
    else:
      mean, var = _compute_stats(
          x,
          reduction_axes,
          dtype=self.dtype,
          axis_name=self.axis_name if not self.is_initializing() else None,
          axis_index_groups=self.axis_index_groups,
          use_fast_variance=self.use_fast_variance,
      )
      custom_mean = mean
      custom_var = var
      if not self.is_initializing():
        # The code below is implemented following the Batch Renormalization paper
        r = 1
        d = 0
        std = jnp.sqrt(var + self.epsilon)
        ra_std = jnp.sqrt(ra_var.value + self.epsilon)
        r = jax.lax.stop_gradient(std / ra_std)
        r = jnp.clip(r, 1 / r_max.value, r_max.value)
        d = jax.lax.stop_gradient((mean - ra_mean.value) / ra_std)
        d = jnp.clip(d, -d_max.value, d_max.value)
        tmp_var = var / (r**2)
        tmp_mean = mean - d * jnp.sqrt(custom_var) / r

        # Warm up batch renorm for 100_000 steps to build up proper running statistics
        warmed_up = jnp.greater_equal(steps.value, 100_000).astype(jnp.float32)
        custom_var = warmed_up * tmp_var + (1. - warmed_up) * custom_var
        custom_mean = warmed_up * tmp_mean + (1. - warmed_up) * custom_mean

        ra_mean.value = (
            self.momentum * ra_mean.value + (1 - self.momentum) * mean
        )
        ra_var.value = self.momentum * ra_var.value + (1 - self.momentum) * var
        steps.value += 1



    return _normalize(
        self,
        x,
        custom_mean,
        custom_var,
        reduction_axes,
        feature_axes,
        self.dtype,
        self.param_dtype,
        self.epsilon,
        self.use_bias,
        self.use_scale,
        self.bias_init,
        self.scale_init,
    )


class Critic(nn.Module):
    net_arch: Sequence[int]
    activation_fn: Type[nn.Module]
    batch_norm_momentum: float
    use_layer_norm: bool = False
    dropout_rate: Optional[float] = None
    use_batch_norm: bool = False
    bn_mode: str = "bn"

    @nn.compact
    def __call__(self, x: jnp.ndarray, action: jnp.ndarray, train) -> jnp.ndarray:
        if 'bn' in self.bn_mode:
            BN = nn.BatchNorm
        elif 'brn' in self.bn_mode:
            BN = BatchRenorm
        else:
            raise NotImplementedError

        x = jnp.concatenate([x, action], -1)

        if self.use_batch_norm:
            x = BN(use_running_average=not train, momentum=self.batch_norm_momentum)(x)
        else:
            # Hack to make flax return state_updates. Is only necessary such that the downstream
            # functions have the same function signature.
            x_dummy = BN(use_running_average=not train)(x)

        for n_units in self.net_arch:
            x = nn.Dense(n_units)(x)

            if self.dropout_rate is not None and self.dropout_rate > 0:
                x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=False)
            
            if self.use_layer_norm:
                x = nn.LayerNorm()(x)

            x = self.activation_fn()(x)

            if self.use_batch_norm:
                x = BN(use_running_average=not train, momentum=self.batch_norm_momentum)(x)
            else:
                x_dummy = BN(use_running_average=not train)(x)
        x = nn.Dense(1)(x)
        return x


###################################""

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
            observations = jnp.repeat(batch['observations'], 10, axis=0)
            discounts = jnp.repeat(batch['discounts'], 10, axis=0)
            masks = jnp.int32(jnp.repeat(batch['masks'], 10, axis=0))

            dist = agent.actor(observations, params=actor_params)
            actions, log_probs = dist.sample_and_log_prob(seed=curr_key)
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
            temp = TrainState.create(temp_def, temp_params, tx=optax.rmsprop(learning_rate=temp_lr))
            #actor = TrainState.create(actor_def, actor_params, tx=optax.rmsprop(learning_rate=actor_lr))
            actor = TrainState.create(actor_def, actor_params, tx=optax.adam(learning_rate=actor_lr,b1=0.1))
        
            
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

