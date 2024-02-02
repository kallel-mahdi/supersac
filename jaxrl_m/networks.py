"""Common networks used in RL.

This file contains nn.Module definitions for common networks used in RL. It is divided into three sets:

1) Common Networks: MLP
2) Common RL Networks:
    For discrete action spaces: DiscreteCritic is a Q-function
    For continuous action spaces: Critic, ValueCritic, and Policy provide the Q-function, value function, and policy respectively.
    For ensembling: ensemblize() provides a wrapper for creating ensembles of networks (e.g. for min-Q / double-Q)
3) Meta Networks for vision tasks:
    WithEncoder: Combines a fully connected network with an encoder network (encoder may come from jaxrl_m.vision)
    ActorCritic: Same as WithEncoder, but for possibly many different networks (e.g. actor, critic, value)
"""

from jaxrl_m.typing import *

import flax.linen as nn
import jax.numpy as jnp

import distrax
import flax.linen as nn
import jax.numpy as jnp

###############################
#
#  Common Networks
#
###############################


def default_init(scale: Optional[float] = 1.0):
    return nn.initializers.variance_scaling(scale, "fan_avg", "uniform")


class MLP(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    activate_final: bool = False
    use_layer_norm: bool = True
    scale_final: Optional[float] = None

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:

        for i, size in enumerate(self.hidden_dims):
            if i + 1 == len(self.hidden_dims) and self.scale_final is not None:
                x = nn.Dense(size,
                             kernel_init=default_init(self.scale_final))(x)
            else:
                x = nn.Dense(size, kernel_init=default_init())(x)

            if i + 1 < len(self.hidden_dims) or self.activate_final:
                if self.use_layer_norm:
                    x = nn.LayerNorm()(x)
                x = self.activations(x)
        return x




class Critic(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    use_layer_norm: bool = True
    scale_final: Optional[float] = None

    @nn.compact
    def __call__(self, observations: jnp.ndarray, actions: jnp.ndarray,
                *args,**kwargs) -> jnp.ndarray:
        inputs = jnp.concatenate([observations, actions], -1)
        critic = MLP((*self.hidden_dims, 1), activations=self.activations,
                     use_layer_norm=self.use_layer_norm)(inputs,*args, **kwargs)
        
        #critic = nn.tanh(critic) ## Bonus
        return jnp.squeeze(critic, -1)


class DeterministicPolicy(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    final_fc_init_scale: float = 1.0
    
    @nn.compact
    def __call__(
        self, observations: jnp.ndarray, 
    ) -> distrax.Distribution:
        outputs = MLP(
            self.hidden_dims,
            activations=nn.relu,### new
            activate_final=True,
            use_layer_norm=True,
        )(observations)

        outputs = nn.Dense(
            self.action_dim, kernel_init=default_init(self.final_fc_init_scale)
        )(outputs)

        outputs = nn.tanh(outputs)
        
        return outputs
    


def ensemblize(cls, num_qs, out_axes=0, **kwargs):
    """
    Useful for making ensembles of Q functions (e.g. double Q in SAC).

    Usage:

        critic_def = ensemblize(Critic, 2)(hidden_dims=hidden_dims)

    """
    split_rngs = kwargs.pop("split_rngs", {})
    return nn.vmap(
        cls,
        variable_axes={"params": 0},
        split_rngs={**split_rngs, "params": True},
        in_axes=None,
        out_axes=out_axes,
        axis_size=num_qs,
        **kwargs
    )


 
    
class Policy(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    log_std_min: Optional[float] = -10
    log_std_max: Optional[float] = 2
    tanh_squash_distribution: bool = True
    state_dependent_std: bool = True
    final_fc_init_scale: float = 1e-2

    @nn.compact
    def __call__(
        self, observations: jnp.ndarray, temperature: float = 1.0
    ) -> distrax.Distribution:
        outputs = MLP(
            self.hidden_dims,
            activations=nn.relu,### new
            activate_final=True,
        )(observations)

        means = nn.Dense(
            self.action_dim, kernel_init=default_init(self.final_fc_init_scale)
        )(outputs)
        if self.state_dependent_std:
            log_stds = nn.Dense(
                self.action_dim, kernel_init=default_init(self.final_fc_init_scale)
            )(outputs)
        else:
            log_stds = self.param("log_stds", nn.initializers.zeros, (self.action_dim,))

        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)

        distribution = distrax.MultivariateNormalDiag(
            loc=means, scale_diag=jnp.exp(log_stds) * temperature
        )
        if self.tanh_squash_distribution:
            distribution = TransformedWithMode(
                distribution, distrax.Block(distrax.Tanh(), ndims=1)
            )

        return distribution


class TransformedWithMode(distrax.Transformed):
    def mode(self) -> jnp.ndarray:
        return self.bijector.forward(self.distribution.mode())


