# cosine_distance_ppo.py
from typing import Tuple, Dict, Any, NamedTuple
import copy
import jax
import jax.numpy as jnp
import numpy as np
import gymnasium as gym
import wandb
from tqdm import tqdm
from functools import partial
from jaxrl_m.ppo_plus import SACAgent
from jaxrl_m.dataset import ReplayBuffer, ActorReplayBuffer
from jaxrl_m.ppo_plus import SuperPPOConfig, create_learner
from jaxrl_m.common import TrainState
import flax.linen as nn
import optax
import jax.flatten_util
import os


os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

class BatchData(NamedTuple):
    """Structured batch data for cleaner function signatures"""
    observations: jnp.ndarray
    actions: jnp.ndarray
    rewards: jnp.ndarray
    next_observations: jnp.ndarray
    masks: jnp.ndarray
    discounts: jnp.ndarray
    log_probs: jnp.ndarray
    pre_actions: jnp.ndarray
    truncateds: jnp.ndarray
    
    
# ======================== EXPERIMENTAL AGENTS SETUP ========================
def create_experimental_agents(base_config, example_transition):
        """
        Create 5 experimental agent variants:
        1. No layer norm agent
        2. Minimum target agent  
        3. On-policy only agent
        4. Normal config agent (control)
        5. Reference agent
        """
        experimental_agents = {}
        
        # 1. Agent without layer norm
        no_layernorm_config = copy.deepcopy(base_config)
        no_layernorm_config.network.use_layer_norm = False
        experimental_agents['no_layernorm'] = create_learner(
            config=no_layernorm_config,
            observations=example_transition['observations'][None],
            actions=example_transition['actions'][None]
        )
        
        # 2. Agent with minimum target
        min_target_config = copy.deepcopy(base_config)
        min_target_config.training.min_target = True
        experimental_agents['min_target'] = create_learner(
            config=min_target_config,
            observations=example_transition['observations'][None],
            actions=example_transition['actions'][None]
        )
        
        # 3. Agent for on-policy data only (same config, different data handling)
        on_policy_config = copy.deepcopy(base_config)
        experimental_agents['on_policy_only'] = create_learner(
            config=on_policy_config,
            observations=example_transition['observations'][None],
            actions=example_transition['actions'][None]
        )
        
        # 4. Off policy config agent (control)
        off_policy_config = copy.deepcopy(base_config)
        experimental_agents['off_policy_control'] = create_learner(
            config=off_policy_config,
            observations=example_transition['observations'][None],
            actions=example_transition['actions'][None]
        )
        
        # 5. Reference agent (will be trained on 100k samples)
        reference_config = copy.deepcopy(base_config)
        experimental_agents['reference'] = create_learner(
            config=reference_config,
            observations=example_transition['observations'][None],
            actions=example_transition['actions'][None]
        )
        
        return experimental_agents
    
  


def train_fresh_critic_ppo(batch_data: BatchData, agent_state: SACAgent, seed: int = 42, num_training_steps: int = 50_000) -> TrainState:
    """Train a fresh critic from scratch."""
    from jaxrl_m.ppo_plus import create_learner
    
    # Create fresh agent with new seed
    fresh_config = copy.deepcopy(agent_state.config)
    fresh_config.training.seed = seed
    
    fresh_agent = create_learner(
        config=fresh_config,
        observations=batch_data.observations[:1],
        actions=batch_data.actions[:1]
    )
    
    # Train the fresh critic
    transitions = {
        'observations': batch_data.observations,
        'actions': batch_data.actions,
        'rewards': batch_data.rewards,
        'next_observations': batch_data.next_observations,
        'masks': batch_data.masks,
        'discounts': jnp.ones_like(batch_data.rewards)
    }
    
    return fresh_agent.update_critics_seq(transitions, n_updates=num_training_steps).critic

def compute_ppo_actor_gradient(agent: SACAgent, batch_data: BatchData, advantages: jnp.ndarray) -> jnp.ndarray:
    """Compute PPO actor gradient using advantages."""
    def actor_loss_fn(actor_params):
        dist = agent.actor.apply_fn({'params': actor_params}, batch_data.observations)
        pre_log_probs = dist.log_prob(batch_data.pre_actions)
        
        # Apply tanh squashing correction if needed
        new_logp = (pre_log_probs - jnp.sum(2 * (jnp.log(2) + batch_data.pre_actions - jax.nn.softplus(2 * batch_data.pre_actions)), axis=-1)
                   if agent.config.training.tanh_squash_actions else pre_log_probs)
        
        # PPO clipped objective
        ratio = jnp.exp(new_logp - batch_data.log_probs)
        clip_coef = agent.config.ppo.clipping_ratio
        
        clipped_advantages = batch_data.masks * advantages
        actor_loss1 = clipped_advantages * ratio
        actor_loss2 = clipped_advantages * jnp.clip(ratio, 1 - clip_coef, 1 + clip_coef)
        return -jnp.minimum(actor_loss1, actor_loss2).mean()
    
    grads = jax.grad(actor_loss_fn)(agent.actor.params)
    flat_grads, _ = jax.flatten_util.ravel_pytree(grads)
    return flat_grads

def compute_advantages_with_critic(agent: SACAgent, critic: TrainState, batch_data: BatchData) -> jnp.ndarray:
    """Compute advantages using a specific critic."""
    temp_agent = agent.replace(critic=critic)
    
    # Compute Q-values and V-values efficiently
    q_values = critic.apply_fn({'params': critic.params}, batch_data.observations, batch_data.actions).mean(axis=0)
    
    # Vectorized V-value computation
    sample_keys = jax.random.split(jax.random.PRNGKey(0), 10)
    
    @partial(jax.vmap, in_axes=(None, 0))
    def evaluate_v(obs, key):
        actions_sample, log_p, _ = temp_agent.sample_actions(obs, seed=key)
        q_all = critic.apply_fn({'params': critic.params}, obs, actions_sample)
        return jnp.mean(q_all, axis=0), log_p
    
    vs, hs = evaluate_v(batch_data.observations, sample_keys)
    v_values, tmp_logp = jnp.mean(vs, axis=0), jnp.mean(hs, axis=0)
    
    # Compute advantages with entropy regularization
    temp_value = temp_agent.temp.apply_fn({'params': temp_agent.temp.params})
    advantages = (q_values - temp_value * batch_data.log_probs) - (v_values - temp_value * tmp_logp)
    
    return advantages.reshape(-1)

def cosine_similarity(grad1: jnp.ndarray, grad2: jnp.ndarray) -> float:
    """Compute cosine similarity between two flattened gradients."""
    norm1, norm2 = jnp.linalg.norm(grad1), jnp.linalg.norm(grad2)
    return float(jnp.dot(grad1, grad2) / (norm1 * norm2) if min(norm1, norm2) > 1e-8 else 0.0)


def compute_q_values_on_data(agent: SACAgent, batch_data: BatchData) -> jnp.ndarray:
    """Compute Q-values for an agent on given batch data."""
    q_values = agent.critic.apply_fn(
        {'params': agent.critic.params}, 
        batch_data.observations, 
        batch_data.actions
    ).mean(axis=0)  # Average over ensemble if multiple critics
    return q_values


def compute_q_bias_metrics(
    agent_q_values: Dict[str, jnp.ndarray],
    reference_q_values: jnp.ndarray
) -> Dict[str, float]:
    """
    Compute Q-function bias metrics using reference as ground truth.
    
    Args:
        agent_q_values: Dictionary mapping agent names to their Q-values
        reference_q_values: Reference Q-values (ground truth)
        
    Returns:
        Dictionary containing bias metrics for each agent
    """
    bias_metrics = {}
    
    for agent_name, q_values in agent_q_values.items():
        if agent_name == 'reference':
            continue  # Skip reference vs reference
            
        bias = q_values - reference_q_values
        
        # Compute various bias metrics
        mean_bias = float(jnp.mean(bias))
        absolute_bias = float(jnp.mean(jnp.abs(bias)))
        relative_bias = float(jnp.mean(bias / (jnp.abs(reference_q_values) + 1e-8)))
        rmse = float(jnp.sqrt(jnp.mean(bias ** 2)))
        
        bias_metrics[f"train/q_mean_bias_{agent_name}"] = mean_bias
        bias_metrics[f"train/q_absolute_bias_{agent_name}"] = absolute_bias
        bias_metrics[f"train/q_relative_bias_{agent_name}"] = relative_bias
        bias_metrics[f"train/q_rmse_{agent_name}"] = rmse
    
    return bias_metrics


def evaluate_experimental_agents_gradients(
    experimental_agents: Dict[str, SACAgent],
    actor_buffer,  # ActorReplayBuffer 
    replay_buffer,  # ReplayBuffer
    reference_buffer,  # ReplayBuffer
    step_num: int,
    critic_training_steps: int = 5000,
    evaluation_batch_size: int = 5000
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate gradients for experimental agents using different buffer combinations.
    
    Args:
        experimental_agents: Dictionary of experimental agents
        actor_buffer: Buffer containing on-policy actor data
        replay_buffer: Buffer containing off-policy replay data  
        reference_buffer: Buffer containing reference data
        step_num: Current training step
        critic_training_steps: Number of steps to train critics
        evaluation_batch_size: Size of evaluation batches
    
    Returns:
        Dictionary mapping agent names to their gradient evaluation results
    """
    print(f"\n=== Evaluating Experimental Agents Gradients at Step {step_num} ===")
    results = {}
    
    # Convert buffers to BatchData format
    def buffer_to_batch_data(buffer):
        data = buffer.get_all()
        return BatchData(**jax.tree.map(jax.device_put, data))
    
    # Prepare buffer data
    actor_data = buffer_to_batch_data(actor_buffer)
    replay_data = buffer_to_batch_data(replay_buffer)
    reference_data = buffer_to_batch_data(reference_buffer)
    
    
    print("Buffer sizes:")
    print(actor_data.observations.shape)
    print(replay_data.observations.shape)
    print(reference_data.observations.shape)
    
    
    agent_gradients = {}
    
    
    print(f"Buffer sizes - Actor: {actor_data.observations.shape[0]}, Replay: {replay_data.observations.shape[0]}, Reference: {reference_data.observations.shape[0]}")
    
    for agent_name, agent in experimental_agents.items():
        print(f"\nEvaluating agent: {agent_name}")
        
        if agent_name == 'no_layernorm':
            critic_training_data = replay_data  # CHANGE THIS: Choose buffer for critic training
            critic_training_steps = 2000
        elif agent_name == 'min_target':
            critic_training_data = replay_data  # CHANGE THIS: Choose buffer for critic training
            critic_training_steps = 2000
        elif agent_name == 'on_policy_only':
            critic_training_data = actor_data  # CHANGE THIS: Choose buffer for critic training
            critic_training_steps = 2000
        elif agent_name == 'off_policy_control':
            critic_training_data = replay_data  # CHANGE THIS: Choose buffer for critic training
            critic_training_steps = 2000
        elif agent_name == 'reference':
            critic_training_data = reference_data  # CHANGE THIS: Choose buffer for critic training
            critic_training_steps = 2000    
        
        print(f"Training {agent_name} critic on buffer with {critic_training_data.observations.shape[0]} samples")
        # Train critic for this agent
        trained_critic = train_fresh_critic_ppo(
            critic_training_data, 
            agent, 
            step_num + hash(agent_name) % 1000,  # Unique seed per agent
            critic_training_steps
        )
        updated_agent = agent.replace(critic=trained_critic)
        
        if agent_name == 'reference':
            gradient_data = reference_data
        else:
            gradient_data = actor_data
        
        print(f"Computing {agent_name} gradients on buffer with {gradient_data.observations.shape[0]} samples")
        advantages = compute_advantages_with_critic(updated_agent, trained_critic, gradient_data)
        agent_gradients[agent_name] = compute_ppo_actor_gradient(updated_agent, gradient_data, advantages)
        experimental_agents[agent_name] = updated_agent
    
    # Compute Q-values for bias analysis
    print(f"\nComputing Q-values for bias analysis on actor_data ({actor_data.observations.shape[0]} samples)")
    agent_q_values = {}
    for agent_name, agent in experimental_agents.items():
        print(f"Computing Q-values for {agent_name}")
        agent_q_values[agent_name] = compute_q_values_on_data(agent, actor_data)
    
    
    gradient_metrics = compute_cosines(agent_gradients)
    # Compute bias metrics using reference as ground truth
    bias_metrics = compute_q_bias_metrics(agent_q_values, agent_q_values['reference'])
    
    return experimental_agents, gradient_metrics, bias_metrics


def compute_cosines(agent_gradients: Dict[str, jnp.ndarray]) -> Dict[str, float]:
    """
    Compute cosine similarities between different agent gradients.
    
    Args:
        agent_gradients: Dictionary mapping agent names to their flattened gradients
        
    Returns:
        Dictionary containing cosine similarity metrics similar to run_lqr.py
    """
    
    rslt = {"train/cosine_true_normal": cosine_similarity(agent_gradients['off_policy_control'], agent_gradients['reference']),
            "train/cosine_true_min": cosine_similarity(agent_gradients['min_target'], agent_gradients['reference']),
            "train/cosine_true_on": cosine_similarity(agent_gradients['on_policy_only'], agent_gradients['reference']),
            "train/cosine_true_no": cosine_similarity(agent_gradients['no_layernorm'], agent_gradients['reference'])
            }
   
    return rslt


    
    

