# cosine_distance_ppo.py
from typing import Tuple, Dict, Any
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
from jaxrl_m.common import TrainState
import flax.linen as nn
from flax.training import train_state
import optax
import jax.flatten_util
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

def collect_rollouts_ppo(
    agent: SACAgent,
    env: gym.Env,
    num_steps: int = 100_000,
    deterministic: bool = True,
    num_parallel_envs: int = 8,
    gamma: float = 0.99,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Collect rollouts using the current policy for training true gradient critic.
    Uses vectorized environments for much faster data collection.
    
    Args:
        agent: Current PPO agent
        env: Single environment (used to get env_id for creating vectorized envs)
        num_steps: Total number of steps to collect
        deterministic: Whether to use deterministic policy (True for true gradient)
        num_parallel_envs: Number of parallel environments to use
        gamma: Discount factor
        
    Returns:
        Tuple of (observations, actions, rewards, next_observations, masks, discounts)
    """
    import gymnasium as gym
    
    # Create vectorized environment
    env_id = env.spec.id
    
    vec_env = gym.make_vec(
        env_id,
        num_envs=num_parallel_envs,
        vectorization_mode="sync"
    )
    print(f"Created vectorized environment with {num_parallel_envs} parallel environments")

    observations = []
    actions = []
    rewards = []
    next_observations = []
    masks = []
    discounts = []
    log_probs = []
    pre_actions = []
    
    # Reset all environments
    obs, _ = vec_env.reset()
    steps_collected = 0
    
    # Calculate steps per environment
    steps_per_env = num_steps // num_parallel_envs
    total_steps_needed = steps_per_env * num_parallel_envs
    
    progress_bar = tqdm(total=total_steps_needed, desc="Collecting parallel rollouts")
    
    discount = jnp.ones(num_parallel_envs)
    rng_key = jax.random.PRNGKey(0)
    
    while steps_collected < total_steps_needed:
        rng_key, sample_key = jax.random.split(rng_key)
        
        if deterministic:
            action = agent.deterministic_action(obs)
            # For deterministic actions, we need to compute log_prob separately
            dist = agent.actor(obs)
            if agent.config.training.tanh_squash_actions:
                pre_action = jnp.arctanh(jnp.clip(action, -0.999, 0.999))
                log_prob = dist.log_prob(pre_action)
                log_prob = log_prob - jnp.sum(2 * (jnp.log(2) + pre_action - jax.nn.softplus(2 * pre_action)), axis=-1)
            else:
                pre_action = action
                log_prob = dist.log_prob(action)
        else:
            action, log_prob, pre_action = agent.sample_actions(obs, sample_key)
        
        # Execute actions in all environments
        next_obs, reward, terminated, truncated, _ = vec_env.step(action)
        done = terminated | truncated
        mask = 1.0 - terminated.astype(np.float32)
        
        # Store transitions for all environments
        observations.append(obs.copy())
        actions.append(action.copy())
        rewards.append(reward.copy())
        next_observations.append(next_obs.copy())
        masks.append(mask.copy())
        discounts.append(discount.copy())
        log_probs.append(log_prob.copy())
        pre_actions.append(pre_action.copy())
        
        discount = discount * gamma
        if any(done): 
            discount = discount.at[done==True].set(1.)
        
        obs = next_obs
        steps_collected += num_parallel_envs
        progress_bar.update(num_parallel_envs)

    progress_bar.close()
    vec_env.close()
        
    # Flatten the collected data
    observations = np.concatenate(observations, axis=0)
    actions = np.concatenate(actions, axis=0)
    rewards = np.concatenate(rewards, axis=0)
    next_observations = np.concatenate(next_observations, axis=0)
    masks = np.concatenate(masks, axis=0)
    discounts = np.concatenate(discounts, axis=0)
    log_probs = np.concatenate(log_probs, axis=0)
    pre_actions = np.concatenate(pre_actions, axis=0)
    
    print(f"Collected {observations.shape[0]} transitions using {num_parallel_envs} parallel environments")
    
    return (
        observations.astype(np.float32),
        actions.astype(np.float32),
        rewards.astype(np.float32),
        next_observations.astype(np.float32),
        masks.astype(np.float32),
        discounts.astype(np.float32),
        log_probs.astype(np.float32),
        pre_actions.astype(np.float32)
    )


def train_fresh_critic_ppo(
    observations: jnp.ndarray,
    actions: jnp.ndarray,
    rewards: jnp.ndarray,
    next_observations: jnp.ndarray,
    masks: jnp.ndarray,
    agent_state: SACAgent,
    seed: int = 42,
    num_training_steps: int = 50_000
) -> train_state.TrainState:
    """
    Train a fresh critic from scratch.
    """
    from jaxrl_m.ppo_plus import create_learner
    import copy
    
    # Create a fresh agent with same config but new parameters
    sample_obs = observations[:1]
    sample_act = actions[:1]
    
    # Extract and copy the config from existing agent
    original_config = agent_state.config
    
    # Create a fresh config with new seed
    fresh_config = copy.deepcopy(original_config)
    fresh_config.training.seed = seed
    
    # Create fresh agent (this will have fresh parameters due to new seed)
    fresh_agent = create_learner(
        config=fresh_config,
        observations=sample_obs,
        actions=sample_act
    )
    
    # Train the fresh critic
    transitions = {
        'observations': jax.device_put(observations),
        'actions': jax.device_put(actions),
        'rewards': jax.device_put(rewards),
        'next_observations': jax.device_put(next_observations),
        'masks': jax.device_put(masks),
        'discounts': jnp.ones_like(rewards)
    }
    
    fresh_agent = fresh_agent.update_critics_seq2(transitions, num_updates=num_training_steps)
    
    return fresh_agent.critic


def compute_ppo_actor_gradient(
    agent: SACAgent,
    observations: jnp.ndarray,
    actions: jnp.ndarray,
    log_probs: jnp.ndarray,
    pre_actions: jnp.ndarray,
    advantages: jnp.ndarray,
    discounts: jnp.ndarray,
    masks: jnp.ndarray
) -> jnp.ndarray:
    """
    Compute PPO actor gradient using advantages.
    
    Args:
        agent: PPO agent
        observations: State observations
        actions: Actions taken
        log_probs: Log probabilities of actions
        pre_actions: Pre-tanh actions (if using tanh squashing)
        advantages: Computed advantages
        discounts: Discount factors
        masks: Episode masks
        
    Returns:
        Flattened actor gradient
    """
    # Create batch dict
    batch = {
        'observations': observations,
        'actions': actions,
        'log_probs': log_probs,
        'pre_actions': pre_actions,
        'discounts': discounts,
        'masks': masks,
        'rewards': jnp.zeros_like(masks),  # Not used in actor loss
        'next_observations': observations,  # Not used in actor loss
        'truncateds': jnp.zeros_like(masks)  # Not used in actor loss
    }
    
    def actor_loss_fn(actor_params):
        """PPO actor loss function"""
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
        logratio = new_logp - batch['log_probs']
        ratio = jnp.exp(logratio)

        # PPO clipped objective
        clip_coef = agent.config.ppo.clipping_ratio
        
        actor_loss1 = masks * advantages * ratio
        actor_loss2 = masks * advantages * jnp.clip(ratio, 1 - clip_coef, 1 + clip_coef)

        # Apply discounting if configured
        if agent.config.training.discount_actor:
            actor_loss = -jnp.minimum(discounts * actor_loss1, discounts * actor_loss2).sum() / (discounts.sum())
        else:
            actor_loss = -jnp.minimum(actor_loss1, actor_loss2).mean()
            
        return actor_loss
    
    # Compute gradient
    grads = jax.grad(actor_loss_fn)(agent.actor.params)
    
    # Flatten gradient
    flat_grads, _ = jax.flatten_util.ravel_pytree(grads)
    return flat_grads


def compute_advantages_with_critic(
    agent: SACAgent,
    critic: train_state.TrainState,
    observations: jnp.ndarray,
    actions: jnp.ndarray,
    rewards: jnp.ndarray,
    next_observations: jnp.ndarray,
    masks: jnp.ndarray,
    discounts: jnp.ndarray,
    log_probs: jnp.ndarray
) -> jnp.ndarray:
    """
    Compute advantages using a specific critic (either current or fresh).
    """
    # Create temporary agent with the specified critic
    temp_agent = agent.replace(critic=critic)
    
    # Compute Q-values for current state-action pairs
    q_values = critic.apply_fn({'params': critic.params}, observations, actions).mean(axis=0)
    
    # Compute V-values by sampling actions and averaging Q-values
    rng_key = jax.random.PRNGKey(0)
    sample_keys = jax.random.split(rng_key, 10)
    
    def evaluate_v(obs, key):
        actions_sample, log_p, _ = temp_agent.sample_actions(obs, seed=key)
        q_all = critic.apply_fn({'params': critic.params}, obs, actions_sample)
        v = jnp.mean(q_all, axis=0)
        return v, log_p
    
    vs, hs = jax.vmap(evaluate_v, in_axes=(None, 0))(observations, sample_keys)
    v_values, tmp_logp = jnp.mean(vs, axis=0), jnp.mean(hs, axis=0)
    
    # Compute advantages as Q(s,a) - V(s) with entropy regularization
    temp_value = temp_agent.temp()
    advantages = (q_values - temp_value * log_probs) - (v_values - temp_value * tmp_logp)
    
    return advantages.reshape(-1)


def compute_cosine_similarity(grad1: jnp.ndarray, grad2: jnp.ndarray) -> float:
    """
    Compute cosine similarity between two flattened gradients.
    
    Args:
        grad1, grad2: Flattened gradient vectors
        
    Returns:
        Cosine similarity (-1 to 1, where 1 means perfect alignment)
    """
    # Compute norms
    norm1 = jnp.linalg.norm(grad1)
    norm2 = jnp.linalg.norm(grad2)
    
    # Handle zero gradients
    if norm1 < 1e-8 or norm2 < 1e-8:
        return 0.0
    
    # Compute cosine similarity
    dot_product = jnp.dot(grad1, grad2)
    cosine_similarity = dot_product / (norm1 * norm2)
    
    return float(cosine_similarity)


def evaluate_gradient_quality_ppo(
    agent: SACAgent,
    env: gym.Env,
    replay_buffer: ReplayBuffer,
    step_num: int,
    batch_size: int = 256,
    rollout_steps: int = 50_000,
    critic_training_steps: int = 5_000,
    evaluation_batch_size: int = 5000,
    num_parallel_envs: int = 10
) -> Dict[str, float]:
    """
    Main function to evaluate PPO gradient quality by comparing with true gradient.
    """
    print(f"\n=== PPO Gradient Quality Evaluation at Step {step_num} ===")
    
    # 1. Collect fresh rollouts using current policy
    print("Collecting fresh rollouts...")
    rollout_data = collect_rollouts_ppo(
        agent, env, num_steps=rollout_steps, deterministic=False, 
        num_parallel_envs=num_parallel_envs, gamma=0.99
    )
    rollout_obs, rollout_acts, rollout_rews, rollout_next_obs, rollout_masks, rollout_discounts, rollout_log_probs, rollout_pre_acts = rollout_data
    
    # 2. Train fresh critic on rollout data
    print("Training fresh critic...")
    true_critic = train_fresh_critic_ppo(
        rollout_obs, rollout_acts, rollout_rews, rollout_next_obs, rollout_masks,
        agent, seed=step_num, num_training_steps=critic_training_steps
    )
    
    # 3. Sample evaluation batches
    # Off-policy batch from replay buffer
    off_policy_data = replay_buffer.sample(evaluation_batch_size)
    
    # On-policy batch from fresh rollouts
    rollout_size = rollout_obs.shape[0]
    eval_indices = np.random.choice(rollout_size, size=evaluation_batch_size, replace=False)
    
    true_batch_obs = jax.device_put(rollout_obs[eval_indices])
    true_batch_acts = jax.device_put(rollout_acts[eval_indices])
    true_batch_rews = jax.device_put(rollout_rews[eval_indices])
    true_batch_next_obs = jax.device_put(rollout_next_obs[eval_indices])
    true_batch_masks = jax.device_put(rollout_masks[eval_indices])
    true_batch_discounts = jax.device_put(rollout_discounts[eval_indices])
    true_batch_log_probs = jax.device_put(rollout_log_probs[eval_indices])
    true_batch_pre_acts = jax.device_put(rollout_pre_acts[eval_indices])
    
    # 4. Compute advantages for both batches
    print("Computing advantages...")
    
    # Off-policy advantages (using replay buffer data and current critic)
    off_policy_advantages = compute_advantages_with_critic(
        agent, agent.critic,
        off_policy_data['observations'], off_policy_data['actions'],
        off_policy_data['rewards'], off_policy_data['next_observations'],
        off_policy_data['masks'], off_policy_data['discounts'],
        off_policy_data['log_probs']
    )
    
    # True advantages (using fresh rollout data and fresh critic)
    true_advantages = compute_advantages_with_critic(
        agent, true_critic,
        true_batch_obs, true_batch_acts, true_batch_rews,
        true_batch_next_obs, true_batch_masks, true_batch_discounts,
        true_batch_log_probs
    )
    
    # True approximation advantages (using fresh rollout data but current critic)
    true_approx_advantages = compute_advantages_with_critic(
        agent, agent.critic,
        true_batch_obs, true_batch_acts, true_batch_rews,
        true_batch_next_obs, true_batch_masks, true_batch_discounts,
        true_batch_log_probs
    )
    
    # 5. Compute gradients
    print("Computing gradients...")
    
    # Off-policy gradient
    off_policy_grad = compute_ppo_actor_gradient(
        agent,
        off_policy_data['observations'], off_policy_data['actions'],
        off_policy_data['log_probs'], off_policy_data['pre_actions'],
        off_policy_advantages, off_policy_data['discounts'], off_policy_data['masks']
    )
    
    # True gradient
    true_grad = compute_ppo_actor_gradient(
        agent,
        true_batch_obs, true_batch_acts, true_batch_log_probs, true_batch_pre_acts,
        true_advantages, true_batch_discounts, true_batch_masks
    )
    
    # True approximation gradient
    true_approx_grad = compute_ppo_actor_gradient(
        agent,
        true_batch_obs, true_batch_acts, true_batch_log_probs, true_batch_pre_acts,
        true_approx_advantages, true_batch_discounts, true_batch_masks
    )
    
    # 6. Compute cosine similarities
    print("Computing cosine similarities...")
    
    cosine_off_policy_vs_true = compute_cosine_similarity(off_policy_grad, true_grad)
    cosine_true_approx_vs_true = compute_cosine_similarity(true_approx_grad, true_grad)
    
    # 7. Compute gradient norms
    norm_off_policy = float(jnp.linalg.norm(off_policy_grad))
    norm_true = float(jnp.linalg.norm(true_grad))
    norm_true_approx = float(jnp.linalg.norm(true_approx_grad))
    
    # 8. Create results dictionary
    results = {
        'cosine_similarity/off_policy_vs_true': cosine_off_policy_vs_true,
        'cosine_similarity/true_approx_vs_true': cosine_true_approx_vs_true,
        'step': step_num
    }
    
    # 9. Log results
    print(f"Off-policy vs True Cosine Similarity: {cosine_off_policy_vs_true:.4f}")
    print(f"True Approx vs True Cosine Similarity: {cosine_true_approx_vs_true:.4f}")
    print(f"Gradient Norms - Off-policy: {norm_off_policy:.4f}, True: {norm_true:.4f}, True Approx: {norm_true_approx:.4f}")
    
    # Interpret results
    print("\nResult Interpretation:")
    print("-" * 40)
    print(f"Off-policy gradient has {cosine_off_policy_vs_true:.6f} cosine similarity with true gradient")
    print(f"True approx gradient has {cosine_true_approx_vs_true:.6f} cosine similarity with true gradient")
    
    # Log to wandb if available
    try:
        wandb.log(results, step=step_num)
        print("Results logged to Wandb.")
    except Exception as e:
        print(f"Failed to log to Wandb: {e}")
    
    print("=== PPO Gradient Quality Evaluation Complete ===\n")
    
    return results 