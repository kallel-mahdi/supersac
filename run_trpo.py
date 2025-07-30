import argparse
import os
import random
import time
import copy

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.distributions.kl import kl_divergence
from jaxrl_m.utils import *
from jaxrl_m.wandb import setup_wandb
from jaxrl_m.rollout import rollout_policy_ppo
import wandb
from collections import deque

os.environ["WANDB_API_KEY"]="28996bd59f1ba2c5a8c3f2cc23d8673c327ae230"


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo_name', type=str, default='trpo', help='the name of the RL algorithm')
    parser.add_argument("--exp_name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=21,
        help="seed of the experiment")
    parser.add_argument("--torch_deterministic", type=str2bool, default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=str2bool, default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=str2bool, default=True, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--project_name", type=str, default="TRPO",
        help="the wandb's project name")
    parser.add_argument("--wandb_entity", type=str, default='jyh',
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture_video", type=str2bool, default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--env_name", type=str, default="Hopper-v5",
        help="the id of the environment")
    parser.add_argument("--max_steps", type=int, default=None,
        help="total timesteps of the experiments")
    parser.add_argument("--learning_rate", type=float, default=3e-4,
        help="the learning rate of the critic optimizer")
    parser.add_argument("--num_envs", type=int, default=1,
        help="the number of parallel game environments")
    parser.add_argument("--num_steps", type=int, default=5120,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal_lr", type=str2bool, default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae_lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num_minibatches", type=int, default=1,
        help="the number of mini-batches")
    parser.add_argument("--update_epochs", type=int, default=1,
        help="the K epochs to update the policy")
    parser.add_argument("--norm_adv", type=str2bool, default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip_coef", type=float, default=0.2,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip_vloss", type=str2bool, default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--vf_coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max_grad_norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target_kl", type=float, default=0.01,
        help="the target KL divergence threshold")
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return args


def make_env(env_id, idx, capture_video, run_name, gamma, evaluation=False):
    def thunk():
        
        env, eval_env = create_environments(args.env_name)
        # Note: create_environments already applies RecordEpisodeStatistics, so we don't add it again
        env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10), env.observation_space)
        if not evaluation:
            env = gym.wrappers.NormalizeReward(env, gamma=gamma)
            env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env

    return thunk

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

def _fisher_vector_product(actor, obs, x, cg_damping=0.1):
    x.detach()
    pi_new = actor(obs)
    with torch.no_grad():
        pi_old = actor(obs)  
    kl = kl_divergence(pi_old, pi_new).mean()
    kl_grads = torch.autograd.grad(kl, tuple(actor.parameters()), create_graph=True)
    flat_kl_grad = torch.cat([grad.view(-1) for grad in kl_grads])
    kl_grad_p = (flat_kl_grad * x).sum()
    kl_hessian_p = torch.autograd.grad(kl_grad_p, tuple(actor.parameters()))
    flat_kl_hessian_p = torch.cat([grad.contiguous().view(-1) for grad in kl_hessian_p])

    # tricks to stablize
    # see https://www2.maths.lth.se/matematiklth/vision/publdb/reports/pdf/byrod-eccv-10.pdf
    return flat_kl_hessian_p + cg_damping * x 


# Refer to https://en.wikipedia.org/wiki/Conjugate_gradient_method for more details
def conjugate_gradient(actor, obs, b, cg_iters, cg_residual_tol=1e-10): 
    '''
        Given a linear system Ax = b and an initial guess x0=0, the conjugate gradient method solves the problem
        Ax = b for x without computing A explicitly. Instead, only the computation of the matrix-vector product Ax is needed.
        In TRPO, A is the Fisher information matrix F (the second derivates of KL divergence) and b is the gradient of the loss function.
    '''
    x = torch.zeros_like(b) 
    r = b.clone()
    p = b.clone()
    rdotr = torch.dot(r, r)
    for _ in range(cg_iters):
        _Ax = _fisher_vector_product(actor, obs, p)
        alpha = rdotr / torch.dot(p, _Ax)
        x += alpha * p
        r -= alpha * _Ax
        new_rdotr = torch.dot(r, r)
        betta = new_rdotr / rdotr
        p = r + betta * p
        rdotr = new_rdotr
        if rdotr < cg_residual_tol:
            break
    return x


def update_model(model, new_params):
    index = 0
    for params in model.parameters():
        params_length = len(params.view(-1))
        new_param = new_params[index: index + params_length]
        new_param = new_param.view(params.size())
        params.data.copy_(new_param)
        index += params_length

def flat_params(model):
    params = []
    for param in model.parameters():
        params.append(param.view(-1))
    flat_params = torch.cat(params)
    return flat_params.detach()


class Actor(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, np.prod(envs.single_action_space.shape)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))

    def get_action(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1)

    def forward(self, x):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        return Normal(action_mean, action_std)


class Critic(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )

    def get_value(self, x):
        return self.critic(x)


if __name__ == "__main__":
    args = parse_args()
    
    # Set max_steps using environment-specific default if not provided
    max_steps = get_max_steps_for_env(args.env_name) if args.max_steps is None else args.max_steps
    
    run_name = f"{args.env_name}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        wandb_config = {
            'project': args.project_name,
            'name': None,
            'hyperparam_dict': args.__dict__,
        }
        wandb_run = setup_wandb(**wandb_config)

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_name, i, args.capture_video, run_name, args.gamma) for i in range(args.num_envs)]
    )
    
    eval_env = gym.vector.SyncVectorEnv(
        [make_env(args.env_name, i, args.capture_video, run_name, args.gamma, evaluation=True) for i in range(args.num_envs)]
    )
    
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    actor = Actor(envs).to(device)
    critic = Critic(envs).to(device)
    optimizer_critic = optim.Adam(critic.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = max_steps // args.batch_size
    video_filenames = set()
    
    # Logging setup
    log_interval = 20000
    unlogged_steps, total_steps = 0, 0
    last_returns = deque([], maxlen=2)

    for update in range(1, num_updates + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer_critic.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            unlogged_steps += 1
            total_steps += 1
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _ = actor.get_action(next_obs)
                values[step] = critic.get_value(next_obs).flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminated, truncated, infos = envs.step(action.cpu().numpy())
            done = np.logical_or(terminated, truncated)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)

            # Log episode returns
            if "episode" in infos.keys():
                wandb.log({"training/episodic_return": infos["episode"]["r"]}, step=global_step, commit=False)
                last_returns.append(infos["episode"]["r"])

        # bootstrap value if not done
        with torch.no_grad():
            next_value = critic.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        actor_lrs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy = actor.get_action(b_obs[mb_inds], b_actions[mb_inds])
                newvalue = critic.get_value(b_obs[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # 1. Train critic
                newvalue = newvalue.view(-1)
                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
                optimizer_critic.zero_grad()
                v_loss.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), args.max_grad_norm)
                optimizer_critic.step()

                # 2. Get the direction of the actor gradient
                pg_loss = (mb_advantages * ratio).mean() # note 
                actor.zero_grad()
                pg_grad = torch.autograd.grad(pg_loss, tuple(actor.parameters()))
                flat_pg_graid = torch.cat([grad.view(-1) for grad in pg_grad])
                step_dir = conjugate_gradient(actor, b_obs[mb_inds], flat_pg_graid, cg_iters=10)
                step_size = torch.sqrt(2 * args.target_kl / (torch.dot(step_dir, _fisher_vector_product(actor, b_obs[mb_inds], step_dir)) + 1e-8))
                step_dir *= step_size

                # 3. Backtracking line search for the learning rate of actor
                old_actor = copy.deepcopy(actor)
                params = torch.cat([param.view(-1) for param in actor.parameters()])
                expected_improve = (flat_pg_graid * step_dir).sum().item()
                fraction = 1.0
                for i in range(10):
                    new_params = params + fraction * step_dir 
                    update_model(actor, new_params)
                    _, newlogprob, entropy = actor.get_action(b_obs[mb_inds], b_actions[mb_inds])
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()
                    new_pg_loss = (advantages[mb_inds] * ratio).mean()
                    loss_improve = new_pg_loss - pg_loss
                    expected_improve *= fraction
                    kl = kl_divergence(old_actor(b_obs[mb_inds]), actor(b_obs[mb_inds])).mean()
                    if kl < args.target_kl and  loss_improve > 0:
                        break
                    fraction *= 0.5
                else:
                    update_model(actor, params)
                    fraction = 0.0
                    # print("Not update")
                actor_lrs.append(fraction) 

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # Log metrics
        wandb.log({
            "charts/critic_learning_rate": optimizer_critic.param_groups[0]["lr"],
            "charts/actor_learning_rate": np.mean(actor_lrs),
            "losses/value_loss": v_loss.item(),
            "losses/policy_loss": pg_loss.item(),
            "losses/old_approx_kl": old_approx_kl.item(),
            "losses/approx_kl": approx_kl.item(),
            "losses/clipfrac": np.mean(clipfracs),
            "losses/explained_variance": explained_var,
            "charts/SPS": int(global_step / (time.time() - start_time))
        }, step=global_step, commit=False)

        # Evaluation rollouts
        if unlogged_steps >= log_interval:
            unlogged_steps = 0
            
            # Copy observation normalization from training env to eval env
            if hasattr(envs.envs[0], 'env') and hasattr(envs.envs[0].env, 'obs_rms'):
                eval_env.envs[0].env.obs_rms = envs.envs[0].env.obs_rms 

            # Add simple adapter for TRPO actor to work with PPO rollout function
            class ActorAdapter:
                def __init__(self, actor, device):
                    self.actor = actor
                    self.device = device
                
                def deterministic_action(self, x, action=None):
                    x = torch.Tensor(x).to(self.device)
                    action_mean = self.actor.actor_mean(x)
                    return action_mean.cpu().detach().numpy()
                                
            actor_adapter = ActorAdapter(actor, device)
            eval_env.envs[0].env.obs_rms = envs.envs[0].env.env.env.obs_rms 
            undisc_policy_return = rollout_policy_ppo(
                actor_adapter, env=eval_env,
                num_rollouts=10,
                discount=args.gamma, max_length=1000
            )
            
            eval_metrics = {"undisc_policy_return": undisc_policy_return}
            eval_metrics = {f'evaluation/{k}': v for k, v in eval_metrics.items()}
            
            wandb.log(eval_metrics, step=int(total_steps), commit=True)

        print("SPS:", int(global_step / (time.time() - start_time)))

    envs.close()
    if args.track:
        wandb_run.finish()

