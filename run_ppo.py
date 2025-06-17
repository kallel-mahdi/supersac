# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
#import tyro
from torch.distributions.normal import Normal
#from torch.utils.tensorboard import SummaryWriter
from jaxrl_m.wandb import setup_wandb
from jaxrl_m.rollout import *
import os
import argparse
#import envpool
import wandb
import numpy as jnp
from collections import deque
import copy
from jaxrl_m.dmc import DMCGym

os.environ["WANDB_API_KEY"]="28996bd59f1ba2c5a8c3f2cc23d8673c327ae230"
np.seterr(all='raise')

def one(env, name):
    """
    If this env does not have the attribute, then we try to 
    recursively access that attribute from inner envs.
    """
    while not hasattr(env, name):
        if hasattr(env, 'env'): # while the env is still wrapped,
            env = env.env
        else: # reached the innermost env and still didn't find it.
            raise AttributeError(f'{env} has no attribute {name}.')
    return getattr(env, name) # reached if env **has** attribute name.


def two(env, name):
    """
    If this env does not have the attribute, then we try to 
    recursively access that attribute from inner envs.
    """
    while not hasattr(env, name):
        if hasattr(env, 'env'): # while the env is still wrapped,
            env = env.env
        else: # reached the innermost env and still didn't find it.
            raise AttributeError(f'{env} has no attribute {name}.')
    return setattr(env, name) # reached if env **has** attribute name.

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    

parser = argparse.ArgumentParser(description='PPO Arguments')
parser.add_argument('--algo_name', type=str, default='ppo', help='the name of the RL algorithm')
parser.add_argument('--seed', type=int, default=1, help='seed of the experiment')
parser.add_argument('--torch_deterministic', default=True,action='store_true', help='if toggled, `torch.backends.cudnn.deterministic=False`')
parser.add_argument('--cuda', action='store_true',default=True, help='if toggled, cuda will be enabled by default')
parser.add_argument('--track', action='store_true', default=True,help='if toggled, this experiment will be tracked with Weights and Biases')
parser.add_argument('--project_name', type=str, default='cleanRL', help='the wandb\'s project name')
parser.add_argument('--capture_video', action='store_true',default=False, help='whether to capture videos of the agent performances (check out `videos` folder)')
parser.add_argument('--save_model', action='store_true',default=False, help='whether to save model into the `runs/{run_name}` folder')
parser.add_argument('--upload_model', action='store_true',default=False, help='whether to upload the saved model to huggingface')
parser.add_argument('--hf_entity', type=str, default='', help='the user or org name of the model repository from the Hugging Face Hub')
parser.add_argument('--env_name', type=str, default='Humanoid-v5', help='the id of the environment')
parser.add_argument('--max_steps', type=int, default=1000000, help='total timesteps of the experiments')
parser.add_argument('--learning_rate', type=float, default=3e-4, help='the learning rate of the optimizer')
parser.add_argument('--num_envs', type=int, default=1, help='the number of parallel game environments')
parser.add_argument('--num_steps', type=int, default=5120, help='the number of steps to run in each environment per policy rollout')


parser.add_argument('--anneal_lr', default=True,type=str2bool, help='Toggle learning rate annealing for policy and value networks')
parser.add_argument('--normalize_reward',type=str2bool, default=True)
parser.add_argument('--normalize_observation',type=str2bool, default=True)
parser.add_argument('--full_batch',type=str2bool, default=False)
parser.add_argument('--gamma', type=float, default=0.99, help='the discount factor gamma')
parser.add_argument('--gae_lambda', type=float, default=0.95, help='the lambda for the general advantage estimation')
parser.add_argument('--num_minibatches', type=int, default=32, help='the number of mini-batches')
parser.add_argument('--update_epochs', type=int, default=10, help='the K epochs to update the policy')
parser.add_argument('--norm_adv',default=True,type=str2bool, help='Toggles advantages normalization')#####
parser.add_argument('--clip_coef', type=float, default=0.2, help='the surrogate clipping coefficient')
parser.add_argument('--clip_vloss', default=True,type=str2bool, help='Toggles whether or not to use a clipped loss for the value function, as per the paper.')
parser.add_argument('--ent_coef', type=float, default=0.0, help='coefficient of the entropy')
parser.add_argument('--vf_coef', type=float, default=0.5, help='coefficient of the value function')
parser.add_argument('--max_grad_norm', type=float, default=0.5, help='the maximum norm for the gradient clipping')
parser.add_argument('--target_kl', type=float, default=None, help='the target KL divergence threshold')
parser.add_argument('--hidden_dims', type=int, default=256, help='the hidden dimensions of the network')
parser.add_argument('--use_layer_norm',type=str2bool, default=True, help='Toggle to use layer norm in the policy/value networks')

args = parser.parse_args()
args.batch_size = int(args.num_envs * args.num_steps)
args.minibatch_size = int(args.batch_size // args.num_minibatches)
args.num_iterations = args.max_steps // args.batch_size

if args.env_name in ["walk","stand","trot","run","Humanoid-v5"]: args.max_steps = 5_000_000

args = parser.parse_args()

print(args.full_batch)





def make_env(env_name, idx, capture_video, run_name, gamma,evaluation=False):
    def thunk():
        
        
        if args.env_name in ["walk","stand","trot","run"]:
            env = DMCGym("dog",args.env_name)
        
        else : env = gym.make(env_name,max_episode_steps=1000)
        
        
        env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        #env = gym.wrappers.NormalizeObservation(env)
        
        if args.normalize_observation:
            env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10),env.observation_space)
        if args.normalize_reward and not evaluation:
            env = gym.wrappers.NormalizeReward(env, gamma=gamma)
            env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs, hidden_size, use_layer_norm):
        super().__init__()
        input_dim = int(np.prod(envs.single_observation_space.shape))
        action_dim = int(np.prod(envs.single_action_space.shape))
        norm_layer = lambda dim: nn.LayerNorm(dim) if use_layer_norm else nn.Identity()
        
        self.critic = nn.Sequential(
            layer_init(nn.Linear(input_dim, hidden_size)),
            norm_layer(hidden_size),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            norm_layer(hidden_size),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(input_dim, hidden_size)),
            norm_layer(hidden_size),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            norm_layer(hidden_size),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, action_dim), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)

    def sample_actions(self, x, *args, **kwargs):
        x = torch.Tensor(x).to(device)
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        action = probs.sample()
        return action.cpu().detach().numpy()

    def deterministic_action(self, x, action=None):
        x = torch.Tensor(x).to(device)
        action_mean = self.actor_mean(x)
        return action_mean.cpu().detach().numpy()

if __name__ == "__main__":
    
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.max_steps // args.batch_size
    run_name = f"{args.env_name}__{args.seed}__{int(time.time())}"
    if args.track:
        wandb_config = {
                'project': args.project_name,
                'name':None,
                'hyperparam_dict':args.__dict__,
                }
        wandb_run = setup_wandb(**wandb_config)
    

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    last_returns = deque([], maxlen=2)
    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_name, i, args.capture_video, run_name, args.gamma) for i in range(args.num_envs)]
    )
    
    eval_env = gym.vector.SyncVectorEnv(
        [make_env(args.env_name, i, args.capture_video, run_name, args.gamma,evaluation=True) for i in range(args.num_envs)]
    )
    
    #envs = make_env(args.env_name,0,args.capture_video,run_name,args.gamma)()
    #eval_env = make_env(args.env_name,0,args.capture_video,run_name,args.gamma,evaluation=True)()
    
    
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"
    
    
    #eval_env = envpool.make(args.env_name, env_type="gymnasium", num_envs=10)
    log_interval = 20000
    unlogged_steps,total_steps = 0,0
    agent = Agent(envs,args.hidden_dims,args.use_layer_norm).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5,betas=(0.,0.999))

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

    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            unlogged_steps += 1
            total_steps +=1
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)
            
            
            if "episode" in infos.keys():
                wandb.log({"training/episodic_return": infos["episode"]["r"]},step=global_step,commit=False)
                last_returns.append(infos["episode"]["r"])

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
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
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size

                if args.full_batch : mb_inds = b_inds
                else : mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()
                wandb.log({"ratio": ratio.mean(),"min_ratio":ratio.min(),"max_ratio":ratio.max()},step=global_step,commit=False)

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        
        if unlogged_steps >= log_interval:
    
            unlogged_steps = 0
           
            #eval_env = copy.deepcopy(envs)
            #eval_env.ret_rms = envs.ret_rms
            #print(envs.envs[0].env.env.obs_rms)
            #print(eval_env.envs[0].env.env.obs_rms)
            
            eval_env.envs[0].env.obs_rms = envs.envs[0].env.env.env.obs_rms 

            

            print("heeeeeeeeeeeeeeeeere")
            undisc_policy_return = rollout_policy_ppo(
                                                                    agent,env = eval_env,
                                                                    num_rollouts=10,
                                                                    discount = args.gamma,max_length=1000)
            print("heeeeeeeeeeeeeeeeere2")
            eval_metrics = {"undisc_policy_return": undisc_policy_return}

            eval_metrics = {f'evaluation/{k}': v for k, v in eval_metrics.items()}
            
            wandb.log(eval_metrics, step=int(total_steps),commit=True)
                    

      
    envs.close()
    wandb_run.finish()