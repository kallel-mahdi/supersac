import pandas as pd
import wandb
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import centralized styling and utilities
from style import (
    create_publication_ready_figure,
    set_axis_labels,
    save_publication_figure
)

# --- Configuration ---
ENTITY = "mahdikallel"
PROJECT = "GAE_LAMBDA2"
PROJECT_PATH = f"{ENTITY}/{PROJECT}"
TEST_MODE = False
N_RUNS = 5


DEFAULT_PPO_CONFIG = {
    "algo_name": "ppo",
    "gamma": 0.99,
    "norm_adv": True,
    "normalize_observation": True,
    "normalize_reward": True,
    "anneal_lr": True,
}

DEFAULT_PPO_PLUS_CONFIG = {
    "algo_name": "superppo",
    "gamma": 0.99,
    "on_policy_data": False,
}

# --- Algorithm Configurations ---
lambdas = [0.95, 0.7, 0.5, 0.0]
ALGO_CONFIGS = {
    "Regular PPO": {
        f"λ={l}": {**DEFAULT_PPO_CONFIG, "gae_lambda": l} for l in lambdas
    },
    "PPO+": {
        f"λ={l}": {**DEFAULT_PPO_PLUS_CONFIG, "gae_lambda": l} for l in lambdas
    }
}


ENV_CONFIG = {
    "env_names": ["Hopper-v5", "Walker2d-v5", "HalfCheetah-v5"],
    "max_steps": [1e6, 1e6, 1e6]
}

# --- Data Fetching ---
def fetch_data_for_group(algo_group_config, env_names, n_runs):
    """Fetch data from W&B for a group of algorithms, limited to the most recent n_runs."""
    api = wandb.Api()
    
    all_data = {}
    for algo_name, config in algo_group_config.items():
        algo_runs_data = []
        for env_name in env_names:
            filters = {"config.env_name": env_name, **{f"config.{k}": v for k, v in config.items()}}
            
            try:
                runs = api.runs(PROJECT_PATH, filters=filters)
                for i, run in enumerate(runs):
                    if i >= n_runs:
                        break
                    
                    history = run.history(keys=['evaluation/undisc_policy_return', '_step'], pandas=True).dropna()
                    if not history.empty:
                        df = history.copy()
                        df['average return'] = df['evaluation/undisc_policy_return'].rolling(window=5).mean()
                        df['step'] = np.round(df['_step'] / 10000) * 10000
                        df['env'] = env_name
                        df['algo'] = algo_name
                        algo_runs_data.append(df[['step', 'average return', 'env', 'algo']])
            except Exception as e:
                print(f"Error fetching data for {algo_name} in {env_name}: {e}")
                continue

        if algo_runs_data:
            all_data[algo_name] = pd.concat(algo_runs_data, ignore_index=True)
            
    return all_data

# --- Plotting ---
def plot_ablation_grid():
    """Create a 2x3 grid of ablation plots using default styling."""
    os.environ["WANDB_API_KEY"] = "28996bd59f1ba2c5a8c3f2cc23d8673c327ae230"
    
    fig, axes = create_publication_ready_figure(nrows=2, ncols=3)
    
    env_names = ENV_CONFIG["env_names"]
    max_steps = ENV_CONFIG["max_steps"]
    palette = sns.color_palette()

    for row_idx, (group_title, algo_group) in enumerate(ALGO_CONFIGS.items()):
        data_group = fetch_data_for_group(algo_group, env_names, n_runs=N_RUNS)
        
        for col_idx, (env_name, max_step) in enumerate(zip(env_names, max_steps)):
            ax_idx = row_idx * 3 + col_idx
            ax = axes[ax_idx]
            
            for i, (algo, data) in enumerate(data_group.items()):
                env_data = data[data['env'] == env_name]
                env_data = env_data[env_data['step'] < max_step]

                if not env_data.empty:
                    grouped = env_data.groupby('step')['average return']
                    x = grouped.mean().index / 1e6
                    mean = grouped.mean()
                    std_err = grouped.std() / np.sqrt(grouped.count())
                    color = palette[i % len(palette)]
                    
                    ax.plot(x, mean, label=algo, color=color)
                    ax.fill_between(x, mean - std_err, mean + std_err, color=color, alpha=0.2)

            set_axis_labels(ax, 'Million Steps', 'Policy Return', title=env_name)
            ax.grid(True, alpha=0.3, linestyle='--')
            
        fig.text(0.02, 0.75 if row_idx == 0 else 0.25, group_title, rotation=90, 
                 fontsize=22, va='center', ha='center')
        
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='upper center', ncol=len(ALGO_CONFIGS["Regular PPO"]),
                   bbox_to_anchor=(0.5, 0.98))

    plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.92])
    save_publication_figure(fig, "/home/mahdi/Desktop/supersac/notebooks/plots/ablations_tmlr/gae_lambda_ablations_final")
    plt.show()
    print("Ablation plot with single shared legend saved!")

if __name__ == "__main__":
    plot_ablation_grid()