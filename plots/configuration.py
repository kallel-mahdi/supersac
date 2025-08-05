import matplotlib.pyplot as plt
import pandas as pd
import wandb
import os
import numpy as np

# Import centralized styling
from style import (
    create_publication_ready_figure,
    set_axis_labels,
    setup_shared_legend, 
    save_publication_figure
)

# --- Configuration ---
ENTITY = "mahdikallel"
PROJECT = "AAAI_ABLATIONS_INTER"
PROJECT_PATH = f"{ENTITY}/{PROJECT}"

# --- Test Mode ---
TEST_MODE = False

# --- Algorithm Configurations ---
ALGORITHMS = {
    
    "PPO+": {"algo_name": "superppo", "gamma": 0.99, "on_policy_data": False, "temperature": 1., "buffer_size": 50000, 
             "use_layer_norm": True, "clipping_ratio": 0.25, "num_critics": 2, "bound_actions": True, "gae_lambda": 0.5, 
             "min_target": False, "intertwine_updates": True},
    "+Min target": {"algo_name": "superppo", "gamma": 0.99, "on_policy_data": False, "temperature": 1., "buffer_size": 50000, 
             "use_layer_norm": True, "clipping_ratio": 0.25, "num_critics": 2, "bound_actions": True, "gae_lambda": 0.5, 
             "min_target": True, "intertwine_updates": True},

    
    "-Off-policy data": {"algo_name": "superppo", "gamma": 0.99, "on_policy_data": True, "temperature": 1., "buffer_size": 50000, 
             "use_layer_norm": True, "clipping_ratio": 0.25, "num_critics": 2, "bound_actions": True, "gae_lambda": 0.5, 
             "min_target": False, "intertwine_updates": True},
    
        "-LayerNorm": {"algo_name": "superppo", "gamma": 0.99, "on_policy_data": False, "temperature": 1., "buffer_size": 50000, 
             "use_layer_norm": True, "clipping_ratio": 0.25, "num_critics": 2, "bound_actions": True, "gae_lambda": 0.5, 
             "min_target": False, "intertwine_updates": True},
    
     "-Bounded actions": {"algo_name": "superppo", "gamma": 0.99, "on_policy_data": False, "temperature": 1., "buffer_size": 50000, 
             "use_layer_norm": True, "clipping_ratio": 0.25, "num_critics": 2, "bound_actions": False, "gae_lambda": 0.5, 
             "min_target": False, "intertwine_updates": True},
      
       "-MaxEnt": {"algo_name": "superppo", "gamma": 0.99, "on_policy_data": False, "temperature": 0., "buffer_size": 50000, 
             "use_layer_norm": True, "clipping_ratio": 0.25, "num_critics": 2, "bound_actions": True, "gae_lambda": 0.5, 
             "min_target": False, "intertwine_updates": True},
}

# # --- Environment Settings ---
ENVS = ["InvertedDoublePendulum-v5", "Hopper-v5", "Walker2d-v5", "HalfCheetah-v5", "Ant-v5", "Humanoid-v5"]
MAX_STEPS = [2e5, 1e6, 1e6, 1e6, 1e6, 5e6]


# ENTITY = "mahdikallel"
# PROJECT = "DMC_DOG"
# PROJECT_PATH = f"{ENTITY}/{PROJECT}"

# ALGORITHMS = {
#     "PPO+": {"algo_name": "superppo"},
#     "PPO": {"algo_name": "ppo", "hidden_dims": 64},
#     "PPO(ours)": {"algo_name": "ppo", "hidden_dims":256},
#     "SAC": {"algo_name": "sac"},
#     "TRPO": {"algo_name": "trpo"},
# }

# ENVS = ["dog-stand", "dog-walk", "dog-trot", "dog-run"]
# MAX_STEPS = [5e6,5e6, 5e6, 5e6]


def generate_mock_data():
    """Generate mock data for testing."""
    np.random.seed(42)
    data = {}
    for algo in ALGORITHMS:
        algo_data = []
        for env in ENVS:
            max_step = MAX_STEPS[ENVS.index(env)]
            steps = np.arange(0, max_step, 10000)
            # Different performance profiles
            base = 1000 if algo == "PPO+" else 800 if algo == "SAC" else 600
            returns = base * (1 - np.exp(-steps / (max_step * 0.3))) + np.random.normal(0, 50, len(steps))
            returns = np.maximum(returns, 0)
            
            df = pd.DataFrame({
                'step': steps,
                'return': returns,
                'env': env,
                'algo': algo
            })
            algo_data.append(df)
        data[algo] = pd.concat(algo_data, ignore_index=True)
    return data

def fetch_data():
    """Fetch data from W&B."""
    try:
        api = wandb.Api()
    except:
        return {}
    
    data = {}
    for algo, config in ALGORITHMS.items():
        algo_data = []
        for env in ENVS:
            # Build filter
            filters = {"config.env_name": env}
            filters.update({f"config.{k}": v for k, v in config.items()})
            
            # Get runs
            runs = api.runs(PROJECT_PATH, filters=filters)
            for run in runs:
                try:
                    history = run.history(keys=['evaluation/undisc_policy_return', '_step'], pandas=True)
                    if not history.empty:
                        # Process data
                        df = history.dropna()
                        if env == "InvertedDoublePendulum-v5":
                            window = 1
                        elif "Humanoid" in env or "dog" in env:
                            window = 5
                        else:
                            window = 5
                        df['return'] = df['evaluation/undisc_policy_return'].rolling(window).mean()
                        df['step'] = np.round(df['_step'] / 10000) * 10000
                        df['env'] = env
                        df['algo'] = algo
                        algo_data.append(df[['step', 'return', 'env', 'algo']])
                except:
                    continue
        
        if algo_data:
            data[algo] = pd.concat(algo_data, ignore_index=True)
    
    return data

def create_benchmark_plot():
    """Create the benchmark plot."""
    os.environ["WANDB_API_KEY"] = "28996bd59f1ba2c5a8c3f2cc23d8673c327ae230"
    
    # Get data
    data = generate_mock_data() if TEST_MODE else fetch_data()
    if not data:
        print("No data available")
        return
    
    # Create plot using utility function
    # Use 1 row for 4 environments, 2 rows for 6+ environments
    nrows = 1 if len(ENVS) <= 4 else 2
    ncols = 4 if len(ENVS) <= 4 else 3  # 4 columns for 4 plots, 3 columns for 6+ plots
    
    # Better figsize for 4 plots: adjust width and height
    if len(ENVS) <= 4:
        figsize = (24, 6)  # Wider figure for 4 plots side by side
    else:
        figsize = (24, 12)
    
    fig, axes = create_publication_ready_figure(figsize=figsize, nrows=nrows, ncols=ncols)
    
    
    for ax, env, max_step in zip(axes, ENVS, MAX_STEPS):
        for algo in ALGORITHMS:
            if algo not in data:
                continue
            
            # Filter and aggregate data
            env_data = data[algo][data[algo]['env'] == env]
            env_data = env_data[env_data['step'] < max_step]
            if env_data.empty:
                continue
            
            grouped = env_data.groupby('step')['return']
            x = grouped.mean().index / 1e6
            mean = grouped.mean()
            std = 2 * grouped.std() / np.sqrt(grouped.count())
            
            # Plot
            
            ax.plot(x, mean, label=algo.upper(), linewidth=2.5)
            ax.fill_between(x, mean - std, mean + std, alpha=0.3)
        
        # Set axis labels using utility function
        set_axis_labels(ax, 'Million Steps', 'Policy Return', env)
        
        # Enable grid for each subplot
        ax.grid(True, alpha=0.3, linestyle='--', color='#BDC3C7', zorder=0)
    
    # Add shared legend using the existing style function
    # Use the first axis for legend handles if only one row, otherwise use the middle row
    legend_ax = axes[0] if len(ENVS) <= 4 else axes[1]
    handles, labels = legend_ax.get_legend_handles_labels()
    if handles:  # Only create legend if there are handles
        # Use standard legend positioning with custom height for 4-plot case
        bbox_y = 1.08 if len(ENVS) <= 4 else 1.02  # Higher for 4 plots, standard for others
        setup_shared_legend(fig, handles, labels, ncol=len(ALGORITHMS), bbox_y=bbox_y)
    
    # Adjusted layout for better legend positioning
    if len(ENVS) <= 4:
        # For 4 plots: adjust spacing for wider layout
        plt.subplots_adjust(
            top=0.85,      # Space for legend
            left=0.04,     # Slightly less left margin for wider layout
            right=0.96,    # Slightly less right margin for wider layout
            bottom=0.15,    
            wspace=0.2,    # Slightly less space between plots for 4 plots
            hspace=0.3
        )
    else:
        # For 6+ plots: keep original spacing
        plt.subplots_adjust(
            top=0.88, 
            left=0.06, 
            right=0.94, 
            bottom=0.12, 
            wspace=0.25, 
            hspace=0.3
        )
    
    save_publication_figure(fig, "./plots/benchmarks/configuration")
    plt.show()
    
    print("Benchmark plot saved!")

if __name__ == "__main__":
    print(f"Running benchmark plot - {'TEST MODE' if TEST_MODE else 'REAL DATA'}")
    create_benchmark_plot()