import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import wandb
import os
import numpy as np

# --- Configuration ---
ENTITY = "mahdikallel"
PROJECT = "BASELINE_FINAL"
PROJECT_PATH = f"{ENTITY}/{PROJECT}"

# --- Test Mode ---
TEST_MODE = True

# --- Styling ---
sns.set_theme(style="ticks", rc={"font.family": "serif"})
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 16,
    'lines.linewidth': 2.5,
})

# --- Algorithm Colors ---
ALGORITHM_COLORS = {
    "PPO+": "#2E8B8B",
    "PPO": "#8E44AD", 
    "PPO(ours)": "#FF8C42",
    "SAC": "#E74C3C",
    "TRPO": "#3498DB"
}

# --- Algorithm Configurations ---
ALGORITHMS = {
    "PPO+": {"algo_name": "superppo", "gamma": 0.99, "on_policy_data": False, "temperature": 1., "buffer_size": 50000, "use_layer_norm": True, "clipping_ratio": 0.25, "num_critics": 2, "bound_actions": True, "gae_lambda": 0.5, "min_target": False},
    "PPO": {"algo_name": "ppo", "hidden_dims": 64},
    "PPO(ours)": {"algo_name": "ppo", "hidden_dims":256},
    "SAC": {"algo_name": "sac"},
    #"TRPO": {"algo_name": "trpo"}
}

# --- Environment Settings ---
ENVS = ["InvertedDoublePendulum-v5", "Hopper-v5", "Walker2d-v5", "HalfCheetah-v5", "Ant-v5", "Humanoid-v5"]
MAX_STEPS = [2e5, 1e6, 1e6, 1e6, 1e6, 5e6]

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
                        window = 1 if env == "InvertedDoublePendulum-v5" else 5
                        df['return'] = df['evaluation/undisc_policy_return'].rolling(window).mean()
                        df['step'] = (df['_step'] // 10000) * 10000
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
    
    # Create plot
    fig, axes = plt.subplots(2, 3, figsize=(24, 12))
    fig.patch.set_facecolor('white')
    
    for ax, env, max_step in zip(axes.flatten(), ENVS, MAX_STEPS):
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
            std = grouped.std() / np.sqrt(grouped.count())
            
            # Plot
            color = ALGORITHM_COLORS.get(algo, 'gray')
            ax.plot(x, mean, label=algo.upper(), color=color, linewidth=2.5)
            ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.3)
        
        # Style subplot
        ax.set_xlabel('Million Steps')
        ax.set_ylabel('Policy Return')
        ax.set_title(env)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    # Add shared legend on top (like original)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, 
                  loc='upper center', 
                  ncol=len(ALGORITHMS), 
                  bbox_to_anchor=(0.5, 1.05),
                  fontsize=16,
                  frameon=True,
                  fancybox=True,
                  shadow=True,
                  columnspacing=1.5,
                  handletextpad=0.8,
                  handlelength=2.0)
    
    # Layout and save (adjusted for legend space)
    plt.subplots_adjust(top=0.85, left=0.06, right=0.94, bottom=0.12, wspace=0.25, hspace=0.3)
    
    os.makedirs("./plots/benchmarks", exist_ok=True)
    plt.savefig("./plots/benchmarks/benchmark.pdf", bbox_inches="tight", dpi=300)
    plt.savefig("./plots/benchmarks/benchmark.png", bbox_inches="tight", dpi=300)
    plt.show()
    
    print("Benchmark plot saved!")

if __name__ == "__main__":
    print(f"Running benchmark plot - {'TEST MODE' if TEST_MODE else 'REAL DATA'}")
    create_benchmark_plot()