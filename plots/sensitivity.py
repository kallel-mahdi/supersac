import matplotlib.pyplot as plt
import pandas as pd
import wandb
import os
import numpy as np

# Import centralized styling
from style import (
    ALGORITHM_COLORS, 
    create_publication_ready_figure,
    set_axis_labels,
    setup_shared_legend, 
    save_publication_figure
)

# --- Configuration ---
ENTITY = "mahdikallel"
PROJECT = "GAE_LAMBDA2"
PROJECT_PATH = f"{ENTITY}/{PROJECT}"

# --- Test Mode ---
TEST_MODE = False

BASELINE_ALGO_NAME = "PPO (λ=0.95)"
DEFAULT_CONFIG = {"algo_name": "ppo", "hidden_dims": 64,"gae_lambda": 0.95,"normalize_observation": True,
            "normalize_reward": True,"norm_adv": True,"anneal_lr":True}

ABLATIONS = {
    "λ=0": {"gae_lambda": 0.0},
    "λ=0.5": {"gae_lambda": 0.5},
    "λ=0.7": {"gae_lambda": 0.7},
    "-Obs norm": {"normalize_observation": False},
    "-Reward norm": {"normalize_reward": False},
    # "-Adv norm": {"norm_adv": False},
    # "-LR anneal": {"anneal_lr": False},
}

ALGORITHMS = {BASELINE_ALGO_NAME: DEFAULT_CONFIG}
for name, changes in ABLATIONS.items():
    new_config = DEFAULT_CONFIG.copy()
    new_config.update(changes)
    ALGORITHMS[name] = new_config




# # --- Environment Settings ---
ENVS = ["Hopper-v5", "Walker2d-v5"]
MAX_STEPS = [1e6, 1e6]

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
        print(f"✅ Successfully connected to W&B API")
    except Exception as e:
        print(f"❌ Failed to connect to W&B API: {e}")
        return {}
    
    data = {}
    print(f"\n Searching for runs in project: {PROJECT_PATH}")
    print(f"📊 Looking for {len(ALGORITHMS)} algorithms across {len(ENVS)} environments")
    
    for algo, config in ALGORITHMS.items():
        print(f"\n--- Algorithm: {algo} ---")
        print(f"   Config: {config}")
        algo_data = []
        
        for env in ENVS:
            print(f"   📍 Environment: {env}")
            
            # Build filter
            filters = {"config.env_name": env}
            filters.update({f"config.{k}": v for k, v in config.items()})
            print(f"   🔍 Filters: {filters}")
            
            # Get runs
            try:
                runs = api.runs(PROJECT_PATH, filters=filters)
                num_runs = len(list(runs))  # Convert to list to count
                print(f"   📈 Found {num_runs} runs for {algo} in {env}")
                
                if num_runs == 0:
                    print(f"   ⚠️  NO RUNS FOUND for {algo} in {env}")
                    print(f"   💡 Check if runs exist with these exact config values:")
                    for key, value in config.items():
                        print(f"      config.{key} = {value}")
                    continue
                
                successful_runs = 0
                max_runs = 5
                for run in runs:
                    if successful_runs >= max_runs:
                        print(f"   ⏹️  Reached maximum of {max_runs} valid runs for {algo} in {env}")
                        break
                        
                    try:
                        print(f"   📊 Processing run: {run.name} (ID: {run.id})")
                        history = run.history(keys=['evaluation/undisc_policy_return', '_step'], pandas=True)
                        
                        if history.empty:
                            print(f"   ⚠️  Run {run.name} has empty history")
                            continue
                        
                        # Process data
                        df = history.dropna()
                        if df.empty:
                            print(f"   ⚠️  Run {run.name} has no valid data after dropna()")
                            continue
                        
                        if env == "InvertedDoublePendulum-v5":
                            window = 1
                        elif "Humanoid" in env or "dog" in env:
                            window = 5
                        else:
                            window = 5
                        
                        df['return'] = df['evaluation/undisc_policy_return'].rolling(window).mean()
                        df['step'] = (df['_step'] // 10000) * 10000
                        df['env'] = env
                        df['algo'] = algo
                        
                        processed_df = df[['step', 'return', 'env', 'algo']]
                        if not processed_df.empty:
                            algo_data.append(processed_df)
                            successful_runs += 1
                            print(f"   ✅ Successfully processed run {run.name} ({len(processed_df)} data points)")
                        else:
                            print(f"   ⚠️  Run {run.name} has no valid data after processing")
                            
                    except Exception as e:
                        print(f"   ❌ Error processing run {run.name}: {e}")
                        continue
                
                if successful_runs < max_runs:
                    print(f"   📊 Successfully processed {successful_runs}/{num_runs} runs for {algo} in {env} (less than maximum of {max_runs})")
                else:
                    print(f"   📊 Successfully processed {successful_runs}/{num_runs} runs for {algo} in {env} (reached maximum of {max_runs})")
                
            except Exception as e:
                print(f"   ❌ Error fetching runs for {algo} in {env}: {e}")
                continue
        
        if algo_data:
            data[algo] = pd.concat(algo_data, ignore_index=True)
            print(f"   ✅ Added {algo} to dataset with {len(data[algo])} total data points")
        else:
            print(f"   ❌ No valid data found for {algo}")
    
    print(f"\n📊 Final dataset summary:")
    for algo in ALGORITHMS:
        if algo in data:
            print(f"   ✅ {algo}: {len(data[algo])} data points")
        else:
            print(f"   ❌ {algo}: NO DATA")
    
    return data

def create_benchmark_plot():
    """Create the benchmark plot."""
    os.environ["WANDB_API_KEY"] = "28996bd59f1ba2c5a8c3f2cc23d8673c327ae230"
    
    # Get data
    data = generate_mock_data() if TEST_MODE else fetch_data()
    if not data:
        print("❌ No data available")
        return
    
    print(f"\n Creating plot with {len(data)} algorithms")
    
    # Create a single figure with 2 subplots - one for each environment
    fig, axes = create_publication_ready_figure(figsize=(16, 6), nrows=1, ncols=2)
    
    for ax, (env, max_step) in zip(axes, zip(ENVS, MAX_STEPS)):
        print(f"\n📊 Plotting data for {env}")
        plotted_algorithms = []
        
        for algo in ALGORITHMS:
            if algo not in data:
                print(f"   ❌ No data for {algo}")
                continue
            
            # Filter and aggregate data for this environment
            env_data = data[algo][data[algo]['env'] == env]
            env_data = env_data[env_data['step'] < max_step]
            
            if env_data.empty:
                print(f"   ⚠️  No data for {algo} in {env} (after filtering)")
                continue
            
            grouped = env_data.groupby('step')['return']
            x = grouped.mean().index / 1e6
            mean = grouped.mean()
            std = grouped.std() / np.sqrt(grouped.count())
            
            # Plot
            ax.plot(x, mean, label=algo.upper(), linewidth=2.5)
            ax.fill_between(x, mean - std, mean + std, alpha=0.3)
            plotted_algorithms.append(algo)
            print(f"   ✅ Plotted {algo} with {len(x)} data points")
        
        print(f"   📊 Total algorithms plotted for {env}: {len(plotted_algorithms)}")
        print(f"   📋 Algorithms: {plotted_algorithms}")
        
        # Set axis labels using utility function
        set_axis_labels(ax, 'Million Steps', 'Policy Return', env)
        
        # Enable grid for the plot
        ax.grid(True, alpha=0.3, linestyle='--', color='#BDC3C7', zorder=0)
    
    # Add shared legend for both subplots
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:  # Only create legend if there are handles
        print(f"\n📋 Creating legend with {len(labels)} items: {labels}")
        setup_shared_legend(fig, handles, labels, ncol=len(ALGORITHMS), font_size=19, bbox_y=1.05)
    else:
        print(f"\n⚠️  No legend items found!")
    
    # Adjust layout for the two subplots
    plt.subplots_adjust(
        top=0.85,      # Space for legend
        left=0.06,     
        right=0.94,    
        bottom=0.15,    
        wspace=0.25,    # Space between subplots
    )
    
    save_publication_figure(fig, "./plots/benchmarks/sensitivity")
    plt.show()
    
    print("✅ Benchmark plot with 2 subplots saved!")

if __name__ == "__main__":
    print(f"Running benchmark plot - {'TEST MODE' if TEST_MODE else 'REAL DATA'}")
    create_benchmark_plot()