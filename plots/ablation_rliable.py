import wandb
import numpy as np
import rliable.library as rly_lib
import rliable.metrics as rly
import rliable.plot_utils as rly_plot
import matplotlib.pyplot as plt
import os

# --- Configuration ---
ENTITY = "mahdikallel"
PROJECT = "AAAI_ABLATIONS_INTER"
PROJECT_PATH = f"{ENTITY}/{PROJECT}"
NUM_SEEDS = 10
LAST_POINTS_TO_AVERAGE = 10
BASELINE_ALGO_NAME = "PPO+"

# --- Test Mode Configuration ---
# Set TEST_MODE = True to use mock data for quick debugging
# Set TEST_MODE = False to fetch real data from Weights & Biases
TEST_MODE = False

# Note: fast_test parameter in generate_ablation_plot() controls environment count

# === Algorithm Name Mapping for Consistency ===
ALGORITHM_NAME_MAPPING = {
    "On-Policy Critic": "- Off-policy data",
    "Unbounded Actions": "- Bounded actions", 
    "No Entropy": "- Entropy",
    "No LayerNorm": "- LayerNorm",
    "Min Target": "+ Min target"
}

# === Step 1: Define the Default and Ablation Configs ===

# Base configuration for the full "PPO+" model
DEFAULT_CONFIG = {
    "algo_name": "superppo",
    "gamma": 0.99,
    "on_policy_data": False,
    "temperature": 1.0,
    "buffer_size": 50000,
    "use_layer_norm": True,
    "clipping_ratio": 0.25,
    "num_critics": 2,
    "bound_actions": True,
    "gae_lambda": 0.5,
    "min_target": False
}

# Define only the parameters that change for each ablation
ABLATIONS = {
    "On-Policy Critic": {"on_policy_data": True},
    "Unbounded Actions": {"bound_actions": False},
    "No Entropy": {"temperature": 0.0},
    "No LayerNorm": {"use_layer_norm": False},
    "Min Target": {"min_target": True}
}

# Dynamically build the final ALGORITHMS dictionary for fetching
ALGORITHMS = {BASELINE_ALGO_NAME: DEFAULT_CONFIG}
for name, changes in ABLATIONS.items():
    new_config = DEFAULT_CONFIG.copy()
    new_config.update(changes)
    ALGORITHMS[name] = new_config

# Define the tasks (environments) for the study
ALL_TASKS = ["InvertedDoublePendulum-v5", "Hopper-v5", "Walker2d-v5", "HalfCheetah-v5", "Ant-v5", "Humanoid-v5"]

# Helper function to generate mock data for testing
def generate_mock_data(tasks, algorithms, num_seeds):
    """Generate mock performance data for testing purposes."""
    print("Using mock data for testing...")
    np.random.seed(42)  # For reproducible results
    
    num_tasks = len(tasks)
    scores = {}
    
    # Create baseline scores (higher performance)
    baseline_scores = np.random.normal(loc=1000, scale=200, size=(num_seeds, num_tasks))
    baseline_scores = np.abs(baseline_scores)  # Ensure positive scores
    scores[BASELINE_ALGO_NAME] = baseline_scores
    
    # Create ablated algorithm scores (slightly worse performance)
    for algo_name in algorithms:
        if algo_name != BASELINE_ALGO_NAME:
            # Each ablation loses 5-25% performance with some noise
            loss_factor = np.random.uniform(0.75, 0.95)
            noise = np.random.normal(0, 50, size=(num_seeds, num_tasks))
            scores[algo_name] = baseline_scores * loss_factor + noise
            scores[algo_name] = np.abs(scores[algo_name])  # Ensure positive
    
    return scores

# Helper function to fetch and prepare data
def fetch_and_prepare_data(project_path, tasks, algorithms, num_seeds, score_key='evaluation/undisc_policy_return'):
    # This function remains the same as the previous version
    try:
        api = wandb.Api()
    except wandb.errors.ApiException as e:
        print(f"Failed to connect to W&B API: {e}")
        return None
        
    num_tasks = len(tasks)
    scores = {name: np.full((num_seeds, num_tasks), np.nan) for name in algorithms}
    
    for algo_name, algo_config in algorithms.items():
        print(f"Fetching data for: {algo_name}")
        for task_idx, task_name in enumerate(tasks):
            w_and_b_filter = { "config.env_name": task_name }
            for key, value in algo_config.items():
                w_and_b_filter[f"config.{key}"] = value

            runs = api.runs(project_path, filters=w_and_b_filter)
            
            for run_idx, run in enumerate(runs):
                if run_idx >= num_seeds: break
                history = run.history(keys=[score_key], pandas=True)
                # Current (last 10 steps):
                #final_performance = history[score_key].dropna().tail(LAST_POINTS_TO_AVERAGE).mean()

                # All steps:
                final_performance = history[score_key].dropna().mean()
                if not np.isnan(final_performance):
                    scores[algo_name][run_idx, task_idx] = final_performance
    return scores

def generate_ablation_plot(ax=None, algorithm_colors=None, show_legend=True, title_fontsize=16, label_fontsize=14, y_axis_order=None, fast_test=False):
    """
    Generate ablation plot. Can be used standalone or as part of a combined plot.
    
    Args:
        ax: matplotlib axes to plot on (if None, creates new figure)
        algorithm_colors: dict mapping algorithm names to colors
        show_legend: whether to show legend
        title_fontsize: font size for title
        label_fontsize: font size for labels
        y_axis_order: list defining the order of algorithms on y-axis
        fast_test: if True, use only 2 environments for faster testing
        
    Returns:
        list of algorithm names present in the plot
    """
    # Use limited tasks for fast testing
    tasks_to_use = ALL_TASKS[:2] if fast_test else ALL_TASKS
    
    # Fetch data (same logic as main script)
    if TEST_MODE:
        absolute_scores = generate_mock_data(tasks_to_use, ALGORITHMS, NUM_SEEDS)
    else:
        absolute_scores = fetch_and_prepare_data(PROJECT_PATH, tasks_to_use, ALGORITHMS, NUM_SEEDS)

    if not absolute_scores:
        print("Could not proceed due to missing ablation data.")
        return []
        
    # Calculate relative percent loss
    baseline_scores = absolute_scores[BASELINE_ALGO_NAME]
    ablated_algo_names = list(ABLATIONS.keys())
    
    loss_scores = {}
    for name in ablated_algo_names:
        ablated_scores = absolute_scores[name]
        percent_loss = 100 * (baseline_scores - ablated_scores) / (np.abs(baseline_scores) + 1e-10)
        loss_scores[name] = percent_loss

    # Get IQM and CIs of the loss scores
    aggregate_func = lambda x: np.array([rly.aggregate_iqm(x)])
    aggregate_loss, aggregate_loss_cis = rly_lib.get_interval_estimates(
        loss_scores, aggregate_func, reps=50000)

    # Map to consistent algorithm names
    mapped_names = [ALGORITHM_NAME_MAPPING.get(name, name) for name in ablated_algo_names]
    
    # Create plot
    if ax is None:
        plt.figure(figsize=(10, 6))
        fig, ax = rly_plot.plot_interval_estimates(
            aggregate_loss, aggregate_loss_cis,
            metric_names=['IQM'],
            algorithms=mapped_names, 
            xlabel='Percent Loss',
            xlabel_y_coordinate=-0.15,
            legend_kwargs={'loc': 'upper right', 'bbox_to_anchor': (1.0, 1.0), 'fontsize': 10} if show_legend else None)
    else:
        # For combined plots, we need to manually create the interval estimates plot
        # Based on debug output: aggregate_loss and aggregate_loss_cis are dicts with algorithm names as keys
        # aggregate_loss[algo_name] = array([iqm_value])
        # aggregate_loss_cis[algo_name] = array([[lower_bound], [upper_bound]])
        
        # Determine y-axis positioning based on global order
        if y_axis_order:
            # Map algorithm names to their positions in the global order
            y_positions = []
            ordered_mapped_names = []
            ordered_ablated_names = []
            
            for algo in y_axis_order:
                if algo in mapped_names:
                    idx = mapped_names.index(algo)
                    y_positions.append(len(y_axis_order) - 1 - y_axis_order.index(algo))  # Reverse for top-to-bottom
                    ordered_mapped_names.append(mapped_names[idx])
                    ordered_ablated_names.append(ablated_algo_names[idx])
            
            mapped_names = ordered_mapped_names
            ablated_algo_names = ordered_ablated_names
            y_positions = np.array(y_positions)
        else:
            y_positions = np.arange(len(mapped_names))
        
        # Plot horizontal bars with error bars
        colors = [algorithm_colors.get(name, 'gray') if algorithm_colors else 'steelblue' 
                 for name in mapped_names]
        
        for i, (mapped_name, orig_name, color, y_pos) in enumerate(zip(mapped_names, ablated_algo_names, colors, y_positions)):
            # Get IQM value (single element array)
            iqm_value = aggregate_loss[orig_name][0]  # Extract scalar from array
            
            ax.barh(y_pos, iqm_value, color=color, alpha=0.7, height=0.6)
            
            # Error bars: ci array is shape (2, 1) - [[lower], [upper]]
            if orig_name in aggregate_loss_cis:
                ci_array = aggregate_loss_cis[orig_name]
                lower_bound = ci_array[0, 0]  # First row, first column
                upper_bound = ci_array[1, 0]  # Second row, first column
                
                ax.errorbar(iqm_value, y_pos, xerr=[[iqm_value-lower_bound], [upper_bound-iqm_value]], 
                           color='black', linewidth=2, capsize=4)
        
        # Remove y-axis labels for combined plots (rely on shared legend)
        ax.set_yticks([])
        ax.set_yticklabels([])
        ax.set_xlabel('Percent Loss', fontsize=label_fontsize, color='#2C3E50')
    
    # Customize the plot
    ax.set_title('Ablations on PPO+', fontsize=title_fontsize, pad=20, fontweight='bold', color='#2C3E50')
    ax.axvline(0, color='black', linestyle='--', lw=1)
    ax.grid(True, alpha=0.3, linestyle='--', color='#BDC3C7', zorder=0)
    ax.set_facecolor('white')
    
    # Remove top and right spines for consistency
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#7F8C8D')
    ax.spines['bottom'].set_color('#7F8C8D')
    
    return mapped_names