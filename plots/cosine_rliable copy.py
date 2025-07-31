import wandb
import numpy as np
import rliable.library as rly_lib
import rliable.metrics as rly
import rliable.plot_utils as rly_plot
import matplotlib.pyplot as plt
from collections import defaultdict

# --- Configuration (Copied from your script) ---
ENTITY = "mahdikallel"
PROJECT = "AAAI_GRAD_100K"
PROJECT_PATH = f"{ENTITY}/{PROJECT}"

CONFIGS = {
    "InvertedDoublePendulum": {"env_name": "InvertedDoublePendulum-v5"},
    "Hopper": {"env_name": "Hopper-v5"},
    "Walker2d": {"env_name": "Walker2d-v5"},
    "HalfCheetah": {"env_name": "HalfCheetah-v5"},
    "Ant": {"env_name": "Ant-v5"},
    "Humanoid": {"env_name": "Humanoid-v5"},
}

METRICS = [
    "train/cosine_true_normal",
    "train/cosine_true_on",
    "train/cosine_true_min",
    "train/cosine_true_no",
]

# METRICS = [
#    "train/q_relative_bias_off_policy_control",     # Corresponds to "On-Policy"
#    "train/q_relative_bias_on_policy_only", # Corresponds to "Off-Policy"   
#    "train/q_relative_bias_min_target",     # Corresponds to "Min Target"
#    "train/q_relative_bias_no_layernorm",    # Corresponds to "No LayerNorm"
# ]

LEGEND_LABELS = ["PPO+", "- Off-policy data", "+ Min target", "- LayerNorm"]

# Add this helper function before fetch_and_reshape_for_rliable
def fetch_and_filter_runs(api, project_path, config_params):
    """Fetches and filters W&B runs based on config parameters."""
    all_runs = api.runs(project_path)
    matching_runs = [
        run for run in all_runs
        if all(run.config.get(key) == value for key, value in config_params.items())
    ]
    return matching_runs

def fetch_and_reshape_for_rliable(project_path, configs, metrics, labels):
    """
    Fetches data from W&B and reshapes it into the format required by rliable.
    Output format: { 'AlgorithmName': np.array((num_runs, num_tasks)), ... }
    """
    try:
        api = wandb.Api()
    except wandb.errors.ApiException as e:
        print(f"Failed to connect to W&B API: {e}")
        return None

    # This dictionary will store scores temporarily
    # { 'AlgorithmName': { 'TaskName': [score1, score2, ...] } }
    temp_scores = defaultdict(lambda: defaultdict(list))
    
    task_names = list(configs.keys())
    num_runs_per_task = -1

    print("Fetching and processing data for rliable...")
    for task_name, task_config in configs.items():
        print(f"  Processing Task: {task_name}")
        # Use the same filtering approach as PLOTS.ipynb
        runs = fetch_and_filter_runs(api, project_path, task_config)
        
        if not runs:
            print(f"    Warning: No runs found for {task_name}")
            continue
            
        print(f"    Found {len(runs)} runs for {task_name}")
        
        if num_runs_per_task == -1:
            num_runs_per_task = len(runs)
        elif num_runs_per_task != len(runs):
            print("    Warning: Tasks have a different number of seeds. RLiable requires consistent run counts.")

        for run in runs:
            # For each metric (algorithm), get its history and calculate a single score
            for metric, label in zip(metrics, labels):
                try:
                    # Using .mean() to get a single score from the time series
                    score = run.history(keys=[metric], pandas=True)[metric].mean()
                    if not np.isnan(score):
                        temp_scores[label][task_name].append(score)
                except (KeyError, AttributeError):
                    # Handle cases where a metric might be missing in a run
                    pass

    # Check if we found any runs at all
    if num_runs_per_task == -1:
        print("Error: No runs found for any task. Please check your project path and configurations.")
        return None

    # Convert the temporary dictionary into the final NumPy array format
    num_tasks = len(task_names)
    final_scores = {}
    for label in labels:
        # Create a matrix of NaNs to handle missing data gracefully
        score_matrix = np.full((num_runs_per_task, num_tasks), np.nan)
        for task_idx, task_name in enumerate(task_names):
            scores_for_task = temp_scores[label].get(task_name, [])
            # Ensure we don't try to fill more runs than we have
            num_scores_to_fill = min(len(scores_for_task), num_runs_per_task)
            score_matrix[:num_scores_to_fill, task_idx] = scores_for_task[:num_scores_to_fill]
        
        # Only include algorithms that had at least some data
        if not np.all(np.isnan(score_matrix)):
            final_scores[label] = score_matrix
            
    return final_scores

def generate_cosine_plot(ax=None, algorithm_colors=None, show_legend=True, title_fontsize=16, label_fontsize=14, y_axis_order=None, fast_test=False):
    """
    Generate cosine similarity plot. Can be used standalone or as part of a combined plot.
    
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
    # Use limited configs for fast testing
    configs_to_use = dict(list(CONFIGS.items())[:2]) if fast_test else CONFIGS
    scores = fetch_and_reshape_for_rliable(PROJECT_PATH, configs_to_use, METRICS, LEGEND_LABELS)

    if not scores:
        print("Could not generate cosine plot due to lack of data.")
        return []
        
    algorithms = list(scores.keys())
    
    # Calculate IQM and confidence intervals using proper rliable functions
    aggregate_func = lambda x: np.array([rly.aggregate_iqm(x)])
    aggregate_scores, aggregate_interval_estimates = rly_lib.get_interval_estimates(
        scores, aggregate_func, reps=50000
    )
    
    # Create plot
    if ax is None:
        plt.figure(figsize=(10, 6))
        fig, ax = rly_plot.plot_interval_estimates(
            aggregate_scores, 
            aggregate_interval_estimates,
            metric_names=['IQM'],
            algorithms=algorithms,
            xlabel='Cosine Similarity',
            xlabel_y_coordinate=-0.15,
            legend_kwargs={'loc': 'upper right', 'bbox_to_anchor': (1.0, 1.0), 'fontsize': 10} if show_legend else None
        )
    else:
        # For combined plots, manually create the interval estimates plot
        # Based on rliable structure: aggregate_scores and aggregate_interval_estimates are dicts with algorithm names as keys
        # aggregate_scores[algo_name] = array([iqm_value])
        # aggregate_interval_estimates[algo_name] = array([[lower_bound], [upper_bound]])
        
        # Determine y-axis positioning based on global order
        if y_axis_order:
            # Map algorithm names to their positions in the global order
            y_positions = []
            ordered_algorithms = []
            
            for algo in y_axis_order:
                if algo in algorithms:
                    idx = algorithms.index(algo)
                    y_positions.append(len(y_axis_order) - 1 - y_axis_order.index(algo))  # Reverse for top-to-bottom
                    ordered_algorithms.append(algorithms[idx])
            
            algorithms = ordered_algorithms
            y_positions = np.array(y_positions)
        else:
            y_positions = np.arange(len(algorithms))
        
        # Plot horizontal bars with error bars
        colors = [algorithm_colors.get(name, 'gray') if algorithm_colors else 'steelblue' 
                 for name in algorithms]
        
        for i, (name, color, y_pos) in enumerate(zip(algorithms, colors, y_positions)):
            # Get IQM value (single element array)
            iqm_value = aggregate_scores[name][0]  # Extract scalar from array
            
            ax.barh(y_pos, iqm_value, color=color, alpha=0.7, height=0.6)
            
            # Error bars: ci array is shape (2, 1) - [[lower], [upper]]
            if name in aggregate_interval_estimates:
                ci_array = aggregate_interval_estimates[name]
                lower_bound = ci_array[0, 0]  # First row, first column
                upper_bound = ci_array[1, 0]  # Second row, first column
                
                ax.errorbar(iqm_value, y_pos, xerr=[[iqm_value-lower_bound], [upper_bound-iqm_value]], 
                           color='black', linewidth=2, capsize=4)
        
        # Remove y-axis labels for combined plots (rely on shared legend)  
        ax.set_yticks([])
        ax.set_yticklabels([])
        ax.set_xlabel('Cosine Similarity', fontsize=label_fontsize, color='#2C3E50')
    
    # Customize the plot
    ax.set_title('Gradient Quality (IQM)', fontsize=title_fontsize, pad=20, fontweight='bold', color='#2C3E50')
    ax.grid(True, alpha=0.3, linestyle='--', color='#BDC3C7', zorder=0)
    ax.set_facecolor('white')
    
    # Remove top and right spines for consistency
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#7F8C8D')
    ax.spines['bottom'].set_color('#7F8C8D')
    
    return algorithms