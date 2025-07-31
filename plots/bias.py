import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

sns.set_theme()

# Set up matplotlib defaults for consistent plots
plt.rcParams['figure.figsize'] = (14, 9)  # Larger, better proportioned figure
plt.rcParams['figure.dpi'] = 150  # Higher resolution for crisp output

# Font sizes - more elegant proportions
plt.rcParams['font.size'] = 12  # Base font size
plt.rcParams['axes.titlesize'] = 18  # Plot title
plt.rcParams['axes.labelsize'] = 15  # Axis labels
plt.rcParams['xtick.labelsize'] = 13  # X-axis tick labels
plt.rcParams['ytick.labelsize'] = 12  # Y-axis tick labels
plt.rcParams['legend.fontsize'] = 12  # Legend text

# Line and marker styles
plt.rcParams['lines.linewidth'] = 2.5
plt.rcParams['lines.markersize'] = 8

# Grid settings
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['grid.linestyle'] = '--'

# Legend settings
plt.rcParams['legend.frameon'] = True
plt.rcParams['legend.framealpha'] = 0.8
plt.rcParams['legend.edgecolor'] = 'gray'

# Save figure settings
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.pad_inches'] = 0.2

import pandas as pd
import wandb
import os
import numpy as np

# --- Configuration ---

# Set your W&B API key. Best practice is to set this as an environment variable.
# os.environ["WANDB_API_KEY"] = "YOUR_API_KEY"

# --- W&B API Setup ---

try:
    api = wandb.Api()
except wandb.errors.ApiException as e:
    print(f"Failed to connect to W&B API. Please check your API key. Error: {e}")
    exit()

# Define the W&B project path
ENTITY = "mahdikallel"
PROJECT = "AAAI_GRAD_100K"
PROJECT_PATH = f"{ENTITY}/{PROJECT}"

# Define the different configurations to plot. These will be our data sources.
CONFIGS = {
    "InvertedDoublePendulum": {"env_name": "InvertedDoublePendulum-v5"},
    "Hopper": {"env_name": "Hopper-v5"},
    "Walker2d": {"env_name": "Walker2d-v5"},
    "HalfCheetah": {"env_name": "HalfCheetah-v5"},
    "Ant": {"env_name": "Ant-v5"},
    "Humanoid": {"env_name": "Humanoid-v5"},  
}

# Define the metrics to be extracted for violin plot aggregation
METRICS = [
   "train/q_relative_bias_off_policy_control", # Corresponds to "Off-Policy"
   "train/q_relative_bias_on_policy_only",     # Corresponds to "On-Policy"
   "train/q_relative_bias_min_target",    # Corresponds to "Min Target"
   "train/q_relative_bias_no_layernorm",     # Corresponds to "No LayerNorm"
]

# --- Aesthetics Configuration ---

# Flag to enable overlaying individual environment medians as points
OVERLAY_ENVIRONMENT_MEDIANS = True

# Define the labels for the legend and the categories
LEGEND_LABELS = ["PPO+", "- Off-policy data", "+ Min target", "- LayerNorm"]

# Define a more vibrant and distinct color palette for better visual appeal
# Colors: Deep Teal, Warm Orange, Rich Purple, Coral Red
CUSTOM_PALETTE = ["#2E8B8B", "#FF8C42", "#8E44AD", "#E74C3C"]


# --- Helper Functions ---

def fetch_and_filter_runs(project_path, config_params):
    """Fetches and filters W&B runs based on config parameters."""
    all_runs = api.runs(project_path)
    matching_runs = [
        run for run in all_runs
        if all(run.config.get(key) == value for key, value in config_params.items())
    ]
    print(f"Found {len(matching_runs)} runs for config: {config_params}")
    return matching_runs

def get_data_for_config(runs, metrics, max_step):
    """Processes a list of runs to extract and combine their history data."""
    df_list = []
    for run in runs:
        try:
            history_df = run.history(samples=20000, keys=metrics + ['_step'])
            tmp_df = history_df.dropna(axis=0)
            tmp_df = tmp_df[tmp_df["_step"] < int(max_step)]
            if not tmp_df.empty:
                df_list.append(tmp_df[metrics])
        except Exception as e:
            print(f"Warning: Could not process history for run '{run.name}'. Error: {e}")
    
    return pd.concat(df_list, ignore_index=True) if df_list else None

def aggregate_data_across_environments(configs, metrics, max_step_limit):
    """
    Aggregate data from all environments for each metric to create violin plot data.
    
    Returns:
        If OVERLAY_ENVIRONMENT_MEDIANS is False:
            Dictionary where keys are metric names and values are lists of all bias values
            across all environments for that metric.
        If OVERLAY_ENVIRONMENT_MEDIANS is True:
            Tuple of (aggregated_data, environment_medians) where environment_medians
            contains median values per environment per metric.
    """
    aggregated_data = {metric: [] for metric in metrics}
    environment_medians = {metric: [] for metric in metrics} if OVERLAY_ENVIRONMENT_MEDIANS else None
    
    print(f"Fetching runs from project: {PROJECT_PATH}")
    
    for config_name, config_params in configs.items():
        print(f"\n--- Processing Config: {config_name} ---")
        
        matching_runs = fetch_and_filter_runs(PROJECT_PATH, config_params)
        if not matching_runs:
            print(f"No data found for {config_name}")
            continue
            
        plot_data = get_data_for_config(matching_runs, metrics, max_step_limit)
        if plot_data is None or plot_data.empty:
            print(f"No valid data for {config_name}")
            continue
        
        # Add all values from this environment to the aggregated data
        for metric in metrics:
            if metric in plot_data.columns:
                values = plot_data[metric].dropna().values
                aggregated_data[metric].extend(values.tolist())
                print(f"  Added {len(values)} {metric} values from {config_name}")
                
                # Store environment median if overlay flag is enabled
                if OVERLAY_ENVIRONMENT_MEDIANS and len(values) > 0:
                    env_median = np.median(values)
                    environment_medians[metric].append(env_median)
    
    # Convert to numpy arrays and filter out any remaining NaN values
    for metric in metrics:
        values = np.array(aggregated_data[metric])
        aggregated_data[metric] = values[~np.isnan(values)]
        print(f"Final {metric}: {len(aggregated_data[metric])} values")
    
    if OVERLAY_ENVIRONMENT_MEDIANS:
        return aggregated_data, environment_medians
    else:
        return aggregated_data

def generate_bias_plot(ax=None, algorithm_colors=None, show_legend=True, title_fontsize=16, label_fontsize=14, y_axis_order=None, fast_test=False):
    """
    Generate bias violin plot. Can be used standalone or as part of a combined plot.
    
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
    MAX_STEP_LIMIT = 500_000
    
    print("=== Aggregating Data Across All Environments ===")
    
    # Use limited configs for fast testing
    configs_to_use = dict(list(CONFIGS.items())[:2]) if fast_test else CONFIGS
    
    # Aggregate data across all environments
    if OVERLAY_ENVIRONMENT_MEDIANS:
        aggregated_data, environment_medians = aggregate_data_across_environments(configs_to_use, METRICS, MAX_STEP_LIMIT)
    else:
        aggregated_data = aggregate_data_across_environments(configs_to_use, METRICS, MAX_STEP_LIMIT)
    
    # Check if we have data for all metrics
    valid_metrics = [metric for metric, data in aggregated_data.items() if len(data) > 0]
    if not valid_metrics:
        print("No valid bias data found for any metrics!")
        return []
    
    print(f"\nCreating violin plot with {len(valid_metrics)} metrics")
    
    # Prepare data for violin plot with outlier removal
    violin_data = []
    violin_labels = []
    
    def remove_outliers_iqr(data):
        """Remove outliers using the IQR method."""
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        return data[(data >= lower_bound) & (data <= upper_bound)]
    
    for i, metric in enumerate(METRICS):
        if metric in aggregated_data and len(aggregated_data[metric]) > 0:
            values = aggregated_data[metric]
            original_count = len(values)
            
            # Remove outliers
            values_no_outliers = remove_outliers_iqr(values)
            outliers_removed = original_count - len(values_no_outliers)
            
            violin_data.append(values_no_outliers)
            violin_labels.append(LEGEND_LABELS[i])
            print(f"  {LEGEND_LABELS[i]}: {len(values_no_outliers)} values (removed {outliers_removed} outliers), range [{np.min(values_no_outliers):.3f}, {np.max(values_no_outliers):.3f}]")
    
    if not violin_data:
        print("No valid data for violin plot!")
        return []
    
    # Create violin plot on provided axes (or new figure if ax is None)
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(14, 9))
        fig.patch.set_facecolor('white')
    
    # Determine y-axis positioning based on global order
    if y_axis_order:
        # Map violin labels to their positions in the global order
        y_positions = []
        ordered_violin_data = []
        ordered_violin_labels = []
        
        for algo in y_axis_order:
            if algo in violin_labels:
                idx = violin_labels.index(algo)
                y_positions.append(len(y_axis_order) - 1 - y_axis_order.index(algo))  # Reverse for top-to-bottom
                ordered_violin_data.append(violin_data[idx])
                ordered_violin_labels.append(violin_labels[idx])
        
        violin_data = ordered_violin_data
        violin_labels = ordered_violin_labels
        positions = np.array(y_positions)
    else:
        # Default positioning if no global order provided
        positions = np.arange(len(violin_data))
    
    # Create horizontal violin plot with enhanced parameters
    parts = ax.violinplot(
        violin_data,
        positions=positions,
        showmeans=True,
        showmedians=True,
        showextrema=True,
        widths=0.7,  # Make violins wider for better visibility
        bw_method=0.3,  # Adjust bandwidth for smoother curves
        vert=False  # Make horizontal violins
    )
    
    # Customize violin colors with enhanced styling
    for i, pc in enumerate(parts['bodies']):
        if i < len(violin_labels):
            if algorithm_colors and violin_labels[i] in algorithm_colors:
                color = algorithm_colors[violin_labels[i]]
            else:
                color = CUSTOM_PALETTE[i] if i < len(CUSTOM_PALETTE) else 'gray'
            pc.set_facecolor(color)
            pc.set_alpha(0.8)  # Slightly more opaque for better visibility
            pc.set_edgecolor('darkgray')
            pc.set_linewidth(1.5)
    
    # Customize other violin plot elements with better contrast
    parts['cmeans'].set_color('#2C3E50')  # Dark blue-gray for means
    parts['cmeans'].set_linewidth(2.5)
    parts['cmeans'].set_linestyle('--')  # Dashed line for means
    
    parts['cmedians'].set_color('white')  # White medians for contrast
    parts['cmedians'].set_linewidth(3)
    
    parts['cbars'].set_color('#34495E')  # Darker gray for bars
    parts['cbars'].set_linewidth(1.8)
    parts['cmins'].set_color('#34495E')
    parts['cmins'].set_linewidth(1.8)
    parts['cmaxes'].set_color('#34495E')
    parts['cmaxes'].set_linewidth(1.8)
    
    # Overlay individual environment medians if flag is enabled
    if OVERLAY_ENVIRONMENT_MEDIANS:
        for i, (pos, label) in enumerate(zip(positions, violin_labels)):
            # Find corresponding metric for this label
            metric_idx = LEGEND_LABELS.index(label) if label in LEGEND_LABELS else None
            if metric_idx is not None and metric_idx < len(METRICS):
                metric = METRICS[metric_idx]
                if metric in environment_medians and len(environment_medians[metric]) > 0:
                    env_medians = environment_medians[metric]
                    # Create y positions with small random jitter for visibility (swapped x/y for horizontal)
                    y_positions_jitter = np.full(len(env_medians), pos) + np.random.normal(0, 0.05, len(env_medians))
                    
                    # Plot star markers for environment medians
                    ax.scatter(
                        env_medians, y_positions_jitter,  # Swapped x/y for horizontal
                        color='gold',  # Bright yellow/gold color
                        alpha=0.9,     # More opaque for better visibility
                        s=120,         # Larger size
                        marker='*',    # Star shape
                        edgecolors='black',
                        linewidth=1.5,
                        zorder=3  # Ensure points appear on top of violins
                    )
    
    # Add median value labels to the right of violins (for horizontal layout)
    for i, (pos, data) in enumerate(zip(positions, violin_data)):
        median_val = np.median(data)
        
        # Calculate violin right position for label placement
        x_max = np.max(data)
        x_offset = (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.02  # 2% of plot width to the right
        
        # Add median label to the right of the violin
        ax.text(
            x_max + x_offset, pos, f'{median_val:.2f}',
            ha='left', va='center',
            color='#2C3E50',
            fontsize=11,
            fontweight='bold',
            bbox=dict(facecolor='white', alpha=0.9, pad=3, boxstyle='round,pad=0.3', edgecolor='#2C3E50', linewidth=1)
        )
    
    # --- Enhanced Styling ---
    ax.set_title('Q-Function Bias Distribution', 
                fontsize=title_fontsize, pad=20, fontweight='bold', color='#2C3E50')
    ax.set_xlabel('Relative Bias', fontsize=label_fontsize, color='#2C3E50')
    
    # Set y-axis labels for transposed plot
    ax.set_yticks(positions)
    if ax != plt.gca() or not show_legend:  # For combined plots, don't show y-labels (rely on legend)
        ax.set_yticklabels([])
    else:
        ax.set_yticklabels(violin_labels, fontweight='medium', fontsize=label_fontsize-2)
    
    # Add vertical line at zero for reference with better styling (changed from horizontal)
    ax.axvline(x=0, color='#7F8C8D', linestyle='-', linewidth=2, alpha=0.8, zorder=1)
    
    # Add subtle grid
    ax.grid(True, alpha=0.3, linestyle='--', color='#BDC3C7', zorder=0)
    
    # Set background color
    ax.set_facecolor('white')
    
    # Remove top and right spines for cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#7F8C8D')
    ax.spines['bottom'].set_color('#7F8C8D')
    
    # Sample count labels removed for cleaner appearance in combined plots
    
    return violin_labels