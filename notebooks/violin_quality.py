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
PROJECT = "AAAI_GRAD_N_BIAS"
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

# Define the labels for the legend and the categories
LEGEND_LABELS = ["Off-Policy", "On-Policy", "Min Target", "No LayerNorm"]

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
        Dictionary where keys are metric names and values are lists of all bias values
        across all environments for that metric.
    """
    aggregated_data = {metric: [] for metric in metrics}
    
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
    
    # Convert to numpy arrays and filter out any remaining NaN values
    for metric in metrics:
        values = np.array(aggregated_data[metric])
        aggregated_data[metric] = values[~np.isnan(values)]
        print(f"Final {metric}: {len(aggregated_data[metric])} values")
    
    return aggregated_data


# --- Main Plotting Script ---

if __name__ == "__main__":
    # Set the overall theme to be simple and clean
    sns.set_theme(style="ticks", rc={"font.family": "serif"})
    
    MAX_STEP_LIMIT = 500_000
    
    print("=== Aggregating Data Across All Environments ===")
    
    # Aggregate data across all environments
    aggregated_data = aggregate_data_across_environments(CONFIGS, METRICS, MAX_STEP_LIMIT)
    
    # Check if we have data for all metrics
    valid_metrics = [metric for metric, data in aggregated_data.items() if len(data) > 0]
    if not valid_metrics:
        print("No valid data found for any metrics!")
        exit()
    
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
    
    # Create the figure with better styling
    fig, ax = plt.subplots(1, 1, figsize=(14, 9))
    fig.patch.set_facecolor('white')
    
    # Create violin plot with enhanced parameters
    positions = np.arange(len(violin_data))
    parts = ax.violinplot(
        violin_data,
        positions=positions,
        showmeans=True,
        showmedians=True,
        showextrema=True,
        widths=0.7,  # Make violins wider for better visibility
        bw_method=0.3  # Adjust bandwidth for smoother curves
    )
    
    # Customize violin colors with enhanced styling
    for i, pc in enumerate(parts['bodies']):
        if i < len(CUSTOM_PALETTE):
            pc.set_facecolor(CUSTOM_PALETTE[i])
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
    
    # Add cleaner median value labels only (less cluttered)
    for i, data in enumerate(violin_data):
        median_val = np.median(data)
        
        # Add median label with better styling
        ax.text(
            i, median_val, f'{median_val:.2f}',
            ha='center', va='center',
            color='white',
            fontsize=11,
            fontweight='bold',
            bbox=dict(facecolor='#2C3E50', alpha=0.9, pad=3, boxstyle='round,pad=0.3', edgecolor='white', linewidth=1)
        )
    
    # --- Enhanced Styling ---
    ax.set_title('Q-Function Relative Bias Distribution Across All Environments', 
                fontsize=18, pad=25, fontweight='bold', color='#2C3E50')
    ax.set_ylabel('Relative Bias', fontsize=16, fontweight='semibold', color='#34495E')
    ax.set_xlabel('Estimator Type', fontsize=16, fontweight='semibold', color='#34495E')
    
    # Set x-axis labels with better spacing
    ax.set_xticks(positions)
    ax.set_xticklabels(violin_labels, rotation=0, fontweight='medium')
    
    # Add horizontal line at zero for reference with better styling
    ax.axhline(y=0, color='#7F8C8D', linestyle='-', linewidth=2, alpha=0.8, zorder=1)
    
    # Add subtle grid
    ax.grid(True, alpha=0.2, linestyle=':', color='#BDC3C7', zorder=0)
    
    # Set background color
    ax.set_facecolor('#FAFAFA')
    
    # Remove top and right spines for cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#7F8C8D')
    ax.spines['bottom'].set_color('#7F8C8D')
    
    # Add subtle statistics as text below x-axis labels
    for i, (label, data) in enumerate(zip(violin_labels, violin_data)):
        mean_val = np.mean(data)
        std_val = np.std(data)
        n_samples = len(data)
        
        # Add sample count below each violin
        ax.text(i, ax.get_ylim()[0] - 0.05 * (ax.get_ylim()[1] - ax.get_ylim()[0]), 
                f'n={n_samples}',
                ha='center', va='top', fontsize=9, color='#7F8C8D', 
                style='italic')
    
    # Adjust layout with better spacing
    plt.tight_layout(pad=3.0)
    
    # Save the figure with high quality
    output_path = "./q_bias_violin_comparison_enhanced.pdf"
    print(f"\nSaving enhanced violin plot to {output_path}")
    plt.savefig(output_path, format="pdf", bbox_inches="tight", dpi=300, facecolor='white')
    
    plt.show()
    
    # Print summary statistics
    print("\n=== Summary Statistics ===")
    for i, (label, data) in enumerate(zip(violin_labels, violin_data)):
        print(f"\n{label}:")
        print(f"  Count: {len(data)}")
        print(f"  Mean: {np.mean(data):.4f}")
        print(f"  Median: {np.median(data):.4f}")
        print(f"  Std: {np.std(data):.4f}")
        print(f"  Min: {np.min(data):.4f}")
        print(f"  Max: {np.max(data):.4f}")
        print(f"  25th percentile: {np.percentile(data, 25):.4f}")
        print(f"  75th percentile: {np.percentile(data, 75):.4f}") 