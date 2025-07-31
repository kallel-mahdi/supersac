

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

sns.set_theme()

# Set up matplotlib defaults for consistent plots
plt.rcParams['figure.figsize'] = (10, 6)  # Standard figure size
plt.rcParams['figure.dpi'] = 100  # Standard resolution

# Font sizes
plt.rcParams['font.size'] = 14  # Base font size
plt.rcParams['axes.titlesize'] = 22  # Plot title
plt.rcParams['axes.labelsize'] = 18  # Axis labels
plt.rcParams['xtick.labelsize'] = 14  # X-axis tick labels
plt.rcParams['ytick.labelsize'] = 14  # Y-axis tick labels
plt.rcParams['legend.fontsize'] = 18  # Legend text

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

# Define the different configurations to plot. These will be our subplots.
CONFIGS = {
    "InvertedDoublePendulum": {"env_name": "InvertedDoublePendulum-v5"},
    "Hopper": {"env_name": "Hopper-v5"},
    "Walker2d": {"env_name": "Walker2d-v5"},
    "HalfCheetah": {"env_name": "HalfCheetah-v5"},
    "Ant": {"env_name": "Ant-v5"},
    "Humanoid": {"env_name": "Humanoid-v5"},  
}

# Define the metrics to be extracted. The order here is important and must match
# the labels and colors defined below.
# METRICS = [
#     "train/cosine_true_normal", # Corresponds to "Off-Policy"
#     "train/cosine_true_on",     # Corresponds to "On-Policy"
#     "train/cosine_true_min",    # Corresponds to "Min Target"
#     "train/cosine_true_no",     # Corresponds to "No LayerNorm"
# ]

METRICS = [
   "train/q_relative_bias_off_policy_control", # Corresponds to "Off-Policy"
   "train/q_relative_bias_on_policy_only",     # Corresponds to "On-Policy"
   "train/q_relative_bias_min_target",    # Corresponds to "Min Target"
   "train/q_relative_bias_no_layernorm",     # Corresponds to "No LayerNorm"
]

# --- Aesthetics Configuration to Match Example Image ---

# Define the labels for the legend and the categories
LEGEND_LABELS = ["Off-Policy", "On-Policy", "Min Target", "No LayerNorm"]

# Define a custom color palette to match the example image
# Colors in order: Light Teal, Light Yellow, Light Purple, Light Red/Salmon
CUSTOM_PALETTE = ["#a1d9d4", "#fefec1", "#d3c9e6", "#f9b6b2"]


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


# --- Main Plotting Script ---

if __name__ == "__main__":
    # Set the overall theme to be simple and clean
    sns.set_theme(style="ticks", rc={"font.family": "serif"})
    
    MAX_STEP_LIMIT = 500_000
    
    # Create a figure with 3 subplots, arranged in 1 row and 3 columns
    fig, axs = plt.subplots(2, 3, figsize=(12, 8), sharey=True)
    
    print(f"Fetching runs from project: {PROJECT_PATH}")

    # Iterate through each configuration and its corresponding subplot axis
    for ax, (config_name, config_params) in zip(axs.flatten(), CONFIGS.items()):
        print(f"\n--- Processing Config: {config_name} ---")
        
        matching_runs = fetch_and_filter_runs(PROJECT_PATH, config_params)
        if not matching_runs:
            ax.text(0.5, 0.5, "No data found", ha='center', va='center')
            continue
            
        plot_data = get_data_for_config(matching_runs, METRICS, MAX_STEP_LIMIT)
        if plot_data is None or plot_data.empty:
            ax.text(0.5, 0.5, "No valid data", ha='center', va='center')
            continue
       
            
        # Ensure the data columns are in the correct order for plotting
        plot_data = plot_data[METRICS]
        
        # Generate the boxplot with custom styling
        sns.boxplot(
            data=plot_data,
            ax=ax,
            palette=CUSTOM_PALETTE,
            showfliers=True, # Show outliers as in the example
            fliersize=3,
            linewidth=1.2,
            medianprops={'color': 'black', 'linewidth': 2},
            whiskerprops={'color': 'black', 'linewidth': 1.2},
            capprops={'color': 'black', 'linewidth': 1.2},
            boxprops={'edgecolor': 'black'}
        )
        
        # Add median value labels inside the boxes, styled like the example
        medians = plot_data.median()
        for i, median_val in enumerate(medians):
            ax.text(
                i, median_val, f'{median_val:.2f}',
                ha='center', va='center',
                color='white',
                fontsize=10,
                fontweight='bold',
                bbox=dict(facecolor='black', alpha=0.6, pad=2, boxstyle='round,pad=0.2')
            )
            
        # --- Subplot Styling ---
        ax.set_title(f"Env name: {config_params['env_name']}", fontsize=12)
        # Remove x-axis labels and ticks as per the example
        ax.set_xlabel('')
        ax.set_xticks([])
        # Set y-axis limits
        ax.set_ylim(-1.1, 1.1)

    # --- Global Figure Styling ---
    
    # Set a shared y-axis label for the leftmost plot
    axs[0,0].set_ylabel('Cosine Similarity', fontsize=12)
    
    # Create custom legend handles (patches) for the categories
    legend_patches = [
        mpatches.Patch(color=color, label=label)
        for color, label in zip(CUSTOM_PALETTE, LEGEND_LABELS)
    ]
    
    # Add the shared, horizontal legend to the top of the figure
    fig.legend(
        handles=legend_patches,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.05), # Position it above the plots
        ncol=len(LEGEND_LABELS),   # Make it horizontal
        frameon=False,             # No border around the legend
        fontsize=11
    )
    
    # handles, labels = axs[0,0].get_legend_handles_labels()
    # fig.legend(handles, labels, loc='upper center', ncol=len(LEGEND_LABELS), bbox_to_anchor=(0.5, 1.05))
    
    # # Adjust layout to prevent titles/labels from overlapping
    # plt.tight_layout(rect=[0, 0, 1, 0.95]) # Adjust rect to make space for the legend
    
    # Save the figure to a file
    output_path = "./gradient_comparison_styled.pdf"
    print(f"\nSaving final plot to {output_path}")
    plt.savefig(output_path, format="pdf", bbox_inches="tight")
    
    plt.show()
