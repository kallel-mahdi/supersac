"""
Generates a 2x3 grid of boxplots comparing cosine similarities for different
components of the gradient update, using data fetched from Weights & Biases.

This script is derived from a Jupyter notebook and adapted to use the centralized
styling provided in `plots/style.py`. It automates fetching data for different
MuJoCo environments and plots the cosine similarity metrics for various algorithm
configurations.

The script will:
1. Connect to the W&B API to fetch experiment data.
2. Iterate through a predefined set of environments (e.g., Hopper, Walker2d).
3. For each environment, create a boxplot of cosine similarity metrics.
4. Use the styling from `plots/style.py` for a consistent, publication-ready look.
5. Generate a shared legend for the plots.
6. Save the final figure as both PDF and PNG.
"""
import os
import pandas as pd
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches

# Import and apply styling from plots/style.py
# This sets the theme, fonts, and other rcParams.
try:
    from plots.style import setup_shared_legend, save_publication_figure
    # The import of `plots.style` itself applies the rcParams
    import plots.style
except ImportError:
    print("Warning: Could not import from plots.style. Plot styling will be basic.")
    print("Ensure the script is run from the project root directory.")
    # Define dummy functions if import fails to avoid crashing
    def setup_shared_legend(fig, handles, labels, ncol, **kwargs):
        fig.legend(handles, labels, loc='upper center', ncol=ncol)

    def save_publication_figure(fig, path_without_extension):
        fig.savefig(f"{path_without_extension}.pdf", bbox_inches='tight')
        fig.savefig(f"{path_without_extension}.png", bbox_inches='tight', dpi=300)

# --- W&B Configuration ---
# Best practice is to set your W&B API key as an environment variable.
# os.environ["WANDB_API_KEY"] = "YOUR_API_KEY"

ENTITY = "mahdikallel"
PROJECT = "AAAI_GRAD_100K"
PROJECT_PATH = f"{ENTITY}/{PROJECT}"
MAX_STEP_LIMIT = 500_000

# --- Plotting Configuration ---
# Environments to generate subplots for.
CONFIGS = {
    "InvertedDoublePendulum": {"env_name": "InvertedDoublePendulum-v5"},
    "Hopper": {"env_name": "Hopper-v5"},
    "Walker2d": {"env_name": "Walker2d-v5"},
    "HalfCheetah": {"env_name": "HalfCheetah-v5"},
    "Ant": {"env_name": "Ant-v5"},
    "Humanoid": {"env_name": "Humanoid-v5"},
}

# Metrics to fetch from W&B. The order is important for the legend.
# METRICS = [
#     "train/cosine_true_normal",
#     "train/cosine_true_on",
#     "train/cosine_true_min",
#     "train/cosine_true_no",
# ]

METRICS = [
   "train/q_relative_bias_off_policy_control",     # Corresponds to "On-Policy"
   "train/q_relative_bias_on_policy_only", # Corresponds to "Off-Policy"   
   "train/q_relative_bias_min_target",     # Corresponds to "Min Target"
   "train/q_relative_bias_no_layernorm",    # Corresponds to "No LayerNorm"
]

# Labels for the legend, corresponding to the METRICS.
LEGEND_LABELS = ["Off-Policy", "On-Policy", "Min Target", "No LayerNorm"]


# --- Helper Functions ---

def fetch_and_filter_runs(api, project_path, config_params):
    """Fetches and filters W&B runs based on config parameters."""
    print(f"Fetching runs for config: {config_params}")
    all_runs = api.runs(project_path)
    matching_runs = [
        run for run in all_runs
        if all(run.config.get(key) == value for key, value in config_params.items())
    ]
    print(f"Found {len(matching_runs)} matching runs.")
    return matching_runs

def get_data_for_config(runs, metrics, max_step):
    """Processes a list of runs to extract and combine their history data."""
    df_list = []
    for run in runs:
        try:
            # Fetch history, drop rows with any NaNs in the specified metrics.
            history_df = run.history(samples=20000, keys=metrics + ['_step'])
            tmp_df = history_df.dropna(subset=metrics, axis=0)
            tmp_df = tmp_df[tmp_df["_step"] < int(max_step)]
            if not tmp_df.empty:
                df_list.append(tmp_df[metrics])
        except Exception as e:
            print(f"Warning: Could not process history for run '{run.name}'. Error: {e}")

    return pd.concat(df_list, ignore_index=True) if df_list else pd.DataFrame()


# --- Main Plotting Script ---

def main():
    """Main function to generate and save the plot."""
    try:
        api = wandb.Api()
    except wandb.errors.ApiException as e:
        print(f"Failed to connect to W&B API. Please check your API key. Error: {e}")
        return



    # Create figure and axes. Figsize is inherited from `plots.style.PLOT_PARAMS`.
    fig, axs = plt.subplots(2, 3, sharey=True)
    print(f"Fetching runs from project: {PROJECT_PATH}")

    for ax, (config_name, config_params) in zip(axs.flatten(), CONFIGS.items()):
        print(f"\n--- Processing Config: {config_name} ---")

        matching_runs = fetch_and_filter_runs(api, PROJECT_PATH, config_params)
        if not matching_runs:
            ax.text(0.5, 0.5, "No data found", ha='center', va='center', transform=ax.transAxes)
            ax.set_title(config_name)
            continue

        plot_data = get_data_for_config(matching_runs, METRICS, MAX_STEP_LIMIT)

        if plot_data.empty:
            ax.text(0.5, 0.5, "No valid data", ha='center', va='center', transform=ax.transAxes)
            ax.set_title(config_name)
            continue

        # Ensure columns are in the correct order for plotting and legend.
        plot_data = plot_data[METRICS]

        # Generate the boxplot, letting seaborn assign colors from the default palette.
        sns.boxplot(
            data=plot_data,
            ax=ax,
            showfliers=False,
            linewidth=1.2,
            medianprops={'color': 'black', 'linewidth': 2},
            whiskerprops={'color': 'black', 'linewidth': 1.2},
            capprops={'color': 'black', 'linewidth': 1.2},
            boxprops={'edgecolor': 'black'}
        )

        # Add median value labels inside the boxes.
        medians = plot_data.median()
        for i, median_val in enumerate(medians):
            ax.text(
                i, median_val, f'{median_val:.2f}',
                ha='center', va='bottom',  # Align text nicely
                color='white',
                fontsize=8,
                fontweight='bold',
                bbox=dict(facecolor='black', alpha=0.6, pad=2, boxstyle='round,pad=0.2')
            )

        # --- Subplot Styling ---
        ax.set_title(config_name)
        ax.set_xlabel('')
        ax.set_xticks([])
        ax.set_ylim(-1.1, 1.1)

    # --- Global Figure Styling ---
    axs[0, 0].set_ylabel('Cosine Similarity')

    # Create legend handles using seaborn's current color palette.
    palette = sns.color_palette(n_colors=len(LEGEND_LABELS))
    legend_patches = [
        mpatches.Patch(color=color, label=label)
        for color, label in zip(palette, LEGEND_LABELS)
    ]

    # Use the shared legend helper from style.py
    setup_shared_legend(fig, legend_patches, [p.get_label() for p in legend_patches], ncol=len(LEGEND_LABELS))

    # Adjust layout to prevent titles/labels from overlapping with the legend.
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save the figure using the helper from style.py
    output_path_no_ext = "plots/cosine_separate"
    print(f"\nSaving final plot to {output_path_no_ext}.pdf and .png")
    save_publication_figure(fig, output_path_no_ext)

    plt.show()

if __name__ == "__main__":
    main()
