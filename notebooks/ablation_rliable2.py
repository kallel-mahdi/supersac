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
TASKS = ["InvertedDoublePendulum-v5", "Hopper-v5", "Walker2d-v5", "HalfCheetah-v5", "Ant-v5", "Humanoid-v5"]

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

if __name__ == "__main__":
    # === Step 2: Fetch absolute performance scores ===
    if TEST_MODE:
        absolute_scores = generate_mock_data(TASKS, ALGORITHMS, NUM_SEEDS)
    else:
        absolute_scores = fetch_and_prepare_data(PROJECT_PATH, TASKS, ALGORITHMS, NUM_SEEDS)

    if not absolute_scores:
        print("Could not proceed due to missing data.")
    else:
        # === Step 3: Calculate relative percent loss ===
        baseline_scores = absolute_scores[BASELINE_ALGO_NAME]
        ablated_algo_names = list(ABLATIONS.keys())
        
        loss_scores = {}
        
        # Calculate loss for each ablation relative to baseline (exclude PPO+ from plot)
        for name in ablated_algo_names:
            ablated_scores = absolute_scores[name]
            percent_loss = 100 * (baseline_scores - ablated_scores) / (np.abs(baseline_scores) + 1e-10)
            loss_scores[name] = percent_loss

        # === Step 4: Get IQM and CIs of the loss scores ===
        # Following rliable documentation pattern with aggregate_func
        aggregate_func = lambda x: np.array([rly.aggregate_iqm(x)])
        aggregate_loss, aggregate_loss_cis = rly_lib.get_interval_estimates(
            loss_scores, aggregate_func, reps=50000)

        # === Step 5: Create the plot using rliable's plot_utils ===
        algo_names_for_plot = list(loss_scores.keys())  # Only ablations, no PPO+
        
        # Create larger figure to prevent overlapping
        plt.figure(figsize=(10, 6))
        
        fig, axes = rly_plot.plot_interval_estimates(
            aggregate_loss, aggregate_loss_cis,
            metric_names=['IQM'],
            algorithms=algo_names_for_plot, 
            xlabel='Percent Loss',
            xlabel_y_coordinate=-0.15,
            legend_kwargs={'loc': 'upper right', 'bbox_to_anchor': (1.0, 1.0), 'fontsize': 10})
        
        # Customize the plot
        axes.set_title('Ablations on PPO+', fontsize=16, pad=20)
        axes.axvline(0, color='black', linestyle='--', lw=1)
        axes.grid(axis='x', linestyle='--', alpha=0.7)
        
        # Adjust layout to prevent overlapping
        plt.subplots_adjust(left=0.25, right=0.85, top=0.9, bottom=0.15)
        
        output_path = "./ablation_loss_full_steps_plot.pdf"
        print(f"\nSaving final plot to {output_path}")
        plt.savefig(output_path, format="pdf", bbox_inches="tight")
        plt.show()