import wandb
import numpy as np
import rliable.metrics as rly
import rliable.plot_utils as rly_plot
import matplotlib.pyplot as plt
from collections import defaultdict

# --- Configuration (Copied from your script) ---
ENTITY = "mahdikallel"
PROJECT = "AAAI_GRAD_110K"
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

LEGEND_LABELS = ["Off-Policy", "On-Policy", "Min Target", "No LayerNorm"]

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


if __name__ == "__main__":
    scores = fetch_and_reshape_for_rliable(PROJECT_PATH, CONFIGS, METRICS, LEGEND_LABELS)

    if not scores:
        print("Could not generate plots due to lack of data.")
    else:
        algorithms = list(scores.keys())
        
        # --- 1. Probability of Improvement Matrix (P(X > Y)) ---
        print("\nGenerating Probability of Improvement Matrix...")
        probability_matrix = rly.create_probability_matrix(scores)
        
        fig_prob, ax_prob = plt.subplots(figsize=(6, 6))
        rly_plot.plot_probability_matrix(probability_matrix, algorithms=algorithms, ax=ax_prob)
        plt.title("Probability of Improvement P(X > Y)")
        plt.savefig("./probability_of_improvement.pdf", bbox_inches="tight")
        print("Saved probability matrix to probability_of_improvement.pdf")
        plt.show()

        # --- 2. Aggregate Metrics Plot (Recommended) ---
        print("\nGenerating Aggregate Metrics Plot (IQM)...")
        aggregate_func = lambda x: np.array([rly.get_interval_estimates(x, rly.IQM)])
        aggregate_scores, aggregate_interval_estimates = rly.get_interval_estimates(
            scores, aggregate_func
        )
        
        fig_agg, ax_agg = plt.subplots(figsize=(7, 5))
        rly_plot.plot_interval_estimates(
            aggregate_scores,
            aggregate_interval_estimates,
            metric_names=['IQM of Cosine Similarity'],
            algorithms=algorithms,
            ax=ax_agg
        )
        plt.title("Interquartile Mean (IQM) of Gradient Quality")
        plt.ylabel("Cosine Similarity")
        plt.savefig("./aggregate_metrics.pdf", bbox_inches="tight")
        print("Saved aggregate metrics plot to aggregate_metrics.pdf")
        plt.show()