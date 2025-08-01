import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import sys
import os

# Add the plots directory to path so we can import the individual scripts
sys.path.append(os.path.dirname(__file__))

# Import centralized styling
from style import (
    ALGORITHM_COLORS, 
    create_publication_ready_figure,
    create_shared_legend_from_algorithms,
    save_publication_figure
)

# Import the plotting functions from the modified individual scripts
from ablation_rliable import generate_ablation_plot
from bias import generate_bias_plot

# For cosine script, we need to handle the space in filename
import importlib.util
cosine_spec = importlib.util.spec_from_file_location("cosine_rliable_copy", 
                                                   os.path.join(os.path.dirname(__file__), "cosine_rliable copy.py"))
cosine_module = importlib.util.module_from_spec(cosine_spec)
cosine_spec.loader.exec_module(cosine_module)
generate_cosine_plot = cosine_module.generate_cosine_plot

# === FAST TEST MODE ===
FAST_TEST = False  # Use only 2 environments for faster testing

# === GLOBAL Y-AXIS ORDER (Consistent across all plots) ===
GLOBAL_Y_AXIS_ORDER = [
    "PPO+",                 # Baseline (appears in cosine & bias)
    "+ Min target",         # Addition to PPO+ (appears in all plots)
    "- Off-policy data",    # Appears in all plots  
    "- LayerNorm",         # Appears in all plots
    "- Bounded actions",   # Only in ablation
    "- Entropy"            # Only in ablation
]

def create_combined_plot():
    """Create the combined three-panel plot with shared legend."""
    
    # Create figure with 3 subplots using utility function
    fig, (ax1, ax2, ax3) = create_publication_ready_figure(figsize=(21, 7), nrows=1, ncols=3)
    
    print("Generating combined plot...")
    print("=" * 50)
    
    # Generate each plot with consistent styling
    print("1. Generating ablation plot...")
    algorithms_ablation = generate_ablation_plot(
        ax=ax1, 
        algorithm_colors=ALGORITHM_COLORS, 
        show_legend=False,
        y_axis_order=GLOBAL_Y_AXIS_ORDER,
        fast_test=FAST_TEST
    )
    print(f"   Ablation algorithms: {algorithms_ablation}")
    
    print("\n2. Generating cosine similarity plot...")
    algorithms_cosine = generate_cosine_plot(
        ax=ax2, 
        algorithm_colors=ALGORITHM_COLORS, 
        show_legend=False,
        y_axis_order=GLOBAL_Y_AXIS_ORDER,
        fast_test=FAST_TEST
    )
    print(f"   Cosine algorithms: {algorithms_cosine}")
    
    print("\n3. Generating bias violin plot...")
    algorithms_bias = generate_bias_plot(
        ax=ax3, 
        algorithm_colors=ALGORITHM_COLORS, 
        show_legend=False,
        y_axis_order=GLOBAL_Y_AXIS_ORDER,
        fast_test=FAST_TEST
    )
    print(f"   Bias algorithms: {algorithms_bias}")
    
    # Collect all unique algorithms across all plots
    all_algorithms = set()
    for algo_list in [algorithms_ablation, algorithms_cosine, algorithms_bias]:
        if algo_list:
            all_algorithms.update(algo_list)
    
    print(f"\nAll algorithms found: {sorted(all_algorithms)}")
    
    # Create shared legend using utility function
    legend_elements = create_shared_legend_from_algorithms(
        fig, 
        [algo for algo in GLOBAL_Y_AXIS_ORDER if algo in all_algorithms], 
        ALGORITHM_COLORS
    )
    print(f"Created shared legend with {len(legend_elements)} algorithms")
    
    # Adjust layout to accommodate legend
    plt.subplots_adjust(top=0.85, left=0.06, right=0.94, bottom=0.12, wspace=0.25)
    
    # Save the combined plot using centralized function
    print(f"\nSaving combined plot...")
    save_publication_figure(fig, "./combined_three_panel_plot")
    
    plt.show()
    
    print("=" * 50)
    print("Combined plot generation complete!")
    
    return fig

if __name__ == "__main__":
    print("Starting combined plot generation...")
    if FAST_TEST:
        print("Note: FAST TEST MODE enabled - using only 2 environments instead of 6 for all plots")
    else:
        print("Note: Using all 6 environments for complete analysis")
    create_combined_plot()