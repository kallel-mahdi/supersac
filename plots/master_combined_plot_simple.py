import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import sys
import os

# Add the plots directory to path so we can import the individual scripts
sys.path.append(os.path.dirname(__file__))

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

# === CONSISTENT STYLING CONFIGURATION ===
sns.set_theme(style="ticks", rc={"font.family": "serif"})

# Unified font sizes and styling
TITLE_FONTSIZE = 16
LABEL_FONTSIZE = 14  
TICK_FONTSIZE = 12
LEGEND_FONTSIZE = 14  # Increased legend font size

plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': TITLE_FONTSIZE,
    'axes.labelsize': LABEL_FONTSIZE,
    'xtick.labelsize': TICK_FONTSIZE,
    'ytick.labelsize': TICK_FONTSIZE,
    'legend.fontsize': LEGEND_FONTSIZE,
    'lines.linewidth': 2.5,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
})

# === ALGORITHM COLORS (Consistent across all plots) ===
ALGORITHM_COLORS = {
    "PPO+": "#2E8B8B",              # Deep Teal
    "- Off-policy data": "#FF8C42", # Warm Orange  
    "Min Target": "#8E44AD",        # Rich Purple
    "- LayerNorm": "#E74C3C",       # Coral Red
    "- Bounded actions": "#27AE60", # Green
    "- Entropy": "#F39C12"          # Orange-Yellow
}

# === GLOBAL Y-AXIS ORDER (Consistent across all plots) ===
GLOBAL_Y_AXIS_ORDER = [
    "PPO+",                 # Baseline (appears in cosine & bias)
    "- Off-policy data",    # Appears in all plots  
    "Min Target",           # Appears in all plots
    "- LayerNorm",         # Appears in all plots
    "- Bounded actions",   # Only in ablation
    "- Entropy"            # Only in ablation
]

def create_combined_plot():
    """Create the combined three-panel plot with shared legend."""
    
    # Create figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(21, 7))
    fig.patch.set_facecolor('white')
    
    print("Generating combined plot...")
    print("=" * 50)
    
    # Generate each plot with consistent styling
    print("1. Generating ablation plot...")
    algorithms_ablation = generate_ablation_plot(
        ax=ax1, 
        algorithm_colors=ALGORITHM_COLORS, 
        show_legend=False,
        title_fontsize=TITLE_FONTSIZE,
        label_fontsize=LABEL_FONTSIZE,
        y_axis_order=GLOBAL_Y_AXIS_ORDER
    )
    print(f"   Ablation algorithms: {algorithms_ablation}")
    
    print("\n2. Generating cosine similarity plot...")
    algorithms_cosine = generate_cosine_plot(
        ax=ax2, 
        algorithm_colors=ALGORITHM_COLORS, 
        show_legend=False,
        title_fontsize=TITLE_FONTSIZE,
        label_fontsize=LABEL_FONTSIZE,
        y_axis_order=GLOBAL_Y_AXIS_ORDER
    )
    print(f"   Cosine algorithms: {algorithms_cosine}")
    
    print("\n3. Generating bias violin plot...")
    algorithms_bias = generate_bias_plot(
        ax=ax3, 
        algorithm_colors=ALGORITHM_COLORS, 
        show_legend=False,
        title_fontsize=TITLE_FONTSIZE,
        label_fontsize=LABEL_FONTSIZE,
        y_axis_order=GLOBAL_Y_AXIS_ORDER
    )
    print(f"   Bias algorithms: {algorithms_bias}")
    
    # Collect all unique algorithms across all plots
    all_algorithms = set()
    for algo_list in [algorithms_ablation, algorithms_cosine, algorithms_bias]:
        if algo_list:
            all_algorithms.update(algo_list)
    
    print(f"\nAll algorithms found: {sorted(all_algorithms)}")
    
    # Create shared legend with consistent ordering based on global order
    legend_elements = []
    # Use global order, but only include algorithms that appear in plots
    for algo in GLOBAL_Y_AXIS_ORDER:
        if algo in all_algorithms and algo in ALGORITHM_COLORS:
            legend_elements.append(
                mpatches.Patch(color=ALGORITHM_COLORS[algo], label=algo)
            )
    
    # Position shared legend at the top
    if legend_elements:
        fig.legend(handles=legend_elements, 
                  loc='upper center', 
                  bbox_to_anchor=(0.5, 0.98),
                  ncol=len(legend_elements),
                  fontsize=LEGEND_FONTSIZE,
                  frameon=True,
                  fancybox=True,
                  shadow=True)
        print(f"Created shared legend with {len(legend_elements)} algorithms")
    
    # Apply consistent styling to all subplots
    for ax in [ax1, ax2, ax3]:
        # Remove top and right spines for cleaner look
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#7F8C8D')
        ax.spines['bottom'].set_color('#7F8C8D')
        
        # Ensure consistent tick label sizes
        ax.tick_params(axis='both', which='major', labelsize=TICK_FONTSIZE)
    
    # Adjust layout to accommodate legend
    plt.subplots_adjust(top=0.85, left=0.06, right=0.94, bottom=0.1, wspace=0.3)
    
    # Save the combined plot
    output_path = "./combined_three_panel_plot.pdf"
    print(f"\nSaving combined plot to {output_path}")
    plt.savefig(output_path, format="pdf", bbox_inches="tight", dpi=300, facecolor='white')
    
    # Also save as PNG for easier viewing
    png_path = "./combined_three_panel_plot.png"
    print(f"Saving combined plot to {png_path}")
    plt.savefig(png_path, format="png", bbox_inches="tight", dpi=300, facecolor='white')
    
    plt.show()
    
    print("=" * 50)
    print("Combined plot generation complete!")
    
    return fig

if __name__ == "__main__":
    print("Starting combined plot generation...")
    print("Note: Using fast test mode for ablation plot (2 environments instead of 6)")
    create_combined_plot()