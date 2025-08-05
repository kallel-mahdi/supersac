"""
Centralized styling configuration for all plots.
Uses Seaborn with your preferred font sizes and styling.
"""

import matplotlib.pyplot as plt
import seaborn as sns

# === Apply Seaborn Base Style ===
sns.set_theme(style="whitegrid")  # Clean white background with subtle grid

# === Your Preferred Font Configuration ===
# Based on your original PLOTS.ipynb settings
PLOT_PARAMS = {
    # Figure settings
    "figure.figsize": (24, 12),   # Good size for 2x3 grid
    "figure.dpi": 150,            # Higher resolution for crisp output
    "figure.facecolor": "white",
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.2,    # Your original padding
    
    # Font sizes for scientific plots (2x3 grid)
    "font.size": 18,              # Base font size
    "axes.titlesize": 22,         # Subplot title
    "axes.labelsize": 20,         # X/Y axis labels
    "xtick.labelsize": 18,        # X-axis tick labels
    "ytick.labelsize": 18,        # Y-axis tick labels
    "legend.fontsize": 24,        # Legend text
    
    # Line and marker styles
    "lines.linewidth": 2.5,       # Your original
    "lines.markersize": 8,        # Your original
    
    # Grid settings - your original style
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    
    # Legend settings - clean white background
    "legend.frameon": True,
    "legend.framealpha": 1.0,  # Fully opaque white
    "legend.edgecolor": "white",  # White edge
    "legend.facecolor": "white",  # White background
    
    # Clean spines
    "axes.spines.top": False,
    "axes.spines.right": False,
}

# === LaTeX Configuration ===
# Enable LaTeX rendering for proper mathematical symbols
try:
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "text.latex.preamble": r"\usepackage{amsmath} \usepackage{amssymb}"
    })
    print("✅ LaTeX rendering enabled")
except Exception as e:
    print(f"⚠️  LaTeX rendering failed: {e}")
    print("   Falling back to Unicode symbols")
    # Fallback to Unicode symbols if LaTeX fails
    plt.rcParams.update({
        "text.usetex": False,
        "font.family": "DejaVu Sans"
    })
else:
    # If LaTeX is not available, use Unicode symbols
    plt.rcParams.update({
        "text.usetex": False,
        "font.family": "DejaVu Sans"
    })

# === Apply the configuration ===
plt.rcParams.update(PLOT_PARAMS)

# === Color Palettes ===
# Consistent colors across all plots
ALGORITHM_COLORS = {
    "PPO+": "#2E8B8B",              # Deep Teal
    "+ Min target": "#8E44AD",      # Rich Purple
    "- Off-policy data": "#FF8C42", # Warm Orange  
    "- LayerNorm": "#E74C3C",       # Coral Red
    "- Bounded actions": "#3498DB", # Bright Blue
    "- Entropy": "#F1C40F",         # Pure Yellow
    "PPO": "#8E44AD",               # Rich Purple
    "PPO(ours)": "#FF8C42",         # Warm Orange  
    "SAC": "#E74C3C",               # Coral Red
    "TRPO": "#3498DB",              # Bright Blue
    # Add colors for norm ablations
    "-Obs norm": "#FF8C42",         # Warm Orange
    "-Reward norm": "#2E8B8B",      # Deep Teal
}

# Custom vibrant palette for violin plots etc.
CUSTOM_PALETTE = ["#2E8B8B", "#FF8C42", "#8E44AD", "#E74C3C"]

# === Utility Functions ===
def apply_spine_styling(ax):
    """Apply consistent spine styling to an axes object."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#7F8C8D')
    ax.spines['bottom'].set_color('#7F8C8D')
    ax.tick_params(axis='both', which='major', labelcolor='#34495E')
    ax.xaxis.label.set_color('#2C3E50')
    ax.yaxis.label.set_color('#2C3E50')

def create_publication_ready_figure(figsize=None, nrows=1, ncols=1):
    """Create a figure with publication-ready settings and white background."""
    # Use default figsize from rcParams if not provided
    if figsize is None:
        figsize = plt.rcParams['figure.figsize']
        
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    fig.patch.set_facecolor('white')
    fig.set_facecolor('white')
    
    # Handle single axis case
    if nrows * ncols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    # Apply styling to all axes
    for ax in axes:
        ax.set_facecolor('white')
        apply_spine_styling(ax)
    
    return fig, axes

def create_shared_legend_from_algorithms(fig, algorithms, algorithm_colors,y_pos=1.02, ncol=None):
    """Create a shared legend from algorithm names and colors."""
    import matplotlib.patches as mpatches
    
    legend_elements = []
    for algo in algorithms:
        if algo in algorithm_colors:
            legend_elements.append(
                mpatches.Patch(color=algorithm_colors[algo], label=algo)
            )
    
    if legend_elements:
        if ncol is None:
            ncol = len(legend_elements)
        
        fig.legend(handles=legend_elements, 
                  loc='upper center', 
                  bbox_to_anchor=(0.5, y_pos),
                  ncol=ncol,
                  fontsize=plt.rcParams['legend.fontsize'],  # Use global setting
                  frameon=True,
                  fancybox=True,
                  shadow=True,
                  columnspacing=1.5,
                  handletextpad=0.8,
                  handlelength=2.0,
                  borderaxespad=0.5,
                  facecolor='white',
                  edgecolor='white')
    
    return legend_elements

def setup_shared_legend(fig, handles, labels, ncol=None,font_size=None, bbox_y=1.02):
    """Create a professionally styled shared legend with white background."""
    if not handles:
        return
    
    if ncol is None:
        ncol = len(labels)
    
    return fig.legend(
        handles, labels,
        loc='upper center',
        bbox_to_anchor=(0.5, bbox_y),  # Customizable y-position with good default
        ncol=ncol,
        fontsize=plt.rcParams['legend.fontsize'] if font_size is None else font_size,  # Use global setting
        frameon=True,
        fancybox=True,
        shadow=True,
        columnspacing=1.5,
        handletextpad=0.8,
        handlelength=2.0,
        borderaxespad=0.5,
        facecolor='white',  # Explicit white background
        edgecolor='white'   # Explicit white edge
    )

def save_publication_figure(fig, path_without_extension):
    """Save figure in both PDF and PNG formats with high quality."""
    import os
    os.makedirs(os.path.dirname(path_without_extension), exist_ok=True)
    
    # Save PDF for publications
    fig.savefig(f"{path_without_extension}.pdf", 
                format="pdf", bbox_inches="tight", dpi=300, facecolor='white')
    
    # Save PNG for presentations/web
    fig.savefig(f"{path_without_extension}.png", 
                format="png", bbox_inches="tight", dpi=300, facecolor='white')

def set_axis_labels(ax, xlabel, ylabel, title=None):
    """Set axis labels with consistent styling."""
    ax.set_xlabel(xlabel, fontsize=plt.rcParams['axes.labelsize'])  # Use global setting
    ax.set_ylabel(ylabel, fontsize=plt.rcParams['axes.labelsize'])  # Use global setting
    if title:
        ax.set_title(title, fontsize=plt.rcParams['axes.titlesize'])  # Use global setting

print("✅ Plot styling configured with your preferred Seaborn + font settings")
print("   Clean white background, no gray, readable fonts for conferences")
