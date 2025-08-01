import matplotlib.pyplot as plt
import seaborn as sns
import jax.numpy as np
import jax.random as random
import jax
import jax.numpy as jnp
import pandas as pd
import wandb
import os
import itertools

# Import centralized styling
from style import (
    create_publication_ready_figure,
    set_axis_labels,
    save_publication_figure
)


# Configure JAX
jax.config.update("jax_enable_x64", True)

def generate_tanh_data():
    """Generate data for the tanh distribution plots."""
    n = 1_000_000
    key = random.PRNGKey(0)
    original_sample = random.normal(key, shape=(n,))
    
    data = original_sample * 0.5 + 0.5
    data2 = data + 0.1
    
    data_bounded = np.clip(data, -1, 1)
    data_bounded2 = np.clip(data2, -1, 1)
    
    data_tanh = np.tanh(data)
    data_tanh2 = np.tanh(data2)
    
    return data, data_bounded, data_tanh, data_bounded2, data_tanh2

# def compute_kl_divergence(data, data2):
#     """Compute KL divergence between two datasets."""
#     bins = np.linspace(min(np.min(data), np.min(data2)),
#                        max(np.max(data), np.max(data2)), 50)
    
#     h1, _ = np.histogram(data, bins=bins, density=True)
#     h2, _ = np.histogram(data2, bins=bins, density=True)
    
#     h2 = 1e-10 * np.ones_like(h2)
#     bin_centers = (bins[:-1] + bins[1:]) / 2
    
#     kl_divergence = np.sum(h1 * np.log(h1/h2))
    
#     return bin_centers, h1, h2, kl_divergence

def compute_kl_divergence(p_samples, q_samples, n_bins=100):
    """
    Estimates the Total Variation (TV) distance between two distributions from samples.

    The TV distance is estimated by discretizing the sample space with histograms
    and computing the distance between the resulting probability mass functions.

    Args:
        p_samples (np.ndarray): A 1D array of samples from distribution P.
        q_samples (np.ndarray): A 1D array of samples from distribution Q.
        n_bins (int): The number of bins to use for the histogram approximation.
                      The accuracy of the estimate depends on this choice.

    Returns:
        float: The estimated Total Variation distance, a value between 0 and 1.
    """
    # 1. Define a common set of bins for both distributions.
    # The range of bins should cover all samples from both P and Q.
    min_val = min(p_samples.min(), q_samples.min())
    max_val = max(p_samples.max(), q_samples.max())
    bins = np.linspace(min_val, max_val, num=n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # 2. Compute the histogram counts for each set of samples.
    p_probs, _ = np.histogram(p_samples, bins=bins,density=True)
    q_probs, _ = np.histogram(q_samples, bins=bins,density=True)



    # 4. Calculate the Total Variation distance.
    # d_TV(P, Q) = 0.5 * sum(|P(x) - Q(x)|)
    tv_distance = 0.5 * np.sum(np.abs(p_probs - q_probs))
    
    return bin_centers,p_probs,q_probs,tv_distance





def generate_kl_evolution_data():
    """Generate data for KL evolution plot."""
    ratios = []
    means = np.linspace(0, 1., 51)
    stds = np.linspace(0.1, 1., 51)[::-1]
    
    list_means = [0., 0.3, 0.6, 0.9]
    bins, hist1, hist2 = [], [], []
    
    #n = 250_000_000
    n = 1000
    key = random.PRNGKey(0)
    original_sample = random.normal(key, shape=(n,))
    
    for mean, _ in zip(means[:-1], stds[:-1]):
        std = 0.2
        data = original_sample * std + mean
        data2 = data + 0.02
        
        data_bounded = np.clip(data, -1, 1)
        data_bounded2 = np.clip(data2, -1, 1)
        
        bin_centers, h1, h2, max_ratio = compute_kl_divergence(data_bounded, data_bounded2)
        ratios.append(max_ratio)
        
        if round(mean, 3) in list_means:
            hist1.append(h1)
            hist2.append(h2)
            bins.append(bin_centers)
    
    return means[:-1], ratios, bins, hist1, hist2

def fetch_oob_data():
    """Fetch out-of-bounds data from W&B."""
    os.environ["WANDB_API_KEY"] = "28996bd59f1ba2c5a8c3f2cc23d8673c327ae230"
    api = wandb.Api()
    entity = "mahdikallel"
    project_name = "PPO_OUTBOUND"
    runs = api.runs(entity + "/" + project_name)
    
    # Select one run from the list
    run = runs[0]
    history = run.history(samples=1000000).dropna(subset=["training/out_of_bound_percentage"])
    
    return history

def create_distribution_plots():
    """Create the three distribution plots side by side."""
    # Create figure with 3 subplots
    fig, axes = create_publication_ready_figure(figsize=(24, 8), nrows=1, ncols=3)
    
    # Generate data
    data, data_bounded, data_tanh, data_bounded2, data_tanh2 = generate_tanh_data()
    means, ratios, bins, hist1, hist2 = generate_kl_evolution_data()
    history = fetch_oob_data()
    
    # Plot A: Probability density function
    ax1 = axes[0]
    
    sns.kdeplot(np.array(data), fill=True, alpha=0.5, label='Gaussian', 
                color=sns.color_palette("Blues", 1)[0], linewidth=2, ax=ax1)
    sns.kdeplot(np.array(data_bounded), fill=True, alpha=0.5, label='Clipped Gaussian', 
                color=sns.color_palette("Oranges", 1)[0], linewidth=2, ax=ax1)
    sns.kdeplot(np.array(data_tanh), fill=True, alpha=0.5, label='Tanh Gaussian', 
                color=sns.color_palette("Greens", 1)[0], linewidth=2, ax=ax1)
    
    set_axis_labels(ax1, 'x', 'Density', 'Probability density function')
    ax1.legend(loc='upper left')
    ax1.set_ylim(0, 3)
    ax1.grid(True, alpha=0.3, linestyle='--', color='#BDC3C7', zorder=0)
    
    # Add label A
    ax1.text(-0.1, 1.05, 'A', transform=ax1.transAxes, fontsize=24, fontweight='bold')
    
    # Plot B: Evolution of effective KL
    ax2 = axes[1]
    
    num_curves = len(bins)
    blue_palette = sns.color_palette("Blues", num_curves)
    
    for i, (bin_centers, h1_hist, h2_hist) in enumerate(zip(bins, hist1, hist2)):
        pdf_color = blue_palette[i]
        ax2.fill_between(bin_centers, h1_hist, color=pdf_color, alpha=0.3)
        ax2.plot(bin_centers, h1_hist, color=pdf_color, lw=4)
    
    set_axis_labels(ax2, '$\mu$', 'Density', 'Evolution of effective KL as $\mu \\to 1$')
    ax2.tick_params(axis='y', colors=blue_palette[-1])
    ax2.set_xlim(0, 1)
    ax2.grid(True, alpha=0.3, linestyle='--', color='#BDC3C7', zorder=0)
    
    kl_color = sns.color_palette("Oranges", 1)[0]
    ax2_twin = ax2.twinx()
    ax2_twin.plot(means, ratios, marker='o', lw=2, color=kl_color)
    ax2_twin.set_ylabel('True KL', color=kl_color)
    ax2_twin.tick_params(axis='y', colors=kl_color)
    
    line_color = sns.color_palette("Greens", 1)[0]
    _, _, _, kl_ref = compute_kl_divergence(data_bounded, data_bounded2)
    ax2_twin.axhline(kl_ref, color=line_color, linestyle='--', lw=2)
    x_mid = means[len(means) // 2]
    ax2_twin.text(x_mid, kl_ref*1.01, "Unbounded KL ($\\delta$)", 
                  color=line_color, ha='center', va='bottom')
    
    # Add legend for the blue curves
    from matplotlib.lines import Line2D
    legend_elements = []
    for i, val in enumerate([0.0, 0.3, 0.6, 0.9]):
        legend_elements.append(Line2D([0], [0], color=blue_palette[i], lw=4, 
                                     label=f'$\mu = {val}$'))
    ax2.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.0, 0.85))
    
    # Add label B
    ax2.text(-0.1, 1.05, 'B', transform=ax2.transAxes, fontsize=24, fontweight='bold')
    
    # Plot C: Evolution of out of bounds actions
    ax3 = axes[2]
    
    # Plot OOB error on the first y-axis
    color1 = 'tab:blue'
    ax3.set_xlabel('Million steps')
    ax3.set_ylabel('OOB error', color=color1)
    ax3.plot(history["_step"]/1e6, history["OOB_error"], marker='o', color=color1, label="Average OOB error")
    ax3.tick_params(axis='y', labelcolor=color1)
    ax3.tick_params(axis='x')
    ax3.grid(True, alpha=0.3, linestyle='--', color='#BDC3C7', zorder=0)
    
    # Create a second y-axis that shares the same x-axis
    ax3_twin = ax3.twinx()
    color2 = 'tab:orange'
    ax3_twin.set_ylabel('Percentage of OOB actions', color=color2)
    ax3_twin.plot(history["_step"]/1e6, history["training/out_of_bound_percentage"], 
                  marker='o', color=color2, label="Percentage of OOB actions")
    ax3_twin.tick_params(axis='y', labelcolor=color2)
    
    # Add title
    ax3.set_title("Evolution of out of bounds actions")
    
    # Add legend for both axes
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_twin.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper left', bbox_to_anchor=(0.0, 0.96))
    
    # Add label C
    ax3.text(-0.1, 1.05, 'C', transform=ax3.transAxes, fontsize=24, fontweight='bold')
    
    # Adjust layout
    plt.subplots_adjust(wspace=0.3)
    
    # Save the figure
    save_publication_figure(fig, "./plots/distrib")
    plt.show()
    
    print("Distribution plots saved!")

if __name__ == "__main__":
    print("Creating distribution plots...")
    create_distribution_plots() 