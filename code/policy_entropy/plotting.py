"""
Policy-entropy visualization utilities for per-hop trends, distributions, and baseline-comparison diagnostics.
"""

import logging
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from policy_entropy.artifacts import (
    aggregate_policy_entropy_batch_results,
    save_policy_entropy_outputs,
)

from typing import Dict, Any, List

_DEFAULT_FIGSIZE = (7.0, 4.5)
_DEFAULT_GRID_ALPHA = 0.3


def _finalize_plot(
    fig: plt.Figure,
    save_path: str,
    logger: logging.Logger = logging.getLogger(__name__),
) -> None:
    """Apply layout, save a figure to disk, close it, and log the output path."""
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved plot to {save_path}")


def _hop_indices(
    length: int
) -> np.ndarray:
    """Return one-based hop indices for plotting along the x-axis."""
    return np.arange(1, length + 1)


def _configure_axes(
    ax: plt.Axes,
    *,
    xlabel: str,
    ylabel: str,
    title: str,
    grid_alpha: float = _DEFAULT_GRID_ALPHA,
) -> None:
    """Set common axis labels/title and enable a lightly transparent grid."""
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=grid_alpha)


def _plot_histogram(
    values: np.ndarray,
    save_path: str,
    *,
    xlabel: str,
    ylabel: str,
    title: str,
    bins: int = 40,
    log_y: bool = False,
    vline_at_zero: bool = False,
) -> None:
    """Create and save a standardized histogram with optional log scale or zero marker."""
    fig, ax = plt.subplots(figsize=_DEFAULT_FIGSIZE)
    ax.hist(values, bins=bins)
    if vline_at_zero:
        ax.axvline(0.0, linestyle='--', linewidth=1.0)
    _configure_axes(ax, xlabel=xlabel, ylabel=ylabel, title=title)
    if log_y:
        ax.set_yscale('log')
    _finalize_plot(fig, save_path)


def plot_per_hop_entropy(
    summary: Dict[str, Any], 
    save_path: str, 
    title_prefix: str = "Policy Entropy"
) -> None:
    """Plot mean per-hop policy entropy with error bars from standard deviations."""
    means = np.asarray(summary["per_hop_mean_entropy_bits"], dtype=np.float32)
    stds = np.asarray(summary["per_hop_std_entropy_bits"], dtype=np.float32)
    hops = _hop_indices(len(means))

    fig, ax = plt.subplots(figsize=_DEFAULT_FIGSIZE)
    ax.errorbar(hops, means, yerr=stds, marker='o', capsize=4)
    ax.set_xticks(hops)
    _configure_axes(
        ax,
        xlabel="Hop",
        ylabel="Entropy (bits)",
        title=f"{title_prefix}: Per-hop entropy",
    )
    _finalize_plot(fig, save_path)


def plot_cumulative_entropy(
    summary: Dict[str, Any], 
    save_path: str, 
    title_prefix: str = "Policy Entropy"
) -> None:
    """Plot cumulative entropy across hops using per-hop mean entropy values."""
    means = np.asarray(summary["per_hop_mean_entropy_bits"], dtype=np.float32)
    cumulative = np.cumsum(means)
    hops = _hop_indices(len(means))

    fig, ax = plt.subplots(figsize=_DEFAULT_FIGSIZE)
    ax.plot(hops, cumulative, marker='o')
    ax.set_xticks(hops)
    _configure_axes(
        ax,
        xlabel="Hop",
        ylabel="Cumulative entropy (bits)",
        title=f"{title_prefix}: Cumulative path cost",
    )
    _finalize_plot(fig, save_path)


def plot_path_entropy_histogram(
    aggregate: Dict[str, np.ndarray], 
    save_path: str, 
    title_prefix: str = "Policy Entropy"
) -> None:
    """Plot the distribution of path-level entropy values over all sampled paths."""
    values = aggregate["all_path_entropies_flat"]
    _plot_histogram(
        values,
        save_path,
        xlabel="Path entropy (bits)",
        ylabel="Count",
        title=f"{title_prefix}: Path entropy distribution",
        log_y=True,
    )


def plot_question_entropy_histogram(
    aggregate: Dict[str, np.ndarray], 
    save_path: str, 
    title_prefix: str = "Policy Entropy"
) -> None:
    """Plot the distribution of question-level entropy values across the dataset."""
    values = aggregate["all_question_entropies_flat"]
    _plot_histogram(
        values,
        save_path,
        xlabel="Question entropy (bits)",
        ylabel="Count",
        title=f"{title_prefix}: Question entropy distribution",
        log_y=True,
    )


def plot_step_entropy_boxplot(
    aggregate: Dict[str, np.ndarray], 
    save_path: str, 
    title_prefix: str = "Policy Entropy"
) -> None:
    """Show per-hop entropy spread via boxplots aggregated across questions and rollouts."""
    step_entropies = aggregate["all_step_entropies"]  # [Nq, R, T]
    T = step_entropies.shape[2]
    by_hop = [step_entropies[:, :, t].reshape(-1) for t in range(T)]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.boxplot(by_hop, showfliers=False)
    _configure_axes(
        ax,
        xlabel="Hop",
        ylabel="Entropy (bits)",
        title=f"{title_prefix}: Step entropy by hop",
    )
    ax.set_xticks(np.arange(1, T + 1))
    _finalize_plot(fig, save_path)

def plot_entropy_vs_surprisal_scatter(
    aggregate: Dict[str, np.ndarray],
    save_path: str,
    title_prefix: str = "Policy Entropy",
    max_points: int = 20000,
    seed: int = 0,
) -> None:
    """Scatter plot policy entropy vs sampled-action surprisal, with optional subsampling."""
    entropy = aggregate["all_step_entropies_flat"]
    surprisal = aggregate["all_action_surprisals_flat"]

    if entropy.size > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(entropy.size, size=max_points, replace=False)
        entropy = entropy[idx]
        surprisal = surprisal[idx]

    fig, ax = plt.subplots(figsize=(6.0, 6.0))
    ax.scatter(entropy, surprisal, s=8, alpha=0.2)
    max_val = float(max(entropy.max(initial=0.0), surprisal.max(initial=0.0)))
    ax.plot([0.0, max_val], [0.0, max_val], linestyle='--', linewidth=1.0)
    _configure_axes(
        ax,
        xlabel="Policy entropy (bits)",
        ylabel="Sampled action surprisal (bits)",
        title=f"{title_prefix}: Entropy vs surprisal",
    )
    _finalize_plot(fig, save_path)


def plot_entropy_vs_identifier_bits(
    summary: Dict[str, Any],
    save_path: str,
    title_prefix: str = "Policy Entropy",
    use_fixed_width: bool = False,
) -> None:
    """Compare per-hop policy entropy against local identifier coding-cost baselines."""
    entropy = np.asarray(summary["per_hop_mean_entropy_bits"], dtype=np.float32)
    if use_fixed_width:
        identifier = np.asarray(summary["per_hop_mean_fixed_identifier_bits"], dtype=np.float32)
        ylabel = "Fixed-width local index cost (bits)"
        title = f"{title_prefix}: Entropy vs fixed-width identifier cost"
    else:
        identifier = np.asarray(summary["per_hop_mean_ideal_identifier_bits"], dtype=np.float32)
        ylabel = "Uniform local identifier cost (bits)"
        title = f"{title_prefix}: Entropy vs local identifier cost"

    hops = _hop_indices(len(entropy))

    fig, ax = plt.subplots(figsize=_DEFAULT_FIGSIZE)
    ax.plot(hops, entropy, marker='o', label='Policy entropy')
    ax.plot(hops, identifier, marker='s', label='Identifier baseline')
    ax.set_xticks(hops)
    _configure_axes(ax, xlabel="Hop", ylabel=ylabel, title=title)
    ax.legend()
    _finalize_plot(fig, save_path)


def plot_per_hop_communication_savings(
    summary: Dict[str, Any],
    save_path: str,
    title_prefix: str = "Policy Entropy",
    use_fixed_width: bool = False,
) -> None:
    """Plot per-hop communication savings relative to the selected identifier baseline."""
    if use_fixed_width:
        savings = np.asarray(summary["per_hop_mean_fixed_savings_bits"], dtype=np.float32)
        title = f"{title_prefix}: Per-hop savings vs fixed-width local index"
    else:
        savings = np.asarray(summary["per_hop_mean_ideal_savings_bits"], dtype=np.float32)
        title = f"{title_prefix}: Per-hop savings vs local identifier"

    hops = _hop_indices(len(savings))

    fig, ax = plt.subplots(figsize=_DEFAULT_FIGSIZE)
    ax.bar(hops, savings)
    ax.axhline(0.0, linestyle='--', linewidth=1.0)
    ax.set_xticks(hops)
    _configure_axes(ax, xlabel="Hop", ylabel="Savings (bits)", title=title)
    _finalize_plot(fig, save_path)


def plot_savings_histogram(
    aggregate: Dict[str, np.ndarray],
    save_path: str,
    title_prefix: str = "Policy Entropy",
    use_fixed_width: bool = False,
) -> None:
    """Plot the distribution of communication savings over all steps in the dataset."""
    if use_fixed_width:
        values = aggregate["all_fixed_savings_bits_flat"]
        title = f"{title_prefix}: Savings distribution vs fixed-width local index"
    else:
        values = aggregate["all_ideal_savings_bits_flat"]
        title = f"{title_prefix}: Savings distribution vs local identifier"

    _plot_histogram(
        values,
        save_path,
        xlabel="Savings (bits)",
        ylabel="Count",
        title=title,
        vline_at_zero=True,
    )


def generate_policy_entropy_plots(
    summary: Dict[str, Any],
    batch_results: List[Dict[str, Any]],
    output_dir: str,
    run_name: str = "test",
    title_prefix: str = "Policy Entropy",
) -> Dict[str, str]:
    """Generate all policy-entropy plots, persist artifacts, and return output paths."""
    os.makedirs(output_dir, exist_ok=True)
    aggregate = aggregate_policy_entropy_batch_results(batch_results)

    save_policy_entropy_outputs(summary, aggregate, output_dir, run_name)

    paths = {
        "per_hop_entropy": os.path.join(output_dir, f"{run_name}_per_hop_entropy.png"),
        "cumulative_entropy": os.path.join(output_dir, f"{run_name}_cumulative_entropy.png"),
        "path_entropy_histogram": os.path.join(output_dir, f"{run_name}_path_entropy_histogram.png"),
        "question_entropy_histogram": os.path.join(output_dir, f"{run_name}_question_entropy_histogram.png"),
        "step_entropy_boxplot": os.path.join(output_dir, f"{run_name}_step_entropy_boxplot.png"),
        "entropy_vs_surprisal": os.path.join(output_dir, f"{run_name}_entropy_vs_surprisal.png"),
        "entropy_vs_identifier": os.path.join(output_dir, f"{run_name}_entropy_vs_identifier.png"),
        "entropy_vs_fixed_identifier": os.path.join(output_dir, f"{run_name}_entropy_vs_fixed_identifier.png"),
        "per_hop_savings": os.path.join(output_dir, f"{run_name}_per_hop_savings.png"),
        "per_hop_fixed_savings": os.path.join(output_dir, f"{run_name}_per_hop_fixed_savings.png"),
        "savings_histogram": os.path.join(output_dir, f"{run_name}_savings_histogram.png"),
        "fixed_savings_histogram": os.path.join(output_dir, f"{run_name}_fixed_savings_histogram.png"),
    }

    plot_per_hop_entropy(summary, paths["per_hop_entropy"], title_prefix=title_prefix)
    plot_cumulative_entropy(summary, paths["cumulative_entropy"], title_prefix=title_prefix)
    plot_path_entropy_histogram(aggregate, paths["path_entropy_histogram"], title_prefix=title_prefix)
    plot_question_entropy_histogram(aggregate, paths["question_entropy_histogram"], title_prefix=title_prefix)
    plot_step_entropy_boxplot(aggregate, paths["step_entropy_boxplot"], title_prefix=title_prefix)
    plot_entropy_vs_surprisal_scatter(aggregate, paths["entropy_vs_surprisal"], title_prefix=title_prefix)
    plot_entropy_vs_identifier_bits(summary, paths["entropy_vs_identifier"], title_prefix=title_prefix, use_fixed_width=False)
    plot_entropy_vs_identifier_bits(summary, paths["entropy_vs_fixed_identifier"], title_prefix=title_prefix, use_fixed_width=True)
    plot_per_hop_communication_savings(summary, paths["per_hop_savings"], title_prefix=title_prefix, use_fixed_width=False)
    plot_per_hop_communication_savings(summary, paths["per_hop_fixed_savings"], title_prefix=title_prefix, use_fixed_width=True)
    plot_savings_histogram(aggregate, paths["savings_histogram"], title_prefix=title_prefix, use_fixed_width=False)
    plot_savings_histogram(aggregate, paths["fixed_savings_histogram"], title_prefix=title_prefix, use_fixed_width=True)

    return paths
