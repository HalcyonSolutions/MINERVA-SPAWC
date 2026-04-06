import json
import logging
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from utils.basic import _jsonify

from typing import Dict, Any, List

def _finalize_plot(
    fig: plt.Figure,
    save_path: str,
    logger: logging.Logger = logging.getLogger(__name__),
) -> None:
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved plot to {save_path}")


def plot_per_hop_entropy(
    summary: Dict[str, Any], 
    save_path: str, 
    title_prefix: str = "Policy Entropy"
) -> None:
    means = np.asarray(summary["per_hop_mean_entropy_bits"], dtype=np.float32)
    stds = np.asarray(summary["per_hop_std_entropy_bits"], dtype=np.float32)
    hops = np.arange(1, len(means) + 1)

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.errorbar(hops, means, yerr=stds, marker='o', capsize=4)
    ax.set_xticks(hops)
    ax.set_xlabel("Hop")
    ax.set_ylabel("Entropy (bits)")
    ax.set_title(f"{title_prefix}: Per-hop entropy")
    ax.grid(True, alpha=0.3)
    _finalize_plot(fig, save_path)


def plot_cumulative_entropy(
    summary: Dict[str, Any], 
    save_path: str, 
    title_prefix: str = "Policy Entropy"
) -> None:
    means = np.asarray(summary["per_hop_mean_entropy_bits"], dtype=np.float32)
    cumulative = np.cumsum(means)
    hops = np.arange(1, len(means) + 1)

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.plot(hops, cumulative, marker='o')
    ax.set_xticks(hops)
    ax.set_xlabel("Hop")
    ax.set_ylabel("Cumulative entropy (bits)")
    ax.set_title(f"{title_prefix}: Cumulative path cost")
    ax.grid(True, alpha=0.3)
    _finalize_plot(fig, save_path)


def plot_path_entropy_histogram(
    aggregate: Dict[str, np.ndarray], 
    save_path: str, 
    title_prefix: str = "Policy Entropy"
) -> None:
    values = aggregate["all_path_entropies_flat"]
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.hist(values, bins=40)
    ax.set_xlabel("Path entropy (bits)")
    ax.set_ylabel("Count")
    ax.set_title(f"{title_prefix}: Path entropy distribution")
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')  # Use log scale for better visibility of tail
    _finalize_plot(fig, save_path)


def plot_question_entropy_histogram(
    aggregate: Dict[str, np.ndarray], 
    save_path: str, 
    title_prefix: str = "Policy Entropy"
) -> None:
    values = aggregate["all_question_entropies_flat"]
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.hist(values, bins=40)
    ax.set_xlabel("Question entropy (bits)")
    ax.set_ylabel("Count")
    ax.set_title(f"{title_prefix}: Question entropy distribution")
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')  # Use log scale for better visibility of tail
    _finalize_plot(fig, save_path)


def plot_step_entropy_boxplot(
    aggregate: Dict[str, np.ndarray], 
    save_path: str, 
    title_prefix: str = "Policy Entropy"
) -> None:
    step_entropies = aggregate["all_step_entropies"]  # [Nq, R, T]
    T = step_entropies.shape[2]
    by_hop = [step_entropies[:, :, t].reshape(-1) for t in range(T)]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.boxplot(by_hop, showfliers=False)
    ax.set_xlabel("Hop")
    ax.set_ylabel("Entropy (bits)")
    ax.set_title(f"{title_prefix}: Step entropy by hop")
    ax.set_xticks(np.arange(1, T + 1))
    ax.grid(True, alpha=0.3)
    _finalize_plot(fig, save_path)

def plot_entropy_vs_surprisal_scatter(
    aggregate: Dict[str, np.ndarray],
    save_path: str,
    title_prefix: str = "Policy Entropy",
    max_points: int = 20000,
    seed: int = 0,
) -> None:
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
    ax.set_xlabel("Policy entropy (bits)")
    ax.set_ylabel("Sampled action surprisal (bits)")
    ax.set_title(f"{title_prefix}: Entropy vs surprisal")
    ax.grid(True, alpha=0.3)
    _finalize_plot(fig, save_path)


def plot_entropy_vs_identifier_bits(
    summary: Dict[str, Any],
    save_path: str,
    title_prefix: str = "Policy Entropy",
    use_fixed_width: bool = False,
) -> None:
    entropy = np.asarray(summary["per_hop_mean_entropy_bits"], dtype=np.float32)
    if use_fixed_width:
        identifier = np.asarray(summary["per_hop_mean_fixed_identifier_bits"], dtype=np.float32)
        ylabel = "Fixed-width local index cost (bits)"
        title = f"{title_prefix}: Entropy vs fixed-width identifier cost"
    else:
        identifier = np.asarray(summary["per_hop_mean_ideal_identifier_bits"], dtype=np.float32)
        ylabel = "Uniform local identifier cost (bits)"
        title = f"{title_prefix}: Entropy vs local identifier cost"

    hops = np.arange(1, len(entropy) + 1)

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.plot(hops, entropy, marker='o', label='Policy entropy')
    ax.plot(hops, identifier, marker='s', label='Identifier baseline')
    ax.set_xticks(hops)
    ax.set_xlabel("Hop")
    ax.set_ylabel("Bits")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    _finalize_plot(fig, save_path)


def plot_per_hop_communication_savings(
    summary: Dict[str, Any],
    save_path: str,
    title_prefix: str = "Policy Entropy",
    use_fixed_width: bool = False,
) -> None:
    if use_fixed_width:
        savings = np.asarray(summary["per_hop_mean_fixed_savings_bits"], dtype=np.float32)
        title = f"{title_prefix}: Per-hop savings vs fixed-width local index"
    else:
        savings = np.asarray(summary["per_hop_mean_ideal_savings_bits"], dtype=np.float32)
        title = f"{title_prefix}: Per-hop savings vs local identifier"

    hops = np.arange(1, len(savings) + 1)

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.bar(hops, savings)
    ax.axhline(0.0, linestyle='--', linewidth=1.0)
    ax.set_xticks(hops)
    ax.set_xlabel("Hop")
    ax.set_ylabel("Savings (bits)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    _finalize_plot(fig, save_path)


def plot_savings_histogram(
    aggregate: Dict[str, np.ndarray],
    save_path: str,
    title_prefix: str = "Policy Entropy",
    use_fixed_width: bool = False,
) -> None:
    if use_fixed_width:
        values = aggregate["all_fixed_savings_bits_flat"]
        title = f"{title_prefix}: Savings distribution vs fixed-width local index"
    else:
        values = aggregate["all_ideal_savings_bits_flat"]
        title = f"{title_prefix}: Savings distribution vs local identifier"

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.hist(values, bins=40)
    ax.axvline(0.0, linestyle='--', linewidth=1.0)
    ax.set_xlabel("Savings (bits)")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    _finalize_plot(fig, save_path)


def generate_policy_entropy_plots(
    summary: Dict[str, Any],
    batch_results: List[Dict[str, Any]],
    output_dir: str,
    run_name: str = "test",
    title_prefix: str = "Policy Entropy",
) -> Dict[str, str]:
    """Generate and save all policy-entropy plots."""
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


def aggregate_policy_entropy_batch_results(
    batch_results: List[Dict[str, Any]]
) -> Dict[str, np.ndarray]:
    """Combine per-batch entropy arrays into full-dataset arrays."""
    all_step_entropies = np.concatenate([b["per_step_entropy_bits"] for b in batch_results], axis=0)      # [Nq, R, T]
    all_path_entropies = np.concatenate([b["per_path_entropy_bits"] for b in batch_results], axis=0)      # [Nq, R]
    all_question_entropies = np.concatenate([b["per_question_entropy_bits"] for b in batch_results], axis=0)  # [Nq]
    all_action_surprisals = np.concatenate([b["action_surprisal_bits"] for b in batch_results], axis=0)   # [Nq, R, T]
    all_valid_action_counts = np.concatenate([b["valid_action_counts"] for b in batch_results], axis=0)   # [Nq, R, T]
    all_ideal_identifier_bits = np.concatenate([b["ideal_identifier_bits"] for b in batch_results], axis=0)   # [Nq, R, T]
    all_fixed_identifier_bits = np.concatenate([b["fixed_identifier_bits"] for b in batch_results], axis=0)   # [Nq, R, T]
    all_ideal_savings_bits = np.concatenate([b["ideal_savings_bits"] for b in batch_results], axis=0)   # [Nq, R, T]
    all_fixed_savings_bits = np.concatenate([b["fixed_savings_bits"] for b in batch_results], axis=0)   # [Nq, R, T]

    return {
        "all_step_entropies": all_step_entropies,
        "all_path_entropies": all_path_entropies,
        "all_question_entropies": all_question_entropies,
        "all_action_surprisals": all_action_surprisals,
        "all_valid_action_counts": all_valid_action_counts,
        "all_ideal_identifier_bits": all_ideal_identifier_bits,
        "all_fixed_identifier_bits": all_fixed_identifier_bits,
        "all_ideal_savings_bits": all_ideal_savings_bits,
        "all_fixed_savings_bits": all_fixed_savings_bits,
        "all_step_entropies_flat": all_step_entropies.reshape(-1),
        "all_path_entropies_flat": all_path_entropies.reshape(-1),
        "all_question_entropies_flat": all_question_entropies.reshape(-1),
        "all_action_surprisals_flat": all_action_surprisals.reshape(-1),
        "all_valid_action_counts_flat": all_valid_action_counts.reshape(-1),
        "all_ideal_identifier_bits_flat": all_ideal_identifier_bits.reshape(-1),
        "all_fixed_identifier_bits_flat": all_fixed_identifier_bits.reshape(-1),
        "all_ideal_savings_bits_flat": all_ideal_savings_bits.reshape(-1),
        "all_fixed_savings_bits_flat": all_fixed_savings_bits.reshape(-1),
    }


def save_policy_entropy_outputs(
    summary: Dict[str, Any], 
    aggregate: Dict[str, np.ndarray], 
    output_dir: str, 
    run_name: str,
    logger: logging.Logger = logging.getLogger(__name__),
) -> None:
    """Save summary JSON and raw arrays for later analysis."""
    os.makedirs(output_dir, exist_ok=True)

    summary_path = os.path.join(output_dir, f"{run_name}_policy_entropy_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(_jsonify(summary), f, indent=2)

    arrays_path = os.path.join(output_dir, f"{run_name}_policy_entropy_arrays.npz")
    np.savez_compressed(
        arrays_path,
        all_step_entropies=aggregate["all_step_entropies"],
        all_path_entropies=aggregate["all_path_entropies"],
        all_question_entropies=aggregate["all_question_entropies"],
        all_action_surprisals=aggregate["all_action_surprisals"],
        all_valid_action_counts=aggregate["all_valid_action_counts"],
        all_ideal_identifier_bits=aggregate["all_ideal_identifier_bits"],
        all_fixed_identifier_bits=aggregate["all_fixed_identifier_bits"],
        all_ideal_savings_bits=aggregate["all_ideal_savings_bits"],
        all_fixed_savings_bits=aggregate["all_fixed_savings_bits"],
    )

    logger.info(f"Saved entropy summary to {summary_path}")
    logger.info(f"Saved entropy arrays to {arrays_path}")