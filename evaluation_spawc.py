"""
Evaluation script for analyzing policy entropy of a trained Minerva agent various datasets.
"""

from __future__ import absolute_import
from __future__ import division

import json
import logging
import os
import re
import sys

from tqdm import tqdm


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

from minerva.code.data.embedding_server import EmbeddingServer
from minerva.code.model.trainer import TrainerNLQ
from minerva.code.data.setup import set_seeds
from minerva.code.options import read_options

from typing import Dict, Any, List

logger = logging.getLogger()
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

def _lookup_vocab_ids(vocab: Dict[str, int], candidates: List[str]) -> List[int]:
    """Return IDs for candidate token names that exist in the given vocabulary."""
    if not vocab:
        return []
    return [int(vocab[name]) for name in candidates if name in vocab]


def infer_valid_action_counts(
    next_relations: np.ndarray,
    next_entities: np.ndarray,
    relation_vocab: Dict[str, int] = None,
    entity_vocab: Dict[str, int] = None,
    fallback_log_probs: np.ndarray = None,
) -> np.ndarray:
    """
    Infer the number of valid, non-padded actions for each example in a batched action set.

    Preference order:
      1. Count actions whose relation/entity IDs are not known padding IDs.
      2. Fall back to counting actions with non-zero probability mass if padding IDs
         are unavailable.

    Args:
        next_relations: [B*R, A]
        next_entities: [B*R, A]
        relation_vocab: relation vocabulary, optionally containing PAD tokens
        entity_vocab: entity vocabulary, optionally containing PAD tokens
        fallback_log_probs: [B*R, A] log-probabilities, used only as fallback

    Returns:
        valid_counts: [B*R] integer count of valid actions, clipped to at least 1
    """
    relation_vocab = relation_vocab or {}
    entity_vocab = entity_vocab or {}

    relation_pad_ids = set(_lookup_vocab_ids(
        relation_vocab,
        ["PAD", "DUMMY_START_RELATION",],
    ))
    entity_pad_ids = set(_lookup_vocab_ids(
        entity_vocab,
        ["PAD", "UNK",],
    ))

    valid_mask = None

    if relation_pad_ids:
        rel_valid = ~np.isin(next_relations, list(relation_pad_ids))
        valid_mask = rel_valid if valid_mask is None else (valid_mask & rel_valid)

    if entity_pad_ids:
        ent_valid = ~np.isin(next_entities, list(entity_pad_ids))
        valid_mask = ent_valid if valid_mask is None else (valid_mask & ent_valid)

    if valid_mask is None:
        if fallback_log_probs is None:
            raise ValueError("Cannot infer valid actions without padding IDs or fallback_log_probs.")
        probs = np.exp(fallback_log_probs)
        valid_mask = probs > 0.0

    valid_counts = valid_mask.sum(axis=1).astype(np.int32)
    valid_counts = np.maximum(valid_counts, 1)
    return valid_counts


def fixed_width_identifier_bits(valid_action_counts: np.ndarray) -> np.ndarray:
    """Bits needed to identify one action among K valid actions using a fixed-width local index."""
    valid_action_counts = np.asarray(valid_action_counts, dtype=np.int32)
    return np.ceil(np.log2(np.maximum(valid_action_counts, 1))).astype(np.float32)


def ideal_uniform_identifier_bits(valid_action_counts: np.ndarray) -> np.ndarray:
    """Ideal lower bound in bits for uniformly identifying one action among K valid actions."""
    valid_action_counts = np.asarray(valid_action_counts, dtype=np.int32)
    return np.log2(np.maximum(valid_action_counts, 1)).astype(np.float32)


def entropy_bits_from_log_probs(log_probs: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Compute policy entropy in bits from log-probabilities (natural log).

    Args:
        log_probs: array of shape [..., num_actions]
        axis: axis corresponding to the action dimension

    Returns:
        Entropy in bits with shape log_probs.shape with `axis` removed.
    """
    probs = np.exp(log_probs)
    entropy_nats = -np.sum(probs * log_probs, axis=axis)
    entropy_bits = entropy_nats / np.log(2.0)
    return entropy_bits



def action_surprisal_bits_from_log_probs(log_probs: np.ndarray, action_idx: np.ndarray) -> np.ndarray:
    """
    Optional helper: realized code length in bits for the sampled action.

    This is -log2 pi(a_t | s_t, q) for the action actually taken.
    Its expectation over sampled actions equals the entropy.

    Args:
        log_probs: [B, A]
        action_idx: [B]

    Returns:
        surprisal_bits: [B]
    """
    chosen_log_probs = log_probs[np.arange(log_probs.shape[0]), action_idx]
    return -chosen_log_probs / np.log(2.0)



def collect_policy_entropy_single_episode(trainer, sess, episode):
    """
    Run one evaluation episode with beam=False and collect policy entropy.

    Returns a dictionary with:
        per_step_entropy_bits_flat: [B*R, T]
        per_step_entropy_bits:      [B, R, T]
        per_path_entropy_bits:      [B, R]
        per_question_entropy_bits:  [B]
        global_mean_step_entropy_bits: scalar
        global_mean_path_entropy_bits: scalar

    Also returns optional diagnostics:
        action_surprisal_bits:      [B, R, T]
        action_idx:                 [B, R, T]
        chosen_relation:            [B, R, T]
        valid_action_counts:        [B, R, T]
        ideal_identifier_bits:      [B, R, T]
        fixed_identifier_bits:      [B, R, T]
        ideal_savings_bits:         [B, R, T]
        fixed_savings_bits:         [B, R, T]
        final_entities:             [B, R]
    """
    temp_batch_size = episode.no_examples
    R = trainer.test_rollouts
    T = trainer.path_length
    BR = temp_batch_size * R

    # Initial environment state
    state = episode.get_state()

    # Initial recurrent memory
    mem_shape = trainer.agent.get_mem_shape()
    agent_mem = np.zeros((mem_shape[0], mem_shape[1], BR, mem_shape[3]), dtype=np.float32)

    previous_relation = (
        np.ones((BR,), dtype=np.int64) * trainer.relation_vocab["DUMMY_START_RELATION"]
    )

    constant_feed = {
        trainer.range_arr: np.arange(BR, dtype=np.int32),
        trainer.question_embedding: episode.get_question_embedding(),
    }

    entropies_bits = []
    surprisals_bits = []
    action_indices = []
    chosen_relations = []
    valid_action_counts_list = []
    ideal_identifier_bits_list = []
    fixed_identifier_bits_list = []
    ideal_savings_bits_list = []
    fixed_savings_bits_list = []

    for _ in range(T):
        feed_dict = {
            **constant_feed,
            trainer.next_relations: state["next_relations"],
            trainer.next_entities: state["next_entities"],
            trainer.current_entities: state["current_entities"],
            trainer.prev_state: agent_mem,
            trainer.prev_relation: previous_relation,
        }

        _, agent_mem, test_log_probs, test_action_idx, chosen_relation = sess.run(
            [
                trainer.test_loss,
                trainer.test_state,
                trainer.test_logits,       # [B*R, A], already log-softmax
                trainer.test_action_idx,   # [B*R]
                trainer.chosen_relation,   # [B*R]
            ],
            feed_dict=feed_dict,
        )

        step_entropy_bits = entropy_bits_from_log_probs(test_log_probs, axis=1)  # [B*R]
        step_surprisal_bits = action_surprisal_bits_from_log_probs(
            test_log_probs, test_action_idx
        )  # [B*R]

        valid_action_counts = infer_valid_action_counts(
            next_relations=state["next_relations"],
            next_entities=state["next_entities"],
            relation_vocab=getattr(trainer, "relation_vocab", {}),
            entity_vocab=getattr(trainer, "entity_vocab", {}),
            fallback_log_probs=test_log_probs,
        )  # [B*R]
        step_ideal_identifier_bits = ideal_uniform_identifier_bits(valid_action_counts)  # [B*R]
        step_fixed_identifier_bits = fixed_width_identifier_bits(valid_action_counts)    # [B*R]
        step_ideal_savings_bits = step_ideal_identifier_bits - step_entropy_bits          # [B*R]
        step_fixed_savings_bits = step_fixed_identifier_bits - step_entropy_bits          # [B*R]

        entropies_bits.append(step_entropy_bits)
        surprisals_bits.append(step_surprisal_bits)
        action_indices.append(test_action_idx)
        chosen_relations.append(chosen_relation)
        valid_action_counts_list.append(valid_action_counts)
        ideal_identifier_bits_list.append(step_ideal_identifier_bits)
        fixed_identifier_bits_list.append(step_fixed_identifier_bits)
        ideal_savings_bits_list.append(step_ideal_savings_bits)
        fixed_savings_bits_list.append(step_fixed_savings_bits)

        previous_relation = chosen_relation
        state = episode(test_action_idx)

    # Stack over time: [B*R, T]
    per_step_entropy_bits_flat = np.stack(entropies_bits, axis=1)
    action_surprisal_bits_flat = np.stack(surprisals_bits, axis=1)
    action_idx_flat = np.stack(action_indices, axis=1)
    chosen_relation_flat = np.stack(chosen_relations, axis=1)
    valid_action_counts_flat = np.stack(valid_action_counts_list, axis=1)
    ideal_identifier_bits_flat = np.stack(ideal_identifier_bits_list, axis=1)
    fixed_identifier_bits_flat = np.stack(fixed_identifier_bits_list, axis=1)
    ideal_savings_bits_flat = np.stack(ideal_savings_bits_list, axis=1)
    fixed_savings_bits_flat = np.stack(fixed_savings_bits_list, axis=1)

    # Reshape to [B, R, T]
    per_step_entropy_bits = per_step_entropy_bits_flat.reshape(temp_batch_size, R, T)
    action_surprisal_bits = action_surprisal_bits_flat.reshape(temp_batch_size, R, T)
    action_idx = action_idx_flat.reshape(temp_batch_size, R, T)
    chosen_relation = chosen_relation_flat.reshape(temp_batch_size, R, T)
    valid_action_counts = valid_action_counts_flat.reshape(temp_batch_size, R, T)
    ideal_identifier_bits = ideal_identifier_bits_flat.reshape(temp_batch_size, R, T)
    fixed_identifier_bits = fixed_identifier_bits_flat.reshape(temp_batch_size, R, T)
    ideal_savings_bits = ideal_savings_bits_flat.reshape(temp_batch_size, R, T)
    fixed_savings_bits = fixed_savings_bits_flat.reshape(temp_batch_size, R, T)

    # Aggregations
    per_path_entropy_bits = per_step_entropy_bits.sum(axis=2)        # [B, R]
    per_question_entropy_bits = per_path_entropy_bits.mean(axis=1)   # [B]
    per_path_ideal_identifier_bits = ideal_identifier_bits.sum(axis=2)  # [B, R]
    per_path_fixed_identifier_bits = fixed_identifier_bits.sum(axis=2)  # [B, R]
    per_path_ideal_savings_bits = ideal_savings_bits.sum(axis=2)        # [B, R]
    per_path_fixed_savings_bits = fixed_savings_bits.sum(axis=2)        # [B, R]

    result = {
        "per_step_entropy_bits_flat": per_step_entropy_bits_flat,            # [B*R, T]
        "per_step_entropy_bits": per_step_entropy_bits,                      # [B, R, T]
        "per_path_entropy_bits": per_path_entropy_bits,                      # [B, R]
        "per_question_entropy_bits": per_question_entropy_bits,              # [B]
        "global_mean_step_entropy_bits": float(per_step_entropy_bits.mean()),
        "global_mean_path_entropy_bits": float(per_path_entropy_bits.mean()),
        "action_surprisal_bits": action_surprisal_bits,                      # [B, R, T]
        "action_idx": action_idx,                                            # [B, R, T]
        "chosen_relation": chosen_relation,                                  # [B, R, T]
        "valid_action_counts": valid_action_counts,                          # [B, R, T]
        "ideal_identifier_bits": ideal_identifier_bits,                      # [B, R, T]
        "fixed_identifier_bits": fixed_identifier_bits,                      # [B, R, T]
        "ideal_savings_bits": ideal_savings_bits,                            # [B, R, T]
        "fixed_savings_bits": fixed_savings_bits,                            # [B, R, T]
        "per_path_ideal_identifier_bits": per_path_ideal_identifier_bits,    # [B, R]
        "per_path_fixed_identifier_bits": per_path_fixed_identifier_bits,    # [B, R]
        "per_path_ideal_savings_bits": per_path_ideal_savings_bits,          # [B, R]
        "per_path_fixed_savings_bits": per_path_fixed_savings_bits,          # [B, R]
        "final_entities": state["current_entities"].reshape(temp_batch_size, R),
    }

    return result



def analyze_policy_entropy_testset(trainer, sess, mode: str = "test", max_batches: int = None):
    """
    Run the full evaluation split with beam=False and collect entropy statistics.

    Returns:
        summary: dict of aggregate metrics
        batch_results: list of per-batch dictionaries from collect_policy_entropy_single_episode()
    """
    trainer.environment.change_mode(mode)
    trainer.environment.change_test_rollouts(trainer.test_rollouts)

    batch_results = []
    all_step_entropies = []
    all_path_entropies = []
    all_question_entropies = []
    all_action_surprisals = []
    all_valid_action_counts = []
    all_ideal_identifier_bits = []
    all_fixed_identifier_bits = []
    all_ideal_savings_bits = []
    all_fixed_savings_bits = []

    question_num = trainer.environment.batcher.get_question_num()
    test_batch_size = trainer.environment.batcher.test_batch_size
    total_batches = (question_num + test_batch_size - 1) // test_batch_size
    num_batches = total_batches if max_batches is None else min(max_batches, total_batches)
    for batch_idx, episode in enumerate(tqdm(trainer.environment.get_episodes(), desc="Evaluating policy entropy", total=num_batches)):
        if max_batches is not None and batch_idx >= max_batches:
            break

        batch_res = collect_policy_entropy_single_episode(trainer, sess, episode)
        batch_results.append(batch_res)

        all_step_entropies.append(batch_res["per_step_entropy_bits"])         # [B, R, T]
        all_path_entropies.append(batch_res["per_path_entropy_bits"])         # [B, R]
        all_question_entropies.append(batch_res["per_question_entropy_bits"]) # [B]
        all_action_surprisals.append(batch_res["action_surprisal_bits"])      # [B, R, T]
        all_valid_action_counts.append(batch_res["valid_action_counts"])      # [B, R, T]
        all_ideal_identifier_bits.append(batch_res["ideal_identifier_bits"])  # [B, R, T]
        all_fixed_identifier_bits.append(batch_res["fixed_identifier_bits"])  # [B, R, T]
        all_ideal_savings_bits.append(batch_res["ideal_savings_bits"])        # [B, R, T]
        all_fixed_savings_bits.append(batch_res["fixed_savings_bits"])        # [B, R, T]

    all_step_entropies = np.concatenate(all_step_entropies, axis=0)          # [Nq, R, T]
    all_path_entropies = np.concatenate(all_path_entropies, axis=0)          # [Nq, R]
    all_question_entropies = np.concatenate(all_question_entropies, axis=0)  # [Nq]
    all_action_surprisals = np.concatenate(all_action_surprisals, axis=0)    # [Nq, R, T]
    all_valid_action_counts = np.concatenate(all_valid_action_counts, axis=0)      # [Nq, R, T]
    all_ideal_identifier_bits = np.concatenate(all_ideal_identifier_bits, axis=0)  # [Nq, R, T]
    all_fixed_identifier_bits = np.concatenate(all_fixed_identifier_bits, axis=0)  # [Nq, R, T]
    all_ideal_savings_bits = np.concatenate(all_ideal_savings_bits, axis=0)        # [Nq, R, T]
    all_fixed_savings_bits = np.concatenate(all_fixed_savings_bits, axis=0)        # [Nq, R, T]

    summary = {
        "num_questions": int(all_question_entropies.shape[0]),
        "num_rollouts": int(all_step_entropies.shape[1]),
        "path_length": int(all_step_entropies.shape[2]),
        "mean_step_entropy_bits": float(all_step_entropies.mean()),
        "std_step_entropy_bits": float(all_step_entropies.std()),
        "mean_path_entropy_bits": float(all_path_entropies.mean()),
        "std_path_entropy_bits": float(all_path_entropies.std()),
        "mean_question_entropy_bits": float(all_question_entropies.mean()),
        "std_question_entropy_bits": float(all_question_entropies.std()),
        "mean_step_surprisal_bits": float(all_action_surprisals.mean()),
        "mean_valid_actions": float(all_valid_action_counts.mean()),
        "mean_ideal_identifier_bits": float(all_ideal_identifier_bits.mean()),
        "mean_fixed_identifier_bits": float(all_fixed_identifier_bits.mean()),
        "mean_ideal_savings_bits": float(all_ideal_savings_bits.mean()),
        "mean_fixed_savings_bits": float(all_fixed_savings_bits.mean()),
        "per_hop_mean_entropy_bits": all_step_entropies.mean(axis=(0, 1)),        # [T]
        "per_hop_std_entropy_bits": all_step_entropies.std(axis=(0, 1)),          # [T]
        "per_hop_mean_valid_actions": all_valid_action_counts.mean(axis=(0, 1)),   # [T]
        "per_hop_mean_ideal_identifier_bits": all_ideal_identifier_bits.mean(axis=(0, 1)),  # [T]
        "per_hop_mean_fixed_identifier_bits": all_fixed_identifier_bits.mean(axis=(0, 1)),  # [T]
        "per_hop_mean_ideal_savings_bits": all_ideal_savings_bits.mean(axis=(0, 1)),        # [T]
        "per_hop_mean_fixed_savings_bits": all_fixed_savings_bits.mean(axis=(0, 1)),        # [T]
    }

    return summary, batch_results



def aggregate_policy_entropy_batch_results(batch_results: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
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



def _jsonify(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {k: _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    return value



def save_policy_entropy_outputs(summary: Dict[str, Any], aggregate: Dict[str, np.ndarray], output_dir: str, run_name: str) -> None:
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



def _finalize_plot(fig, save_path: str) -> None:
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved plot to {save_path}")



def plot_per_hop_entropy(summary: Dict[str, Any], save_path: str, title_prefix: str = "Policy Entropy") -> None:
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



def plot_cumulative_entropy(summary: Dict[str, Any], save_path: str, title_prefix: str = "Policy Entropy") -> None:
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



def plot_path_entropy_histogram(aggregate: Dict[str, np.ndarray], save_path: str, title_prefix: str = "Policy Entropy") -> None:
    values = aggregate["all_path_entropies_flat"]
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.hist(values, bins=40)
    ax.set_xlabel("Path entropy (bits)")
    ax.set_ylabel("Count")
    ax.set_title(f"{title_prefix}: Path entropy distribution")
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')  # Use log scale for better visibility of tail
    _finalize_plot(fig, save_path)



def plot_question_entropy_histogram(aggregate: Dict[str, np.ndarray], save_path: str, title_prefix: str = "Policy Entropy") -> None:
    values = aggregate["all_question_entropies_flat"]
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.hist(values, bins=40)
    ax.set_xlabel("Question entropy (bits)")
    ax.set_ylabel("Count")
    ax.set_title(f"{title_prefix}: Question entropy distribution")
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')  # Use log scale for better visibility of tail
    _finalize_plot(fig, save_path)



def plot_step_entropy_boxplot(aggregate: Dict[str, np.ndarray], save_path: str, title_prefix: str = "Policy Entropy") -> None:
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



def infer_run_name(options: Dict[str, Any]) -> str:
    """Infer a compact run label from available paths."""
    value = options.get("data_input_dir", "")
    if value:
        name = os.path.basename(os.path.normpath(value))
        if name:
            safe_name = "".join(ch if ch.isalnum() or ch in ('-', '_') else '_' for ch in name)
            # remove any _ followed by 'v' and digits, to avoid merging version numbers into the name
            safe_name = re.sub(r'_v\d+', '', safe_name)
            return safe_name
    return "test"


if __name__ == '__main__':
    # Read command line options and setup logging
    options = read_options()

    # Set logging
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%Y/%m/%d %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)
    logfile = logging.FileHandler(options['log_file_name'], 'w')
    logfile.setFormatter(fmt)
    logger.addHandler(logfile)

    # Load vocabularies
    logger.info('Reading vocab files (ent & rel to id)...')
    relation_vocab = json.load(open(os.path.join(options['vocab_dir'], 'relation_vocab.json')))
    entity_vocab = json.load(open(os.path.join(options['vocab_dir'], 'entity_vocab.json')))

    logger.info('Total number of entities {}'.format(len(entity_vocab)))
    logger.info('Total number of relations {}'.format(len(relation_vocab)))

    # Configure TensorFlow for deterministic behavior
    save_path = ''
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = False
    config.log_device_placement = False
    config.allow_soft_placement = True

    # Set seed for reproducibility
    set_seeds(options['seed'])

    embedding_server = EmbeddingServer(options['question_tokenizer_name'])

    logger.info(f"Loading model from {options['model_load_dir']}")

    save_path = options['model_load_dir']
    path_logger_file = options['path_logger_file']
    output_dir = options['output_dir']

    # Evaluation phase
    trainer = TrainerNLQ(
        batch_size=options['batch_size'],
        test_batch_size=options['test_batch_size'],
        num_rollouts=options['num_rollouts'],
        test_rollouts=options['test_rollouts'],
        positive_reward=options['positive_reward'],
        negative_reward=options['negative_reward'],
        path_length=options['path_length'],
        data_input_dir=options['data_input_dir'],
        question_tokenizer_name=options['question_tokenizer_name'],
        question_format=options['question_format'],
        cached_QAMetaData_path=options['cached_QAMetaData_path'],
        raw_QAData_path=options['raw_QAData_path'],
        force_data_prepro=False,
        evaluate_paraphrases=options['evaluate_paraphrases'],
        multi_answers=options['multi_answers'],
        max_num_actions=options['max_num_actions'],
        embedding_size=options['embedding_size'],
        hidden_size=options['hidden_size'],
        use_entity_embeddings=options['use_entity_embeddings'],
        train_entity_embeddings=options['train_entity_embeddings'],
        train_relation_embeddings=options['train_relation_embeddings'],
        LSTM_layers=options['LSTM_layers'],
        projection_adapter=options['projection_adapter'],
        projection_layers=options['projection_layers'],
        projection_hidden=options['projection_hidden'],
        learning_rate=options['learning_rate'],
        grad_clip_norm=options['grad_clip_norm'],
        gamma=options['gamma'],
        Lambda=options['Lambda'],
        beta=options['beta'],
        total_iterations=options['total_iterations'],
        eval_every=options['eval_every'],
        output_dir=options['output_dir'],
        model_dir=options['model_dir'],
        path_logger_file=options['path_logger_file'],
        pool=options['pool'],
        use_beam=options['use_beam'],
        seed=options['seed'],
        entity_vocab=entity_vocab,
        relation_vocab=relation_vocab,
        use_full_graph=options['use_full_graph'],
        use_directed_graph=options['use_directed_graph'],
        use_stop_signal=options['use_stop_signal'],
        use_restart_signal=options['use_restart_signal'],
        stop_signal_reward=options['stop_signal_reward'],
        stop_signal_penalty=options['stop_signal_penalty'],
        length_penalty=options['length_penalty'],
        embedding_server=embedding_server,
        use_wandb=False  # Do not use WANDB for Evaluation
    )

    with tf.compat.v1.Session(config=config) as sess:
        # Set seeds again after session creation to ensure TF operations are deterministic
        set_seeds(options['seed'])
        trainer.initialize(restore=save_path, sess=sess)

        summary, batch_results = analyze_policy_entropy_testset(
            trainer=trainer,
            sess=sess,
            mode='test',
            max_batches=None,
        )

        print("=== Policy Entropy Summary ===")
        for k, v in summary.items():
            print(f"{k}: {v}")

        run_name = infer_run_name(options)
        entropy_plot_paths = generate_policy_entropy_plots(
            summary=summary,
            batch_results=batch_results,
            output_dir=os.path.join(output_dir, "policy_entropy"),
            run_name=run_name,
            title_prefix=f"{run_name} Policy Entropy",
        )

        print("=== Saved Policy Entropy Plots ===")
        for k, v in entropy_plot_paths.items():
            print(f"{k}: {v}")

    logging.info("Evaluation completed. Closing Server")
    embedding_server.close()  # Close the embedding server connection
