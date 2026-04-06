"""
Core policy-entropy math utilities: entropy/surprisal, valid-action counts, identifier-bit baselines, and savings.
"""

import numpy as np

from typing import List

def count_valid_action(
    next_relations: np.ndarray,
    next_entities: np.ndarray,
    invalid_relation_ids: List[int] = None,
    invalid_entity_ids: List[int] = None,
    fallback_log_probs: np.ndarray = None,
) -> np.ndarray:
    """
    Count valid actions based on next_relations and next_entities, excluding any with invalid relation/entity IDs.

    Preference order:
      1. Count actions whose relation/entity IDs are not known padding IDs.
      2. Fall back to counting actions with non-zero probability mass if padding IDs
         are unavailable.

    Args:
        next_relations: [B*R, A]
        next_entities: [B*R, A]
        invalid_relation_ids: list of relation IDs to consider invalid (e.g. padding IDs), or None to ignore
        invalid_entity_ids: list of entity IDs to consider invalid (e.g. padding IDs), or None to ignore
        fallback_log_probs: [B*R, A] log-probabilities, used only as fallback

    Returns:
        valid_counts: [B*R] integer count of valid actions, clipped to at least 1 (self loop)
    """

    valid_mask = None

    if invalid_relation_ids:
        rel_valid = ~np.isin(next_relations, invalid_relation_ids)
        valid_mask = rel_valid if valid_mask is None else (valid_mask & rel_valid)

    if invalid_entity_ids:
        ent_valid = ~np.isin(next_entities, invalid_entity_ids)
        valid_mask = ent_valid if valid_mask is None else (valid_mask & ent_valid)

    if valid_mask is None:
        if fallback_log_probs is None:
            raise ValueError("Cannot infer valid actions without padding IDs or fallback_log_probs.")
        probs = np.exp(fallback_log_probs)
        valid_mask = probs > 0.0

    valid_counts = valid_mask.sum(axis=1).astype(np.int32)
    valid_counts = np.maximum(valid_counts, 1)
    return valid_counts


def fixed_width_identifier_bits(
        valid_action_counts: np.ndarray
) -> np.ndarray:
    """Bits needed to identify one action among K valid actions using a fixed-width local index where valid_action_counts [B*R]."""
    valid_action_counts = np.asarray(valid_action_counts, dtype=np.int32)
    return np.ceil(np.log2(np.maximum(valid_action_counts, 1))).astype(np.float32)


def ideal_uniform_identifier_bits(
    valid_action_counts: np.ndarray
) -> np.ndarray:
    """Ideal lower bound in bits for uniformly identifying one action among K valid actions where valid_action_counts [B*R]."""
    valid_action_counts = np.asarray(valid_action_counts, dtype=np.int32)
    return np.log2(np.maximum(valid_action_counts, 1)).astype(np.float32)


def entropy_bits_from_log_probs(
    log_probs: np.ndarray, 
    axis: int = -1
) -> np.ndarray:
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


def action_surprisal_bits_from_log_probs(
    log_probs: np.ndarray, 
    action_idx: np.ndarray
) -> np.ndarray:
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
