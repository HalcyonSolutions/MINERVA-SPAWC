"""
Policy-entropy artifact helpers for merging batch outputs and saving summary/array files.
"""
import json
import logging
import os

import numpy as np

from typing import Dict, Any, List

def _jsonify(
    value: Any
) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {k: _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    return value

def aggregate_policy_entropy_batch_results(
    batch_results: List[Dict[str, Any]]
) -> Dict[str, np.ndarray]:
    """Merge per-batch metrics into full arrays and matching flattened views."""
    array_keys = {
        "all_step_entropies": "per_step_entropy_bits",
        "all_path_entropies": "per_path_entropy_bits",
        "all_question_entropies": "per_question_entropy_bits",
        "all_action_surprisals": "action_surprisal_bits",
        "all_valid_action_counts": "valid_action_counts",
        "all_ideal_identifier_bits": "ideal_identifier_bits",
        "all_fixed_identifier_bits": "fixed_identifier_bits",
        "all_ideal_savings_bits": "ideal_savings_bits",
        "all_fixed_savings_bits": "fixed_savings_bits",
    }
    aggregate = {
        out_key: np.concatenate([batch[in_key] for batch in batch_results], axis=0)
        for out_key, in_key in array_keys.items()
    }
    aggregate.update({f"{key}_flat": value.reshape(-1) for key, value in aggregate.items()})
    return aggregate


def save_policy_entropy_outputs(
    summary: Dict[str, Any], 
    aggregate: Dict[str, np.ndarray], 
    output_dir: str, 
    run_name: str,
    logger: logging.Logger = logging.getLogger(__name__),
) -> None:
    """Persist summary metadata and aggregated raw arrays for later analysis."""
    os.makedirs(output_dir, exist_ok=True)

    summary_path = os.path.join(output_dir, f"{run_name}_policy_entropy_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(_jsonify(summary), f, indent=2)

    arrays_path = os.path.join(output_dir, f"{run_name}_policy_entropy_arrays.npz")
    array_keys = [
        "all_step_entropies",
        "all_path_entropies",
        "all_question_entropies",
        "all_action_surprisals",
        "all_valid_action_counts",
        "all_ideal_identifier_bits",
        "all_fixed_identifier_bits",
        "all_ideal_savings_bits",
        "all_fixed_savings_bits",
    ]
    np.savez_compressed(
        arrays_path,
        **{key: aggregate[key] for key in array_keys},
    )

    logger.info(f"Saved entropy summary to {summary_path}")
    logger.info(f"Saved entropy arrays to {arrays_path}")
