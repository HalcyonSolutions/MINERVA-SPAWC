"""
Policy-entropy evaluation pipeline for running episodes/batches and producing aggregated statistics.
"""

import os
import re

from tqdm import tqdm

import tensorflow as tf
import numpy as np

from minerva.code.model.trainer import TrainerNLQ

from policy_entropy.metrics import (
    entropy_bits_from_log_probs,
    action_surprisal_bits_from_log_probs,
    count_valid_action,
    ideal_uniform_identifier_bits,
    fixed_width_identifier_bits,
)

from typing import Dict, Any, List

def extract_dataset_name(
    options: Dict[str, Any]
) -> str:
    """Infer a safe dataset name from the data_input_dir option for use in plot titles and filenames."""
    value = options.get("data_input_dir", "")
    if value:
        name = os.path.basename(os.path.normpath(value))
        if name:
            safe_name = "".join(ch if ch.isalnum() or ch in ('-', '_') else '_' for ch in name)
            # remove any _ followed by 'v' and digits, to avoid merging version numbers into the name
            safe_name = re.sub(r'_v\d+', '', safe_name)
            return safe_name
    return "test"


def collect_policy_entropy_single_episode(
    trainer: TrainerNLQ, 
    sess: tf.compat.v1.Session,
    episode: Any,
    invalid_rel_ids: List[int] = None,
    invalid_ent_ids: List[int] = None,
) -> Dict[str, Any]:
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

        valid_action_counts = count_valid_action(
            next_relations=state["next_relations"],
            next_entities=state["next_entities"],
            invalid_relation_ids=invalid_rel_ids,
            invalid_entity_ids=invalid_ent_ids,
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

    invalid_rel_ids= [
        trainer.relation_vocab['PAD'], 
        trainer.relation_vocab['UNK'], 
        trainer.relation_vocab['DUMMY_START_RELATION']
    ]

    invalid_ent_ids=[
        trainer.entity_vocab['PAD'], 
        trainer.entity_vocab['UNK']
    ]

    question_num = trainer.environment.batcher.get_question_num()
    test_batch_size = trainer.environment.batcher.test_batch_size
    total_batches = (question_num + test_batch_size - 1) // test_batch_size
    num_batches = total_batches if max_batches is None else min(max_batches, total_batches)
    for batch_idx, episode in enumerate(tqdm(trainer.environment.get_episodes(), desc="Evaluating policy entropy", total=num_batches)):
        if max_batches is not None and batch_idx >= max_batches:
            break

        batch_res = collect_policy_entropy_single_episode(
            trainer, 
            sess, 
            episode, 
            invalid_rel_ids=invalid_rel_ids, 
            invalid_ent_ids=invalid_ent_ids
        )
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
