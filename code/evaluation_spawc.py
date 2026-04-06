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

import tensorflow as tf
import numpy as np

from minerva.code.data.embedding_server import EmbeddingServer
from minerva.code.model.trainer import TrainerNLQ
from minerva.code.data.setup import set_seeds
from minerva.code.options import read_options

from utils.plotting import generate_policy_entropy_plots

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
