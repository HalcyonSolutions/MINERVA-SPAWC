"""
Evaluation script for analyzing policy entropy of a trained Minerva agent various datasets.
"""

from __future__ import absolute_import
from __future__ import division

import json
import logging
import os
import sys

import tensorflow as tf

from minerva.code.data.embedding_server import EmbeddingServer
from minerva.code.model.trainer import TrainerNLQ
from minerva.code.data.setup import set_seeds
from minerva.code.options import read_options

from policy_entropy.eval import analyze_policy_entropy_testset, extract_dataset_name
from policy_entropy.plotting import generate_policy_entropy_plots

logger = logging.getLogger()
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

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

        logger.info("=== Policy Entropy Summary ===")
        for k, v in summary.items():
            logger.info(f"{k}: {v}")

        dataset_name = extract_dataset_name(options)
        entropy_plot_paths = generate_policy_entropy_plots(
            summary=summary,
            batch_results=batch_results,
            output_dir=os.path.join(output_dir, "policy_entropy"),
            run_name=dataset_name,
            title_prefix=f"{dataset_name.upper()} Policy Entropy",
        )

        logger.info("=== Saved Policy Entropy Plots ===")
        for k, v in entropy_plot_paths.items():
            logger.info(f"{k}: {v}")

    logger.info("Evaluation completed. Closing Server")
    embedding_server.close()  # Close the embedding server connection
