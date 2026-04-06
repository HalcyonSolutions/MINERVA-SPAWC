# MINERVA-SPAWC

Official implementation of the SPAWC 2026 paper:

**"The Information Cost of Multi-Hop Reasoning: Distributed Graph Navigation under Communication Constraints"**

This repository builds on MINERVA and provides the scripts, configs, and pretrained assets used for evaluation and policy-entropy analysis.

## Dependencies

Initialize the MINERVA submodule:

```bash
git submodule update --init --recursive
```

Install the MINERVA requirements before running anything in this repository:

```bash
cd minerva
pip install -r requirements.txt
cd ..
```

## Data and Pretrained Assets

Download the `.cache`, `datasets`, and `saved_models` directories from:

https://storage.googleapis.com/halcyon_data/multihop_ds/conferences/spawc/minerva_spawc.zip

Extract the archive in the repository root so the directories appear at the top level.

## Run

Use a dataset config to run evaluation and policy-entropy analysis:

```bash
bash run_spawc.sh --config_yaml configs/<dataset>.yaml
```

Example:

```bash
bash run_spawc.sh --config_yaml configs/kinshiphinton.yaml
```

Available configs:

- `configs/kinshiphinton.yaml`
- `configs/metaqa.yaml`
- `configs/mquake.yaml`

Outputs are written under `./saved_models/<dataset>/<run_name>/`:

- Entropy analysis artifacts: `./saved_models/<dataset>/<run_name>/policy_entropy/`
- Evaluation results: `./saved_models/<dataset>/<run_name>/scores.txt`
- Individual question answers: `./saved_models/<dataset>/<run_name>/test_beam/test_paths.txt`

## Folder Structure

The repository is organized as follows:
```
minerva_spawc/
├── .cache/                 # Cached files for processed datasets
├── code/
│   ├── evaluation_spawc.py # Evaluation entrypoint for SPAWC experiments
│   └── policy_entropy/
│       ├── eval.py         # Policy-entropy evaluation pipeline and summaries
│       ├── metrics.py      # Core entropy/surprisal and identifier-bit utilities
│       ├── artifacts.py    # Save/load helpers for policy-entropy artifacts
│       └── plotting.py     # Plot generation for policy-entropy analysis
├── configs/                # YAML config files for different datasets
├── datasets/               # Datasets for evaluation (preprocessed and ready to use)
├── minerva/                # MINERVA codebase (submodule)
├── run_spawc.sh            # Convenience script to launch configured runs
└── saved_models/            # Pretrained models and evaluation outputs
    ├── kinshiphinton/      # Outputs for the Kinship-Hinton dataset
    ├── metaqa/             # Outputs for the MetaQA dataset
    └── mquake/             # Outputs for the MQuAKE dataset

```

## Citation

If you use this code for your research, please cite our paper:

```
@inproceedings{minerva_spawc_2026,
  title={The Information Cost of Multi-Hop Reasoning: Distributed Graph Navigation under Communication Constraints},
  author={Hernandez, Eduin E and Garcia, Luis F, and Askar, Nurassyl, and Rini, Stefano},
  booktitle={2026 IEEE 27th International Workshop on Signal Processing and Artificial Intelligence in Wireless Communications},
  year={2026},
  url={https://github.com/HalcyonSolutions/MINERVA-SPAWC}
}
