# ORBIT: Experiment Reproduction Guide

## 1. Prerequisites

- Python 3.10 or newer.
- Required packages from `requirements.txt`, including `numpy`, `pyyaml`, `lz4`, `zstandard`, `matplotlib`, and `pytest`.
- A local checkout of the ORBIT repository with the corpus paths updated in `evaluation/datasets.yaml`.

## 2. Dataset Preparation

1. Generate the three corpora with `make_corpus.py`.

```bash
python make_corpus.py --output data/mixed_corpus.bin
python make_corpus.py --output data/text_corpus.txt
python make_corpus.py --output data/binary_corpus.bin
```

2. Update `evaluation/datasets.yaml` so each dataset entry points to the generated files and reflects the actual file sizes.

3. Verify that the dataset manifest contains these three corpora before running experiments.

## 3. Experiment 1: Core Comparison

Run the full experiment suite, which starts with the core comparison benchmark.

```bash
python run_experiments.py
```

- Output: `outputs/core_comparison.json`
- Expected runtime: 30-90 minutes

## 4. Experiment 2: Regret Curve

The regret curve is generated automatically during Experiment 1.

- Output: `outputs/regret_curve_aggregated.json`
- Note: generated automatically during Experiment 1

## 5. Experiment 3: Feature Ablation

The ablation study evaluates feature subsets across the datasets.

```bash
python run_experiments.py
```

- Output: `outputs/ablation_results.json`

## 6. Experiment 4: Block Size Sweep

The block size sweep is also run as part of the main experiment driver.

```bash
python run_experiments.py
```

- Output: `outputs/block_size_sweep.json`

## 7. Results Validation

Use the following checklist to verify that the reproduced run is consistent with the ORBIT execution playbook:

1. ORBIT beats LZ4 on the mixed corpus.
2. ORBIT beats LZMA on the mixed corpus.
3. Regret is non-zero and decreases over time.
4. The mixed-corpus regret curve shows a strong downward trend across runs.
5. Standard deviation of repeated measurements stays below 0.03 where reported.
6. The ablation study produces ranked rows with the full feature set as the best or near-best baseline.
7. The block size sweep emits a convergence block for at least one tested block size.
8. The reproducibility manifest is generated and records a valid ORBIT version, dataset path, and runtime metadata.

## 8. Paper Figure Reproduction

- `outputs/core_comparison.json` -> paper results behind the core comparison table.
- `outputs/regret_curve_aggregated.json` and `outputs/regret_plot_data.json` -> Fig. 1, Normalized Cumulative Regret over Blocks.
- `outputs/block_size_sweep.json` and `outputs/block_size_plot.json` -> Fig. 2, Block Size Sensitivity.
- `outputs/ablation_results.json` and `outputs/ablation_table.json` -> Fig. 4 and the ablation table.
- `outputs/table1.json` -> Table 1, the main comparison table used in the paper.

## 9. Running the Full Pipeline

To reproduce all experiment outputs and derived tables in one pass, run:

```bash
python run_experiments.py
```

This command generates the core comparison, regret curves, ablation results, block size sweep, and the derived summary files under `outputs/`.
