# ORBIT: Online Routing of Blocks via Incremental Contextual Bandit-Guided Compression

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

ORBIT is a block-level adaptive compression framework that routes each block to a codec using a LinUCB contextual bandit. For every block, ORBIT extracts discriminative features and selects among LZ4, Zstd, LZMA, and Raw backends to balance compression ratio, throughput, and online decision quality. The system is designed for sequential, regret-aware compression where codec selection adapts to the evolving statistics of the input stream.

## Key Contributions

- Online routing framework for adaptive per-block codec selection using a contextual bandit policy.
- Compression-discriminative feature design that captures block-level structure for codec choice.
- Regret-based evaluation protocol for measuring routing quality over time and across datasets.

## Architecture Overview

ORBIT follows a simple streaming pipeline:

Input → Block Split → Feature Extraction → LinUCB Decision → Codec → Output

Each input stream is partitioned into blocks, transformed into feature vectors, scored by LinUCB, and routed to the selected codec before being written to the compressed output.

## Installation

ORBIT targets Python 3.10+.

```bash
python -m pip install -r requirements.txt
```

If you are working in a fresh environment, create and activate a virtual environment first, then install the project dependencies from `requirements.txt`.

## Usage

Run the main ORBIT pipeline:

```bash
python run_orbit.py
```

Run the experiment suite:

```bash
python run_experiments.py
```

Both entry points are designed to use the repository's configured inputs, codecs, and evaluation routines.

## Experiments

The repository includes four core experiments:

- Core comparison
- Regret curve
- Ablation study
- Block size sweep

These experiments evaluate routing quality, codec selection behavior, feature importance, and sensitivity to block granularity.

## Results Summary

ORBIT demonstrates consistent gains over fixed-codec baselines in the reported experiments:

- Outperforms LZ4 by up to 20.4%.
- Outperforms LZMA by up to 9.9%.
- Regret decreases 35-fold across 5002 blocks.

These results highlight the benefit of online, feature-driven codec routing for heterogeneous data streams.

## Citation

If you use ORBIT in your research, please cite it as follows:

```bibtex
@article{orbit2026,
	title   = {ORBIT: Online Routing of Blocks via Incremental Contextual Bandit-Guided Compression},
	author  = {Pranay Sharma},
	journal = {IEEE Transactions on ...},
	year    = {2026},
	note    = {BibTeX placeholder; replace with the final publication details.}
}
```

## License

ORBIT is released under the MIT License. See the repository `LICENSE` file for full terms.