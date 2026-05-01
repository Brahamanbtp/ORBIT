# Contributing to ORBIT

## Welcome

Thank you for your interest in contributing to ORBIT. This project is a research codebase for contextual bandit-based compression routing, so contributions are especially valuable when they improve reproducibility, clarity, experimental rigor, or codec-routing performance. We welcome bug fixes, documentation improvements, new codecs, new features, validation helpers, and experiment refinements.

## Getting Started

1. Fork the repository and clone your fork locally.
2. Install the project dependencies:

```bash
pip install -r requirements.txt
```

3. Run the smoke test to confirm your environment is set up correctly:

```bash
python smoke_test.py
```

4. Make sure all 5 checks in the smoke test pass before you contribute changes.

## Code Standards

- Use Python 3.10+ compatible syntax.
- Use 4-space indentation and no tabs.
- Add type hints to all public functions.
- Add docstrings to all classes and public methods.
- Avoid circular imports between modules.

## Adding a New Codec

1. Subclass `CodecAdapter` in `orbit_codecs/`.
2. Set a unique `codec_id` class attribute.
3. Register the codec in `orbit_codecs/__init__.py`.
4. Run `validate_all_codecs()` to verify the codec integrates correctly.

## Adding a New Feature

1. Add the feature module in `features/`.
2. Register the feature in `features/extractor.py`.
3. Run the ablation study to verify the feature's contribution.

## Pull Request Process

1. Create a focused branch for your change.
2. Run the smoke test and any relevant experiment or validation scripts before opening a pull request.
3. Describe the change clearly, including any dataset, codec, or evaluation impact.
4. Keep the pull request scope narrow and update documentation if behavior changes.

## Reporting Issues

If you find a problem, please report it with enough detail to reproduce it. Include the command you ran, the dataset or configuration used, the observed behavior, and any relevant logs or output files. For research-facing issues, it is especially helpful to note whether the problem affects reproducibility, evaluation results, or codec selection behavior.
