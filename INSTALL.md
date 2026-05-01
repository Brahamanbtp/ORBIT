# ORBIT: Installation Guide

## 1. System Requirements

1. Python 3.10 or higher.
2. `pip` or `conda` for package management.
3. At least 2GB of disk space for the evaluation corpora and generated outputs.
4. At least 4GB of RAM for running the experiments comfortably.

## 2. Quick Install

1. Clone the repository and enter the project directory.

```bash
git clone <repo-url>
cd ORBIT
```

2. Install the Python dependencies.

```bash
pip install -r requirements.txt
```

## 3. Codec Dependency Verification

1. Verify that the ORBIT codec registry is available and contains the expected codecs.

```bash
python -c "from orbit_codecs import available_codecs; print(available_codecs())"
```

2. The expected output is a list of 4 codec names.

## 4. Conda Alternative

1. If you prefer Conda, create an environment from the repository's `environment.yml` file.

```bash
conda env create -f environment.yml
```

2. Activate the environment before running ORBIT commands.

```bash
conda activate orbit
```

## 5. Verify Installation

1. Run the smoke test to confirm the environment is working.

```bash
python smoke_test.py
```

2. All 5 checks must show `PASS`.

## 6. Troubleshooting

1. If you see `ImportError` for `lz4`, install it directly:

```bash
pip install lz4
```

2. If you see `ImportError` for `zstandard`, install it directly:

```bash
pip install zstandard
```

3. If you see `ModuleNotFoundError` for `orbit_codecs`, make sure you are running commands from the project root directory.

## 7. Platform Notes

1. Tested on Ubuntu 24.04.
2. Tested on macOS 13 and later.
3. Tested on Windows 11 with WSL2.
