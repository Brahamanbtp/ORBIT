from dataclasses import dataclass
from typing import List
import os
import yaml
from configs.schema import ORBITConfig

@dataclass
class DatasetEntry:
    name: str
    path: str
    size_bytes: int
    content_type: str
    description: str

def load_dataset_manifest(yaml_path: str) -> List[DatasetEntry]:
    """
    Load a dataset manifest YAML file into a list of DatasetEntry objects.
    """
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    entries = []
    for entry in data:
        entries.append(DatasetEntry(**entry))
    return entries

def validate_manifest(entries: List[DatasetEntry]) -> List[str]:
    """
    Return a list of file paths from entries that are missing or unreadable.
    """
    missing = []
    for entry in entries:
        if not os.path.isfile(entry.path) or not os.access(entry.path, os.R_OK):
            missing.append(entry.path)
    return missing


def preflight_check(entries: list[DatasetEntry], config: ORBITConfig) -> None:
    """
    Validate dataset files and ensure they are large enough for meaningful training.
    """
    min_required_size = config.block_size * (config.n_actions * 10)
    warning_threshold = config.block_size * 10

    missing_paths: list[str] = []
    for entry in entries:
        if not os.path.isfile(entry.path):
            missing_paths.append(entry.path)
            continue

        assert os.access(entry.path, os.R_OK), (
            f"Dataset file is not readable: {entry.path}"
        )

        actual_size = os.path.getsize(entry.path)
        assert int(entry.size_bytes) == int(actual_size), (
            f"Size mismatch for {entry.path}: manifest size_bytes={entry.size_bytes}, actual={actual_size}"
        )

        assert actual_size >= min_required_size, (
            f"Dataset file too small for meaningful bandit learning: {entry.path}. "
            f"Required >= {min_required_size} bytes, got {actual_size} bytes."
        )

        if actual_size < warning_threshold:
            print(
                f"WARNING: dataset '{entry.name}' at {entry.path} is below 10x block threshold "
                f"({actual_size} < {warning_threshold} bytes)."
            )

    if missing_paths:
        raise RuntimeError(
            "Missing dataset files: " + ", ".join(missing_paths)
        )
