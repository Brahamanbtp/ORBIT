from dataclasses import dataclass
from typing import List
import os
import yaml

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
