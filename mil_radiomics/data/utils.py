"""
Utility functions for data loading in MIL-radiomics.

This module contains helper functions for data integrity checking,
checksum computation, and batch collation.
"""

import hashlib
import json
import os
from typing import List, Tuple, Dict, Any
import torch
from logger import get_logger

logger = get_logger(__name__)


def compute_data_dir_checksum(data_dir: str) -> str:
    """
    Compute a checksum of the data directory to detect changes.

    Args:
        data_dir: Directory containing .pkl files

    Returns:
        SHA-256 checksum string representing the directory state

    Example:
        >>> checksum = compute_data_dir_checksum('/path/to/data')
        >>> print(f"Directory checksum: {checksum[:16]}...")
    """
    pkl_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.pkl')])
    file_info: List[Dict[str, Any]] = []

    for f in pkl_files:
        file_path = os.path.join(data_dir, f)
        file_info.append({
            'name': f,
            'size': os.path.getsize(file_path),
            'mtime': os.path.getmtime(file_path)
        })

    # Create a string representation and hash it
    dir_info_str = json.dumps(file_info, sort_keys=True)
    # Using SHA-256 for secure integrity checking (MD5 is cryptographically broken)
    checksum = hashlib.sha256(dir_info_str.encode()).hexdigest()

    return checksum


def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]]) -> Tuple[torch.Tensor, torch.Tensor, List[Dict[str, Any]]]:
    """
    Custom collate function for DataLoader that handles variable-length sequences.

    Args:
        batch: List of tuples containing (features, label, identifiers)
               - features: Tensor of shape [n_patches, feature_dim]
               - label: Tensor scalar
               - identifiers: Dict with patient_id, center, etc.

    Returns:
        Tuple of:
            - features: Stacked tensor of shape [batch_size, max_patches, feature_dim]
            - labels: Stacked tensor of shape [batch_size]
            - identifiers: List of identifier dicts

    Example:
        >>> loader = DataLoader(dataset, batch_size=16, collate_fn=collate_fn)
        >>> for features, labels, ids in loader:
        ...     print(features.shape)  # [16, max_patches, feature_dim]
    """
    # Separate the batch into features, labels, and identifiers
    features_list: List[torch.Tensor] = []
    labels_list: List[torch.Tensor] = []
    identifiers_list: List[Dict[str, Any]] = []

    for features, label, identifiers in batch:
        features_list.append(features)
        labels_list.append(label)
        identifiers_list.append(identifiers)

    # Stack features and labels
    # All features should already be padded to max_patches by the dataset
    features_batch = torch.stack(features_list, dim=0)
    labels_batch = torch.stack(labels_list, dim=0)

    return features_batch, labels_batch, identifiers_list
