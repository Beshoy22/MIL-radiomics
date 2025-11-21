"""
Feature preprocessing utilities for MIL-radiomics.

This module handles the complex task of processing pickle files containing
medical imaging patch features in various formats and normalizing them to
a consistent [n_patches, feature_dim] tensor format.
"""

import os
import pickle
from typing import List, Dict, Any, Tuple, Union
import numpy as np
import torch
from logger import get_logger

logger = get_logger(__name__)


def process_pkl_file(
    pkl_file: str,
    endpoint: str,
    feature_dim: int = 512
) -> Tuple[int, List[Dict[str, Any]]]:
    """
    Process a single .pkl file and extract instances with the specified endpoint.

    Handles feature structures where features can be:
    - List of tensors/arrays with varying shapes
    - Multi-dimensional tensors/arrays
    - Pre-stacked tensors/arrays

    All features are normalized to [n_patches, feature_dim] tensor format.

    Args:
        pkl_file: Path to the .pkl file containing instances
        endpoint: Which endpoint label to use (e.g., 'OS_6', 'OS_12')
        feature_dim: Expected dimension of features (default: 512)

    Returns:
        Tuple of:
            - max_patches: Maximum number of patches across all instances in this file
            - instances: List of dictionaries containing:
                - features: Tensor of shape [n_patches, feature_dim]
                - label: Binary label (0 or 1)
                - center: Center identifier (filename)
                - patient_id: Patient identifier

    Example:
        >>> max_patches, instances = process_pkl_file('patient_001.pkl', 'OS_6', 512)
        >>> print(f"Found {len(instances)} instances, max patches: {max_patches}")
        Found 5 instances, max patches: 127

    Notes:
        - Skips instances without the specified endpoint
        - Handles varying feature formats automatically
        - Pads or truncates features to match feature_dim
        - Logs warnings for unsupported formats
    """
    try:
        instances: List[Dict[str, Any]] = []
        max_patches = 0

        with open(pkl_file, 'rb') as f:
            instances_list = pickle.load(f)
            center = os.path.basename(pkl_file)  # Use filename as center identifier

            for idx, instance in enumerate(instances_list):
                # Skip instances without the specified endpoint
                if endpoint not in instance:
                    continue

                # Get features - these can be a list of tensors or multi-dimensional
                features = instance['features']

                # Process features to standard format
                processed_features = _process_features(
                    features, pkl_file, feature_dim
                )

                if processed_features is None:
                    continue

                # Update max_patches
                n_patches = processed_features.shape[0]
                max_patches = max(max_patches, n_patches)

                # Get and normalize label
                label = _normalize_label(instance[endpoint])

                # Create patient ID
                patient_id = instance.get('patient_id', f"{center}_{idx}")

                instances.append({
                    'features': processed_features,
                    'label': label,
                    'center': center,
                    'patient_id': patient_id
                })

        return max_patches, instances
    except Exception as e:
        logger.error(f"Error processing {pkl_file}: {e}", exc_info=True)
        return 0, []


def _process_features(
    features: Union[List, torch.Tensor, np.ndarray],
    pkl_file: str,
    feature_dim: int
) -> Union[torch.Tensor, None]:
    """
    Process features into standard [n_patches, feature_dim] format.

    Args:
        features: Features in various formats
        pkl_file: Path to pkl file (for logging)
        feature_dim: Target feature dimension

    Returns:
        Processed features tensor or None if processing failed
    """
    processed_features: Union[torch.Tensor, None] = None

    if isinstance(features, list):
        processed_features = _process_list_features(features, pkl_file, feature_dim)
    elif isinstance(features, torch.Tensor):
        processed_features = _process_tensor_features(features, feature_dim)
    elif isinstance(features, np.ndarray):
        processed_features = _process_array_features(features, feature_dim)
    else:
        logger.warning(f"Unsupported feature type {type(features)} in {pkl_file}")
        return None

    if processed_features is None:
        return None

    # Final normalization
    return _normalize_features(processed_features, feature_dim)


def _process_list_features(
    features: List,
    pkl_file: str,
    feature_dim: int
) -> Union[torch.Tensor, None]:
    """Process list of features (tensors/arrays)."""
    patch_count = len(features)

    if patch_count == 0:
        return None

    # Check the type of the first patch
    if isinstance(features[0], torch.Tensor):
        return _stack_tensor_list(features, pkl_file, feature_dim)
    elif isinstance(features[0], np.ndarray):
        return _stack_array_list(features, pkl_file, feature_dim)
    else:
        # Handle other types (e.g., lists of lists)
        return _process_nested_list(features, pkl_file, feature_dim)


def _stack_tensor_list(
    features: List[torch.Tensor],
    pkl_file: str,
    feature_dim: int
) -> Union[torch.Tensor, None]:
    """Stack list of tensors with normalization if needed."""
    try:
        # Try direct stacking first
        return torch.stack(features)
    except (RuntimeError, ValueError, TypeError) as e:
        # If tensors have different shapes, normalize them
        logger.debug(f"Direct tensor stacking failed in {pkl_file}: {e}. Normalizing features.")

        norm_features: List[torch.Tensor] = []
        for feature in features:
            normalized = _normalize_single_tensor(feature, feature_dim)
            if normalized is not None:
                norm_features.append(normalized)

        if norm_features:
            return torch.stack(norm_features)
        return None


def _normalize_single_tensor(
    feature: torch.Tensor,
    feature_dim: int
) -> Union[torch.Tensor, None]:
    """Normalize a single tensor to [feature_dim] shape."""
    if feature.dim() > 1:
        # Flatten multi-dimensional tensor
        try:
            flattened = feature.reshape(-1)
            if flattened.size(0) >= feature_dim:
                return flattened[:feature_dim]
            else:
                padded = torch.zeros(feature_dim, dtype=torch.float32)
                padded[:flattened.size(0)] = flattened
                return padded
        except (RuntimeError, ValueError) as e:
            logger.warning(f"Skipping patch that couldn't be reshaped: {e}")
            return None
    elif feature.dim() == 1:
        # For 1D vectors, ensure they're feature_dim
        if feature.shape[0] >= feature_dim:
            return feature[:feature_dim]
        else:
            padded = torch.zeros(feature_dim, dtype=torch.float32)
            padded[:feature.shape[0]] = feature
            return padded
    else:
        # Skip empty tensors
        return None


def _stack_array_list(
    features: List[np.ndarray],
    pkl_file: str,
    feature_dim: int
) -> Union[torch.Tensor, None]:
    """Stack list of numpy arrays with normalization if needed."""
    try:
        # Try direct stacking first
        return torch.tensor(np.stack(features), dtype=torch.float32)
    except (ValueError, RuntimeError, TypeError) as e:
        # If arrays have different shapes, normalize them
        logger.debug(f"Direct numpy stacking failed in {pkl_file}: {e}. Normalizing features.")

        norm_features: List[np.ndarray] = []
        for feature in features:
            normalized = _normalize_single_array(feature, feature_dim)
            if normalized is not None:
                norm_features.append(normalized)

        if norm_features:
            return torch.tensor(np.stack(norm_features), dtype=torch.float32)
        return None


def _normalize_single_array(
    feature: np.ndarray,
    feature_dim: int
) -> Union[np.ndarray, None]:
    """Normalize a single numpy array to [feature_dim] shape."""
    if feature.ndim > 1:
        try:
            flattened = feature.reshape(-1)
            if flattened.shape[0] >= feature_dim:
                return flattened[:feature_dim]
            else:
                padded = np.zeros(feature_dim, dtype=np.float32)
                padded[:flattened.shape[0]] = flattened
                return padded
        except (ValueError, RuntimeError) as e:
            logger.warning(f"Skipping numpy patch that couldn't be reshaped: {e}")
            return None
    elif feature.ndim == 1:
        if feature.shape[0] >= feature_dim:
            return feature[:feature_dim]
        else:
            padded = np.zeros(feature_dim, dtype=np.float32)
            padded[:feature.shape[0]] = feature
            return padded
    else:
        return None


def _process_nested_list(
    features: List,
    pkl_file: str,
    feature_dim: int
) -> Union[torch.Tensor, None]:
    """Process nested lists or other list types."""
    try:
        raw_array = np.array(features)
        if raw_array.ndim == 2 and raw_array.shape[1] == feature_dim:
            return torch.tensor(raw_array, dtype=torch.float32)
        else:
            processed_features = torch.tensor(np.array(features), dtype=torch.float32)
            if processed_features.dim() > 1:
                processed_features = processed_features.reshape(-1, processed_features.shape[-1])
                if processed_features.shape[1] != feature_dim:
                    if processed_features.shape[1] > feature_dim and processed_features.shape[0] == feature_dim:
                        processed_features = processed_features.transpose(0, 1)
            elif processed_features.dim() == 1:
                processed_features = processed_features.reshape(1, -1)
            return processed_features
    except (ValueError, RuntimeError, TypeError) as e:
        logger.warning(f"Could not process features in {pkl_file}: {e}")
        return None


def _process_tensor_features(
    features: torch.Tensor,
    feature_dim: int
) -> torch.Tensor:
    """Process tensor directly into standard format."""
    if features.dim() > 2:
        return _process_multidim_tensor(features, feature_dim)
    elif features.dim() == 2:
        return _process_2d_tensor(features, feature_dim)
    elif features.dim() == 1:
        return features.unsqueeze(0)
    else:
        return features


def _process_multidim_tensor(
    features: torch.Tensor,
    feature_dim: int
) -> torch.Tensor:
    """Process multi-dimensional tensor."""
    if features.shape[-1] == feature_dim:
        return features.reshape(-1, feature_dim)
    else:
        # Try to find feature dimension
        feature_dim_idx = None
        for i, dim_size in enumerate(features.shape):
            if dim_size == feature_dim:
                feature_dim_idx = i
                break

        if feature_dim_idx is not None:
            permutation = list(range(features.dim()))
            permutation.remove(feature_dim_idx)
            permutation.append(feature_dim_idx)
            permuted = features.permute(*permutation)
            return permuted.reshape(-1, feature_dim)
        else:
            processed = features.reshape(-1, features.shape[-1])
            if processed.shape[1] != feature_dim and processed.shape[0] == feature_dim:
                return processed.transpose(0, 1)
            return processed


def _process_2d_tensor(
    features: torch.Tensor,
    feature_dim: int
) -> torch.Tensor:
    """Process 2D tensor."""
    if features.shape[0] == feature_dim and features.shape[1] != feature_dim:
        return features.transpose(0, 1)
    return features


def _process_array_features(
    features: np.ndarray,
    feature_dim: int
) -> torch.Tensor:
    """Process numpy array directly into standard format."""
    if features.ndim > 2:
        return _process_multidim_array(features, feature_dim)
    elif features.ndim == 2:
        return _process_2d_array(features, feature_dim)
    elif features.ndim == 1:
        return torch.tensor(features, dtype=torch.float32).unsqueeze(0)
    else:
        return torch.tensor(features, dtype=torch.float32)


def _process_multidim_array(
    features: np.ndarray,
    feature_dim: int
) -> torch.Tensor:
    """Process multi-dimensional array."""
    if features.shape[-1] == feature_dim:
        return torch.tensor(features.reshape(-1, feature_dim), dtype=torch.float32)
    else:
        # Try to find feature dimension
        feature_dim_idx = None
        for i, dim_size in enumerate(features.shape):
            if dim_size == feature_dim:
                feature_dim_idx = i
                break

        if feature_dim_idx is not None:
            permutation = list(range(features.ndim))
            permutation.remove(feature_dim_idx)
            permutation.append(feature_dim_idx)
            permuted = np.transpose(features, permutation)
            return torch.tensor(permuted.reshape(-1, feature_dim), dtype=torch.float32)
        else:
            reshaped = features.reshape(-1, features.shape[-1])
            processed = torch.tensor(reshaped, dtype=torch.float32)
            if processed.shape[1] != feature_dim and processed.shape[0] == feature_dim:
                return processed.transpose(0, 1)
            return processed


def _process_2d_array(
    features: np.ndarray,
    feature_dim: int
) -> torch.Tensor:
    """Process 2D array."""
    if features.shape[0] == feature_dim and features.shape[1] != feature_dim:
        return torch.tensor(features.transpose(), dtype=torch.float32)
    return torch.tensor(features, dtype=torch.float32)


def _normalize_features(
    features: torch.Tensor,
    feature_dim: int
) -> torch.Tensor:
    """Final normalization to ensure [n_patches, feature_dim] format."""
    features = features.float()

    if features.dim() == 2:
        if features.shape[1] != feature_dim and features.shape[0] == feature_dim:
            features = features.transpose(0, 1)

        if features.shape[1] != feature_dim:
            if features.shape[1] < feature_dim:
                # Pad with zeros
                padded = torch.zeros(features.shape[0], feature_dim, dtype=torch.float32)
                padded[:, :features.shape[1]] = features
                features = padded
            else:
                # Truncate to feature_dim
                features = features[:, :feature_dim]

    elif features.dim() == 1:
        if features.shape[0] < feature_dim:
            padded = torch.zeros(feature_dim, dtype=torch.float32)
            padded[:features.shape[0]] = features
            features = padded.unsqueeze(0)
        else:
            features = features[:feature_dim].unsqueeze(0)

    elif features.dim() > 2:
        features = features.reshape(-1, feature_dim)

    return features


def _normalize_label(label: Any) -> int:
    """
    Normalize label to binary (0 or 1).

    Args:
        label: Label in various formats (bool, int, float, etc.)

    Returns:
        Binary label (0 or 1)
    """
    if isinstance(label, bool):
        return 1 if label else 0
    elif not isinstance(label, int):
        try:
            label = int(label)
        except (ValueError, TypeError):
            label = 1 if label else 0

    # Ensure label is either 0 or 1
    return 1 if label > 0 else 0
