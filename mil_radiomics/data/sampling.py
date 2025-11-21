"""
Sampling utilities for handling class imbalance in MIL-radiomics.

This module provides weighted sampling strategies to address class imbalance
in medical imaging datasets.
"""

from collections import Counter
from typing import Union, List, Optional, Tuple, Dict
import numpy as np
from torch.utils.data import WeightedRandomSampler
from logger import get_logger

logger = get_logger(__name__)


def create_weighted_sampler(
    labels: Union[List[int], np.ndarray],
    oversample_factor: float = 1.0
) -> Tuple[Optional[WeightedRandomSampler], Dict[int, float]]:
    """
    Create a weighted random sampler for oversampling the minority class.

    This function calculates class weights inversely proportional to class frequency
    and creates a sampler that can be used with PyTorch DataLoader to handle
    class imbalance through weighted sampling.

    Args:
        labels: Class labels for all samples in the dataset
        oversample_factor: Factor to multiply minority class weight
                          - 1.0: balanced sampling (equal probability for all classes)
                          - >1.0: oversample minority class (higher probability)
                          - 0.0: no oversampling (uniform sampling, returns None)

    Returns:
        Tuple of:
            - WeightedRandomSampler or None: Sampler for DataLoader.
              None if oversample_factor is 0 (no oversampling)
            - Dict mapping class labels to their weights

    Example:
        >>> labels = [0, 0, 0, 0, 1]  # Imbalanced: 80% class 0, 20% class 1
        >>> sampler, weights = create_weighted_sampler(labels, oversample_factor=2.0)
        >>> loader = DataLoader(dataset, batch_size=16, sampler=sampler)
        >>> print(f"Class weights: {weights}")
        Class weights: {0: 1.25, 1: 10.0}

    Notes:
        - Uses replacement=True to allow oversampling
        - Minority class is determined automatically
        - Returns class weights for logging/analysis
    """
    # If oversample_factor is 0, return None to indicate no oversampling
    if oversample_factor == 0:
        logger.info("Oversample factor is 0, using uniform sampling (no weighted sampler)")
        # Still calculate and return class weights for information
        label_counts = Counter(labels)
        n_samples = len(labels)
        class_weights = {cls: n_samples / count for cls, count in label_counts.items()}
        return None, class_weights

    # Count instances per class
    label_counts = Counter(labels)
    n_classes = len(label_counts)

    logger.info(f"Class distribution: {dict(label_counts)}")

    # Calculate weights per class (inversely proportional to class frequency)
    n_samples = len(labels)
    class_weights: Dict[int, float] = {
        cls: n_samples / count for cls, count in label_counts.items()
    }

    # Apply oversample factor to minority class (for binary classification)
    if n_classes == 2 and 0 in class_weights and 1 in class_weights:
        minority_class = 0 if label_counts[0] < label_counts[1] else 1
        majority_class = 1 - minority_class

        original_weight = class_weights[minority_class]
        class_weights[minority_class] *= oversample_factor

        logger.info(
            f"Minority class: {minority_class} "
            f"(count: {label_counts[minority_class]}, weight: {original_weight:.2f} → {class_weights[minority_class]:.2f})"
        )
        logger.info(
            f"Majority class: {majority_class} "
            f"(count: {label_counts[majority_class]}, weight: {class_weights[majority_class]:.2f})"
        )
    elif n_classes > 2:
        # For multi-class, apply oversample factor to all minority classes
        max_count = max(label_counts.values())
        for cls in class_weights:
            if label_counts[cls] < max_count:
                class_weights[cls] *= oversample_factor

    # Assign weights to each sample
    weights = [class_weights[label] for label in labels]

    # Create sampler
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True
    )

    logger.info(f"Created WeightedRandomSampler with {len(weights)} samples")

    return sampler, class_weights
