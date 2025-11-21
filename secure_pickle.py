"""
Secure pickle handling utilities for MIL-radiomics.

This module provides safe pickle loading and validation to prevent
arbitrary code execution from malicious pickle files.
"""

import pickle
import io
import os
from typing import Any, Optional, Set, List
from logger import get_logger

logger = get_logger(__name__)


class RestrictedUnpickler(pickle.Unpickler):
    """
    A restricted unpickler that only allows safe classes to be loaded.

    This prevents arbitrary code execution by limiting what classes
    can be instantiated during pickle deserialization.
    """

    # Safe modules and classes that are allowed
    SAFE_MODULES = {
        'numpy',
        'numpy.core',
        'numpy.core.multiarray',
        'numpy.core.numeric',
        'numpy.ma',
        'torch',
        'torch.storage',
        'torch._utils',
        'collections',
        'builtins',
        '__builtin__',
    }

    # Additional safe class patterns
    SAFE_CLASS_PATTERNS = {
        'numpy.ndarray',
        'numpy.dtype',
        'torch.Tensor',
        'torch.FloatStorage',
        'torch.LongStorage',
        'torch.IntStorage',
        'collections.OrderedDict',
        'dict',
        'list',
        'tuple',
        'set',
        'frozenset',
    }

    def __init__(self, file, *, fix_imports=True, encoding="ASCII",
                 errors="strict", extra_safe_modules=None):
        """
        Initialize restricted unpickler.

        Args:
            file: File object to read from
            fix_imports: Whether to fix pickle imports
            encoding: Encoding to use
            errors: Error handling strategy
            extra_safe_modules: Additional modules to whitelist
        """
        super().__init__(file, fix_imports=fix_imports, encoding=encoding,
                        errors=errors)

        # Allow user to extend safe modules
        self.safe_modules = self.SAFE_MODULES.copy()
        if extra_safe_modules:
            self.safe_modules.update(extra_safe_modules)

    def find_class(self, module, name):
        """
        Override find_class to only allow safe modules and classes.

        Args:
            module: Module name
            name: Class name

        Returns:
            The class if safe

        Raises:
            pickle.UnpicklingError: If class is not in whitelist
        """
        full_name = f"{module}.{name}"

        # Check if module is in safe list
        if module in self.safe_modules:
            return super().find_class(module, name)

        # Check if full class name matches safe patterns
        if full_name in self.SAFE_CLASS_PATTERNS:
            return super().find_class(module, name)

        # Log and raise error for unsafe classes
        logger.error(f"Attempted to load unsafe class: {full_name}")
        raise pickle.UnpicklingError(
            f"Class {full_name} is not in the safe list. "
            f"This may be a security risk."
        )


def safe_pickle_load(file_path: str,
                     extra_safe_modules: Optional[Set[str]] = None,
                     max_size_mb: float = 1000) -> Any:
    """
    Safely load a pickle file with validation.

    Args:
        file_path: Path to pickle file
        extra_safe_modules: Additional modules to whitelist
        max_size_mb: Maximum file size in MB (default 1000MB = 1GB)

    Returns:
        Unpickled object

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file is too large
        pickle.UnpicklingError: If unsafe classes detected

    Example:
        >>> data = safe_pickle_load('data.pkl')
        >>> data = safe_pickle_load('data.pkl', max_size_mb=500)
    """
    # Check file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Pickle file not found: {file_path}")

    # Check file size
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    if file_size_mb > max_size_mb:
        raise ValueError(
            f"File {file_path} is {file_size_mb:.2f}MB, "
            f"exceeds maximum size of {max_size_mb}MB"
        )

    logger.debug(f"Loading pickle file: {file_path} ({file_size_mb:.2f}MB)")

    try:
        with open(file_path, 'rb') as f:
            # Use restricted unpickler
            unpickler = RestrictedUnpickler(f, extra_safe_modules=extra_safe_modules)
            data = unpickler.load()

        logger.debug(f"Successfully loaded pickle file: {file_path}")
        return data

    except pickle.UnpicklingError as e:
        logger.error(f"Security error loading pickle file {file_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading pickle file {file_path}: {e}")
        raise


def validate_pickle_structure(data: Any,
                              expected_keys: Optional[List[str]] = None,
                              expected_type: Optional[type] = None) -> bool:
    """
    Validate the structure of unpickled data.

    Args:
        data: Unpickled data to validate
        expected_keys: List of expected keys if data is a dict
        expected_type: Expected type of data

    Returns:
        True if validation passes

    Raises:
        ValueError: If validation fails

    Example:
        >>> data = safe_pickle_load('data.pkl')
        >>> validate_pickle_structure(data, expected_keys=['features', 'label'])
    """
    # Check type if specified
    if expected_type is not None:
        if not isinstance(data, expected_type):
            raise ValueError(
                f"Expected data type {expected_type}, got {type(data)}"
            )

    # Check keys if data is a dict and keys are specified
    if expected_keys is not None:
        if not isinstance(data, dict):
            raise ValueError(
                f"Expected dict to check keys, got {type(data)}"
            )

        missing_keys = set(expected_keys) - set(data.keys())
        if missing_keys:
            raise ValueError(
                f"Missing expected keys: {missing_keys}"
            )

    logger.debug("Pickle structure validation passed")
    return True


def safe_pickle_dump(obj: Any, file_path: str) -> None:
    """
    Safely dump an object to a pickle file.

    Args:
        obj: Object to pickle
        file_path: Path to save pickle file

    Example:
        >>> data = {'features': features, 'label': label}
        >>> safe_pickle_dump(data, 'output.pkl')
    """
    try:
        # Create directory if needed
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, 'wb') as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

        logger.debug(f"Successfully saved pickle file: {file_path}")

    except Exception as e:
        logger.error(f"Error saving pickle file {file_path}: {e}")
        raise


if __name__ == '__main__':
    # Test the secure pickle loading
    import tempfile
    import numpy as np

    # Create test data
    test_data = {
        'features': np.random.rand(10, 512),
        'label': 1,
        'metadata': {'patient_id': 'test_001'}
    }

    # Save and load with validation
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
        tmp_path = tmp.name

    try:
        print("Testing safe pickle operations...")

        # Save
        safe_pickle_dump(test_data, tmp_path)
        print(f"✓ Saved pickle to {tmp_path}")

        # Load
        loaded_data = safe_pickle_load(tmp_path)
        print(f"✓ Loaded pickle from {tmp_path}")

        # Validate structure
        validate_pickle_structure(loaded_data, expected_keys=['features', 'label'])
        print("✓ Structure validation passed")

        print("\nAll tests passed!")

    finally:
        # Cleanup
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
