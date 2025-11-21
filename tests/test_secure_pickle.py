"""
Tests for secure pickle handling.
"""

import pytest
import pickle
import numpy as np
import torch
from pathlib import Path
from secure_pickle import (
    safe_pickle_load,
    safe_pickle_dump,
    validate_pickle_structure,
    RestrictedUnpickler
)


class TestSafePickleLoad:
    """Tests for safe_pickle_load function."""

    def test_load_safe_pickle(self, temp_dir):
        """Test loading a safe pickle file."""
        # Create a safe pickle file
        test_file = temp_dir / "safe_data.pkl"
        data = {
            'features': np.array([[1, 2, 3], [4, 5, 6]]),
            'label': 1
        }

        with open(test_file, 'wb') as f:
            pickle.dump(data, f)

        # Load with safe_pickle_load
        loaded_data = safe_pickle_load(str(test_file))

        assert 'features' in loaded_data
        assert 'label' in loaded_data
        assert np.array_equal(loaded_data['features'], data['features'])
        assert loaded_data['label'] == 1

    def test_load_nonexistent_file(self):
        """Test loading a file that doesn't exist."""
        with pytest.raises(FileNotFoundError):
            safe_pickle_load('nonexistent_file.pkl')

    def test_load_file_too_large(self, temp_dir):
        """Test loading a file that exceeds size limit."""
        test_file = temp_dir / "large_data.pkl"

        # Create a small file
        data = {'small': 'data'}
        with open(test_file, 'wb') as f:
            pickle.dump(data, f)

        # Try to load with very small max_size
        with pytest.raises(ValueError, match="exceeds maximum size"):
            safe_pickle_load(str(test_file), max_size_mb=0.000001)

    def test_load_with_torch_tensors(self, temp_dir):
        """Test loading pickle with PyTorch tensors."""
        test_file = temp_dir / "torch_data.pkl"

        data = {
            'tensor': torch.randn(5, 10),
            'label': 1
        }

        with open(test_file, 'wb') as f:
            pickle.dump(data, f)

        loaded_data = safe_pickle_load(str(test_file))

        assert 'tensor' in loaded_data
        assert torch.is_tensor(loaded_data['tensor'])
        assert loaded_data['tensor'].shape == (5, 10)


class TestSafePickleDump:
    """Tests for safe_pickle_dump function."""

    def test_dump_basic_data(self, temp_dir):
        """Test dumping basic data."""
        test_file = temp_dir / "dump_test.pkl"

        data = {
            'features': np.random.randn(10, 512),
            'label': 1,
            'metadata': {'patient_id': 'test_001'}
        }

        safe_pickle_dump(data, str(test_file))

        # Verify file exists
        assert test_file.exists()

        # Load and verify
        with open(test_file, 'rb') as f:
            loaded = pickle.load(f)

        assert 'features' in loaded
        assert 'label' in loaded
        assert loaded['metadata']['patient_id'] == 'test_001'

    def test_dump_creates_directory(self, temp_dir):
        """Test that dump creates directory if needed."""
        subdir = temp_dir / "subdir" / "nested"
        test_file = subdir / "data.pkl"

        data = {'test': 'data'}
        safe_pickle_dump(data, str(test_file))

        assert test_file.exists()


class TestValidatePickleStructure:
    """Tests for validate_pickle_structure function."""

    def test_validate_with_expected_keys(self):
        """Test validation with expected keys."""
        data = {'features': np.array([1, 2, 3]), 'label': 1}

        # Should pass
        assert validate_pickle_structure(data, expected_keys=['features', 'label'])

    def test_validate_missing_keys(self):
        """Test validation with missing keys."""
        data = {'features': np.array([1, 2, 3])}

        with pytest.raises(ValueError, match="Missing expected keys"):
            validate_pickle_structure(data, expected_keys=['features', 'label'])

    def test_validate_expected_type(self):
        """Test validation with expected type."""
        data = {'features': np.array([1, 2, 3]), 'label': 1}

        # Should pass
        assert validate_pickle_structure(data, expected_type=dict)

    def test_validate_wrong_type(self):
        """Test validation with wrong type."""
        data = [1, 2, 3]  # List instead of dict

        with pytest.raises(ValueError, match="Expected data type"):
            validate_pickle_structure(data, expected_type=dict)

    def test_validate_keys_on_non_dict(self):
        """Test validation keys check on non-dict data."""
        data = [1, 2, 3]

        with pytest.raises(ValueError, match="Expected dict to check keys"):
            validate_pickle_structure(data, expected_keys=['features'])


class TestRestrictedUnpickler:
    """Tests for RestrictedUnpickler class."""

    def test_load_safe_classes(self, temp_dir):
        """Test loading pickle with safe classes."""
        test_file = temp_dir / "safe_classes.pkl"

        # Create pickle with safe classes
        data = {
            'numpy_array': np.array([1, 2, 3]),
            'torch_tensor': torch.tensor([1.0, 2.0, 3.0]),
            'list': [1, 2, 3],
            'dict': {'key': 'value'}
        }

        with open(test_file, 'wb') as f:
            pickle.dump(data, f)

        # Load with RestrictedUnpickler
        with open(test_file, 'rb') as f:
            unpickler = RestrictedUnpickler(f)
            loaded = unpickler.load()

        assert 'numpy_array' in loaded
        assert 'torch_tensor' in loaded
        assert isinstance(loaded['list'], list)
        assert isinstance(loaded['dict'], dict)


class TestIntegration:
    """Integration tests for secure pickle handling."""

    def test_roundtrip_numpy_data(self, temp_dir):
        """Test roundtrip save and load of numpy data."""
        test_file = temp_dir / "roundtrip.pkl"

        original_data = {
            'features': np.random.randn(100, 512),
            'labels': np.array([0, 1, 0, 1, 1]),
            'metadata': {
                'patient_ids': ['p001', 'p002', 'p003'],
                'centers': ['A', 'B', 'C']
            }
        }

        # Save
        safe_pickle_dump(original_data, str(test_file))

        # Load
        loaded_data = safe_pickle_load(str(test_file))

        # Verify
        assert np.array_equal(loaded_data['features'], original_data['features'])
        assert np.array_equal(loaded_data['labels'], original_data['labels'])
        assert loaded_data['metadata'] == original_data['metadata']

    def test_roundtrip_torch_data(self, temp_dir):
        """Test roundtrip save and load of torch data."""
        test_file = temp_dir / "torch_roundtrip.pkl"

        original_data = {
            'features': torch.randn(50, 256),
            'label': torch.tensor(1, dtype=torch.long)
        }

        # Save
        safe_pickle_dump(original_data, str(test_file))

        # Load
        loaded_data = safe_pickle_load(str(test_file))

        # Verify
        assert torch.allclose(loaded_data['features'], original_data['features'])
        assert loaded_data['label'] == original_data['label']

    def test_validate_loaded_structure(self, temp_dir):
        """Test validation of loaded pickle structure."""
        test_file = temp_dir / "validate_test.pkl"

        data = {
            'features': np.random.randn(10, 512),
            'label': 1,
            'patient_id': 'test_001'
        }

        safe_pickle_dump(data, str(test_file))
        loaded = safe_pickle_load(str(test_file))

        # Validate structure
        assert validate_pickle_structure(
            loaded,
            expected_keys=['features', 'label', 'patient_id'],
            expected_type=dict
        )
