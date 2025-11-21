"""
Pytest fixtures and configuration for MIL-radiomics tests.

This file contains shared fixtures that can be used across all test files.
"""

import pytest
import numpy as np
import torch
import tempfile
import os
import pickle
from pathlib import Path


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_features():
    """Generate sample feature tensor for testing."""
    np.random.seed(42)
    # Shape: [n_patches, feature_dim]
    n_patches = 10
    feature_dim = 512
    features = np.random.randn(n_patches, feature_dim).astype(np.float32)
    return features


@pytest.fixture
def sample_torch_features():
    """Generate sample PyTorch feature tensor for testing."""
    torch.manual_seed(42)
    n_patches = 10
    feature_dim = 512
    features = torch.randn(n_patches, feature_dim, dtype=torch.float32)
    return features


@pytest.fixture
def sample_instance_data():
    """Generate sample instance data with all fields."""
    return {
        'features': np.random.randn(10, 512).astype(np.float32),
        'label': 1,
        'patient_id': 'patient_001',
        'center': 'center_A',
        'OS_6': 1,
        'OS_12': 0,
        'OS_24': 0,
    }


@pytest.fixture
def sample_pkl_file(temp_dir, sample_instance_data):
    """Create a sample pickle file for testing."""
    pkl_file = temp_dir / "test_data.pkl"

    # Create a list of instances as expected by the dataloader
    instances = [sample_instance_data.copy() for _ in range(5)]

    with open(pkl_file, 'wb') as f:
        pickle.dump(instances, f)

    return str(pkl_file)


@pytest.fixture
def sample_dataset_dir(temp_dir):
    """Create a sample dataset directory with multiple pickle files."""
    data_dir = temp_dir / "dataset"
    data_dir.mkdir()

    # Create multiple pickle files
    for i in range(3):
        pkl_file = data_dir / f"patient_{i:03d}.pkl"
        instances = [{
            'features': np.random.randn(10, 512).astype(np.float32),
            'label': i % 2,  # Alternating labels
            'patient_id': f'patient_{i:03d}',
            'center': f'center_{chr(65 + i % 3)}',  # A, B, C
            'OS_6': i % 2,
            'OS_12': (i + 1) % 2,
            'OS_24': i % 2,
        } for _ in range(5)]

        with open(pkl_file, 'wb') as f:
            pickle.dump(instances, f)

    return str(data_dir)


@pytest.fixture
def device():
    """Get the device for testing (CPU by default)."""
    return torch.device('cpu')


@pytest.fixture
def sample_model():
    """Create a simple model for testing."""
    from conv_mil_model import ConvMILModel

    model = ConvMILModel(
        feature_dim=512,
        hidden_dim=256,
        num_classes=2,
        dropout=0.1
    )
    return model


@pytest.fixture
def sample_dataloader():
    """Create a sample dataloader for testing."""
    # This will be implemented when we create actual tests
    pass


@pytest.fixture(autouse=True)
def set_random_seeds():
    """Set random seeds for reproducibility in tests."""
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)


@pytest.fixture
def mock_neptune_run(monkeypatch):
    """Mock Neptune run for testing without actual Neptune connection."""
    class MockNeptuneRun:
        def __init__(self):
            self.data = {}

        def __setitem__(self, key, value):
            self.data[key] = value

        def __getitem__(self, key):
            return self.data.get(key)

        def stop(self):
            pass

    return MockNeptuneRun()


# Skip tests that require GPU if not available
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "requires_gpu: mark test as requiring GPU"
    )
    config.addinivalue_line(
        "markers", "requires_data: mark test as requiring actual data files"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to skip tests based on markers."""
    skip_gpu = pytest.mark.skip(reason="GPU not available")

    for item in items:
        if "requires_gpu" in item.keywords and not torch.cuda.is_available():
            item.add_marker(skip_gpu)
