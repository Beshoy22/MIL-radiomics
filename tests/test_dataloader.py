"""
Tests for the dataloader module.
"""

import pytest
import numpy as np
import torch
import pickle
from pathlib import Path
from dataloader import (
    compute_data_dir_checksum,
    create_weighted_sampler,
    CachedDataset
)


class TestComputeDataDirChecksum:
    """Tests for compute_data_dir_checksum function."""

    def test_checksum_empty_directory(self, temp_dir):
        """Test checksum of empty directory."""
        checksum = compute_data_dir_checksum(str(temp_dir))

        # Should return a valid SHA-256 hash (64 characters)
        assert len(checksum) == 64
        assert all(c in '0123456789abcdef' for c in checksum)

    def test_checksum_with_pkl_files(self, sample_dataset_dir):
        """Test checksum with pickle files."""
        checksum = compute_data_dir_checksum(sample_dataset_dir)

        # Should return a valid SHA-256 hash
        assert len(checksum) == 64
        assert isinstance(checksum, str)

    def test_checksum_changes_with_files(self, temp_dir):
        """Test that checksum changes when files change."""
        # Get initial checksum
        checksum1 = compute_data_dir_checksum(str(temp_dir))

        # Add a file
        pkl_file = temp_dir / "new_file.pkl"
        with open(pkl_file, 'wb') as f:
            pickle.dump({'data': 'test'}, f)

        # Get new checksum
        checksum2 = compute_data_dir_checksum(str(temp_dir))

        # Checksums should be different
        assert checksum1 != checksum2

    def test_checksum_consistent(self, sample_dataset_dir):
        """Test that checksum is consistent for same directory."""
        checksum1 = compute_data_dir_checksum(sample_dataset_dir)
        checksum2 = compute_data_dir_checksum(sample_dataset_dir)

        assert checksum1 == checksum2


class TestCreateWeightedSampler:
    """Tests for create_weighted_sampler function."""

    def test_balanced_dataset(self):
        """Test sampler creation for balanced dataset."""
        labels = [0, 1, 0, 1, 0, 1]  # Balanced

        sampler, class_weights = create_weighted_sampler(labels, oversample_factor=1.0)

        # Check class weights
        assert len(class_weights) == 2
        assert class_weights[0] > 0
        assert class_weights[1] > 0

    def test_imbalanced_dataset(self):
        """Test sampler creation for imbalanced dataset."""
        labels = [0, 0, 0, 0, 0, 1]  # Imbalanced: 5:1

        sampler, class_weights = create_weighted_sampler(labels, oversample_factor=2.0)

        # Minority class should have higher weight
        assert class_weights[1] > class_weights[0]

    def test_no_oversampling(self):
        """Test sampler with no oversampling."""
        labels = [0, 0, 1, 1]

        sampler, class_weights = create_weighted_sampler(labels, oversample_factor=0)

        # Should still return valid weights
        assert len(class_weights) == 2
        assert all(w > 0 for w in class_weights.values())

    def test_single_class(self):
        """Test sampler with single class."""
        labels = [0, 0, 0, 0]

        sampler, class_weights = create_weighted_sampler(labels, oversample_factor=1.0)

        # Should handle single class gracefully
        assert len(class_weights) == 1
        assert 0 in class_weights


class TestCachedDataset:
    """Tests for CachedDataset class."""

    @pytest.fixture
    def sample_instances(self):
        """Create sample instances for testing."""
        instances = []
        for i in range(5):
            instances.append({
                'features': torch.randn(10, 512),
                'label': i % 2,
                'patient_id': f'patient_{i:03d}',
                'center': f'center_{chr(65 + i % 2)}'
            })
        return instances

    def test_dataset_creation(self, sample_instances):
        """Test basic dataset creation."""
        dataset = CachedDataset(
            instances=sample_instances,
            max_patches=20,
            feature_dim=512
        )

        assert len(dataset) == 5
        assert dataset.max_patches == 20
        assert dataset.feature_dim == 512

    def test_dataset_getitem(self, sample_instances):
        """Test getting items from dataset."""
        dataset = CachedDataset(
            instances=sample_instances,
            max_patches=20,
            feature_dim=512
        )

        # Get first item
        features, label, identifiers = dataset[0]

        # Check types and shapes
        assert isinstance(features, torch.Tensor)
        assert isinstance(label, torch.Tensor)
        assert isinstance(identifiers, dict)

        assert features.shape == (20, 512)  # Padded to max_patches
        assert label.shape == ()
        assert 'patient_id' in identifiers
        assert 'center' in identifiers

    def test_dataset_label_extraction(self, sample_instances):
        """Test correct label extraction."""
        dataset = CachedDataset(
            instances=sample_instances,
            max_patches=20,
            feature_dim=512
        )

        # Check labels for all instances
        for i in range(len(dataset)):
            _, label, _ = dataset[i]
            expected_label = i % 2
            assert label.item() == expected_label

    def test_dataset_with_padding(self):
        """Test dataset with features that need padding."""
        instances = [{
            'features': torch.randn(5, 512),  # Less than max_patches
            'label': 1,
            'patient_id': 'test_001',
            'center': 'center_A'
        }]

        dataset = CachedDataset(
            instances=instances,
            max_patches=20,
            feature_dim=512
        )

        features, _, _ = dataset[0]

        # Should be padded to max_patches
        assert features.shape == (20, 512)

        # First 5 patches should have non-zero values, rest should be padding
        assert torch.any(features[:5] != 0)

    def test_dataset_with_truncation(self):
        """Test dataset with features that need truncation."""
        instances = [{
            'features': torch.randn(30, 512),  # More than max_patches
            'label': 0,
            'patient_id': 'test_002',
            'center': 'center_B'
        }]

        dataset = CachedDataset(
            instances=instances,
            max_patches=20,
            feature_dim=512
        )

        features, _, _ = dataset[0]

        # Should be truncated to max_patches
        assert features.shape == (20, 512)


class TestDataloaderIntegration:
    """Integration tests for dataloader functionality."""

    def test_full_pipeline_with_sample_data(self, sample_dataset_dir):
        """Test full data loading pipeline."""
        # This is a placeholder for a more complex integration test
        # that would test the full prepare_dataloaders function

        # For now, just verify the directory is set up correctly
        from pathlib import Path
        data_dir = Path(sample_dataset_dir)

        pkl_files = list(data_dir.glob('*.pkl'))
        assert len(pkl_files) == 3

        # Verify files can be opened
        for pkl_file in pkl_files:
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
                assert isinstance(data, list)
                assert len(data) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
