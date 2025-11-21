import os
import pickle
import hashlib
import json
import time
from collections import Counter
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split

from logger import get_logger
from secure_pickle import safe_pickle_load, safe_pickle_dump

logger = get_logger(__name__)


def compute_data_dir_checksum(data_dir):
    """
    Compute a checksum of the data directory to detect changes.
    
    Args:
        data_dir (str): Directory containing .pkl files
        
    Returns:
        str: Checksum string representing the directory state
    """
    pkl_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.pkl')])
    file_info = []
    
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


def process_pkl_file(pkl_file, endpoint, feature_dim=512):
    """
    Process a single .pkl file and extract instances with the specified endpoint.
    Handles feature structures where features can be a list of tensors or multi-dimensional.
    
    Args:
        pkl_file (str): Path to the .pkl file
        endpoint (str): Which endpoint to use
        feature_dim (int): Expected dimension of features (default: 512)
        
    Returns:
        tuple: (max_patches, instances)
    """
    try:
        instances = []
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
                
                # Handle different feature formats
                processed_features = None
                
                if isinstance(features, list):
                    # List of patch embeddings
                    patch_count = len(features)
                    
                    # Process the patches to create a tensor of shape [n_patches, feature_dim]
                    if patch_count > 0:
                        # Check the type of the first patch
                        if isinstance(features[0], torch.Tensor):
                            # Handle list of tensors
                            try:
                                # Try direct stacking first
                                processed_features = torch.stack(features)
                            except (RuntimeError, ValueError, TypeError) as e:
                                # If tensors have different shapes, normalize them
                                logger.debug(f"Direct tensor stacking failed in {pkl_file}: {e}. Normalizing features.")
                                norm_features = []
                                for feature in features:
                                    # Handle multi-dimensional tensors
                                    if feature.dim() > 1:
                                        # Try to reshape to feature_dim
                                        try:
                                            flattened = feature.reshape(-1)
                                            # Take first feature_dim elements or pad
                                            if flattened.size(0) >= feature_dim:
                                                norm_features.append(flattened[:feature_dim])
                                            else:
                                                padded = torch.zeros(feature_dim, dtype=torch.float32)
                                                padded[:flattened.size(0)] = flattened
                                                norm_features.append(padded)
                                        except (RuntimeError, ValueError) as e:
                                            # Skip patches we can't process
                                            logger.warning(f"Skipping patch that couldn't be reshaped: {e}")
                                            continue
                                    elif feature.dim() == 1:
                                        # For 1D vectors, ensure they're feature_dim
                                        if feature.shape[0] >= feature_dim:
                                            norm_features.append(feature[:feature_dim])
                                        else:
                                            padded = torch.zeros(feature_dim, dtype=torch.float32)
                                            padded[:feature.shape[0]] = feature
                                            norm_features.append(padded)
                                    else:
                                        # Skip empty tensors
                                        continue
                                
                                if norm_features:
                                    processed_features = torch.stack(norm_features)
                                else:
                                    # Skip this instance if no valid patches
                                    continue
                                    
                        elif isinstance(features[0], np.ndarray):
                            # Handle list of numpy arrays
                            try:
                                # Try direct stacking first
                                processed_features = torch.tensor(np.stack(features), dtype=torch.float32)
                            except (ValueError, RuntimeError, TypeError) as e:
                                # If arrays have different shapes, normalize them
                                logger.debug(f"Direct numpy stacking failed in {pkl_file}: {e}. Normalizing features.")
                                norm_features = []
                                for feature in features:
                                    # Handle multi-dimensional arrays
                                    if feature.ndim > 1:
                                        # Try to reshape to feature_dim
                                        try:
                                            flattened = feature.reshape(-1)
                                            # Take first feature_dim elements or pad
                                            if flattened.shape[0] >= feature_dim:
                                                norm_features.append(flattened[:feature_dim])
                                            else:
                                                padded = np.zeros(feature_dim, dtype=np.float32)
                                                padded[:flattened.shape[0]] = flattened
                                                norm_features.append(padded)
                                        except (ValueError, RuntimeError) as e:
                                            # Skip patches we can't process
                                            logger.warning(f"Skipping numpy patch that couldn't be reshaped: {e}")
                                            continue
                                    elif feature.ndim == 1:
                                        # For 1D vectors, ensure they're feature_dim
                                        if feature.shape[0] >= feature_dim:
                                            norm_features.append(feature[:feature_dim])
                                        else:
                                            padded = np.zeros(feature_dim, dtype=np.float32)
                                            padded[:feature.shape[0]] = feature
                                            norm_features.append(padded)
                                    else:
                                        # Skip empty arrays
                                        continue
                                
                                if norm_features:
                                    processed_features = torch.tensor(np.stack(norm_features), dtype=torch.float32)
                                else:
                                    # Skip this instance if no valid patches
                                    continue
                        else:
                            # Handle other types (e.g., lists of lists)
                            try:
                                raw_array = np.array(features)
                                # If it's already structured as [n_patches, feature_dim]
                                if raw_array.ndim == 2 and raw_array.shape[1] == feature_dim:
                                    processed_features = torch.tensor(raw_array, dtype=torch.float32)
                                else:
                                    # Try to reshape assuming each sub-list is a flat vector
                                    processed_features = torch.tensor(np.array(features), dtype=torch.float32)
                                    # Ensure correct shape
                                    if processed_features.dim() > 1:
                                        # If multi-dimensional, flatten to patches
                                        processed_features = processed_features.reshape(-1, processed_features.shape[-1])
                                        # If feature dim doesn't match, try to adapt
                                        if processed_features.shape[1] != feature_dim:
                                            # Transpose if possible
                                            if processed_features.shape[1] > feature_dim and processed_features.shape[0] == feature_dim:
                                                processed_features = processed_features.transpose(0, 1)
                                    elif processed_features.dim() == 1:
                                        # Single vector, make it a single patch
                                        processed_features = processed_features.reshape(1, -1)
                            except (ValueError, RuntimeError, TypeError) as e:
                                logger.warning(f"Could not process features in {pkl_file}: {e}")
                                continue
                    else:
                        # Skip instances with no patches
                        continue
                        
                elif isinstance(features, torch.Tensor):
                    # Handle tensor directly
                    
                    # Handle multi-dimensional tensors
                    if features.dim() > 2:
                        # Collapse all dimensions except the last one (assuming it's the feature dimension)
                        if features.shape[-1] == feature_dim:
                            # Last dimension is feature_dim, reshape to [n_patches, feature_dim]
                            processed_features = features.reshape(-1, feature_dim)
                        else:
                            # Try to identify the feature dimension
                            feature_dim_idx = None
                            for i, dim_size in enumerate(features.shape):
                                if dim_size == feature_dim:
                                    feature_dim_idx = i
                                    break
                            
                            if feature_dim_idx is not None:
                                # Found feature dimension, reshape to move it to the end
                                # Create permutation to move feature dimension to the end
                                permutation = list(range(features.dim()))
                                permutation.remove(feature_dim_idx)
                                permutation.append(feature_dim_idx)
                                # Permute and reshape
                                permuted = features.permute(*permutation)
                                processed_features = permuted.reshape(-1, feature_dim)
                            else:
                                # No feature dimension found, reshape all dimensions
                                # Assuming the last dimension is features
                                processed_features = features.reshape(-1, features.shape[-1])
                                # If still doesn't match feature_dim, we might need to transpose
                                if processed_features.shape[1] != feature_dim and processed_features.shape[0] == feature_dim:
                                    processed_features = processed_features.transpose(0, 1)
                    
                    elif features.dim() == 2:
                        # Already 2D, check if dimensions need transposing
                        if features.shape[0] == feature_dim and features.shape[1] != feature_dim:
                            # If in [feature_dim, n_patches] format, transpose
                            processed_features = features.transpose(0, 1)
                        else:
                            # Already in [n_patches, feature_dim] format or other 2D shape
                            processed_features = features
                            
                    elif features.dim() == 1:
                        # 1D tensor, treat as a single patch
                        processed_features = features.unsqueeze(0)
                    
                elif isinstance(features, np.ndarray):
                    # Handle numpy array directly
                    
                    # Handle multi-dimensional arrays
                    if features.ndim > 2:
                        # Collapse all dimensions except the last one (assuming it's the feature dimension)
                        if features.shape[-1] == feature_dim:
                            # Last dimension is feature_dim, reshape to [n_patches, feature_dim]
                            processed_features = torch.tensor(features.reshape(-1, feature_dim), dtype=torch.float32)
                        else:
                            # Try to identify the feature dimension
                            feature_dim_idx = None
                            for i, dim_size in enumerate(features.shape):
                                if dim_size == feature_dim:
                                    feature_dim_idx = i
                                    break
                            
                            if feature_dim_idx is not None:
                                # Found feature dimension, reshape to move it to the end
                                # Create permutation to move feature dimension to the end
                                permutation = list(range(features.ndim))
                                permutation.remove(feature_dim_idx)
                                permutation.append(feature_dim_idx)
                                # Permute and reshape
                                permuted = np.transpose(features, permutation)
                                processed_features = torch.tensor(permuted.reshape(-1, feature_dim), dtype=torch.float32)
                            else:
                                # No feature dimension found, reshape all dimensions
                                # Assuming the last dimension is features
                                reshaped = features.reshape(-1, features.shape[-1])
                                processed_features = torch.tensor(reshaped, dtype=torch.float32)
                                # If still doesn't match feature_dim, we might need to transpose
                                if processed_features.shape[1] != feature_dim and processed_features.shape[0] == feature_dim:
                                    processed_features = processed_features.transpose(0, 1)
                    
                    elif features.ndim == 2:
                        # Already 2D, check if dimensions need transposing
                        if features.shape[0] == feature_dim and features.shape[1] != feature_dim:
                            # If in [feature_dim, n_patches] format, transpose
                            processed_features = torch.tensor(features.transpose(), dtype=torch.float32)
                        else:
                            # Already in [n_patches, feature_dim] format or other 2D shape
                            processed_features = torch.tensor(features, dtype=torch.float32)
                            
                    elif features.ndim == 1:
                        # 1D array, treat as a single patch
                        processed_features = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
                
                else:
                    # Skip instances with unsupported feature types
                    logger.warning(f"Unsupported feature type {type(features)} in {pkl_file}")
                    continue
                
                # Final check on processed features to ensure it has the right dimensions
                if processed_features is None:
                    # Skip if processing failed
                    continue
                
                # Convert to float for consistency
                processed_features = processed_features.float()
                
                # Make final adjustment to ensure [n_patches, feature_dim] format
                if processed_features.dim() == 2:
                    if processed_features.shape[1] != feature_dim and processed_features.shape[0] == feature_dim:
                        # If in [feature_dim, n_patches] format, transpose
                        processed_features = processed_features.transpose(0, 1)
                    
                    # If still doesn't match expected feature dimension after all our efforts,
                    # we need to either skip or adapt
                    if processed_features.shape[1] != feature_dim:
                        # Check if we can pad/truncate
                        if processed_features.shape[1] < feature_dim:
                            # Pad with zeros
                            padded = torch.zeros(processed_features.shape[0], feature_dim, dtype=torch.float32)
                            padded[:, :processed_features.shape[1]] = processed_features
                            processed_features = padded
                        else:
                            # Truncate to feature_dim
                            processed_features = processed_features[:, :feature_dim]
                
                elif processed_features.dim() == 1:
                    # If 1D, reshape to a single patch
                    if processed_features.shape[0] < feature_dim:
                        # Pad with zeros
                        padded = torch.zeros(feature_dim, dtype=torch.float32)
                        padded[:processed_features.shape[0]] = processed_features
                        processed_features = padded.unsqueeze(0)
                    else:
                        # Truncate and reshape
                        processed_features = processed_features[:feature_dim].unsqueeze(0)
                        
                elif processed_features.dim() > 2:
                    # If still multi-dimensional after all processing, reshape directly
                    processed_features = processed_features.reshape(-1, feature_dim)
                    
                # At this point, processed_features should be in [n_patches, feature_dim] format
                
                # Update max_patches
                n_patches = processed_features.shape[0]
                max_patches = max(max_patches, n_patches)
                
                # Get label
                label = instance[endpoint]  # Get the specified endpoint
                
                # Convert to binary if not already
                if isinstance(label, bool):
                    label = 1 if label else 0
                elif not isinstance(label, int):
                    # Try to convert to int if it's a different type
                    try:
                        label = int(label)
                    except (ValueError, TypeError):
                        # If conversion fails, treat as binary
                        label = 1 if label else 0
                
                # Ensure label is either 0 or 1
                label = 1 if label > 0 else 0
                
                # Create a patient ID from center and index
                # Try to get an existing ID from instance, if available
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


class CachedDataset(Dataset):
    """Dataset that works with pre-processed and cached data"""
    
    def __init__(self, data, transform=None, max_patches=300, feature_dim=512):
        """
        Args:
            data (list): List of pre-processed instances
            transform (callable, optional): Optional transform to be applied on features
            max_patches (int): Maximum number of patches for padding
            feature_dim (int): Expected dimension of features
        """
        self.data = data
        self.transform = transform
        self.max_patches = max_patches
        self.feature_dim = feature_dim
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        try:
            item = self.data[idx]
            features = item['features']
            
            if self.transform:
                features = self.transform(features)
            
            # Features should already be tensors from process_pkl_file function
            # But handle any edge cases
            if not isinstance(features, torch.Tensor):
                try:
                    if isinstance(features, list) and features:
                        # Try to stack if it's a list of tensors
                        if all(isinstance(f, torch.Tensor) for f in features):
                            features = torch.stack(features)
                        else:
                            # Convert to numpy first if not all tensors
                            features = torch.tensor(np.array(features), dtype=torch.float32)
                    else:
                        features = torch.tensor(features, dtype=torch.float32)
                except Exception as e:
                    logger.error(f"Error converting features to tensor at idx {idx}: {e}", exc_info=True)
                    # Provide a dummy tensor as fallback
                    features = torch.zeros((self.max_patches, self.feature_dim), dtype=torch.float32)
            
            # Ensure correct data type
            features = features.float()
            
            # Handle multi-dimensional tensors - collapse to [n_patches, feature_dim]
            if features.dim() > 2:
                # Check if last dimension is feature_dim
                if features.shape[-1] == self.feature_dim:
                    # Reshape to [n_patches, feature_dim]
                    features = features.reshape(-1, self.feature_dim)
                else:
                    # Try to identify the feature dimension
                    feature_dim_idx = None
                    for i, dim_size in enumerate(features.shape):
                        if dim_size == self.feature_dim:
                            feature_dim_idx = i
                            break
                    
                    if feature_dim_idx is not None:
                        # Found feature dimension, reshape to move it to the end
                        # Create permutation to move feature dimension to the end
                        permutation = list(range(features.dim()))
                        permutation.remove(feature_dim_idx)
                        permutation.append(feature_dim_idx)
                        # Permute and reshape
                        permuted = features.permute(*permutation)
                        features = permuted.reshape(-1, self.feature_dim)
                    else:
                        # No feature dimension found, reshape assuming last dim is features
                        features = features.reshape(-1, features.shape[-1])
                        # If feature dimension doesn't match expected, adapt
                        if features.shape[1] != self.feature_dim:
                            if features.shape[1] < self.feature_dim:
                                # Pad with zeros
                                padded = torch.zeros(features.shape[0], self.feature_dim, dtype=torch.float32)
                                padded[:, :features.shape[1]] = features
                                features = padded
                            else:
                                # Truncate
                                features = features[:, :self.feature_dim]
            
            # Ensure [n_patches, feature_dim] format
            if features.dim() == 2:
                if features.shape[1] != self.feature_dim and features.shape[0] == self.feature_dim:
                    # If in [feature_dim, n_patches] format, transpose
                    features = features.transpose(0, 1)
                
                # Final check to ensure correct feature dimension
                if features.shape[1] != self.feature_dim:
                    if features.shape[1] < self.feature_dim:
                        # Pad with zeros
                        padded = torch.zeros(features.shape[0], self.feature_dim, dtype=torch.float32)
                        padded[:, :features.shape[1]] = features
                        features = padded
                    else:
                        # Truncate
                        features = features[:, :self.feature_dim]
            
            # Handle case where features might be 1D
            elif features.dim() == 1:
                if features.shape[0] == self.feature_dim:
                    # If it's a single feature_dim vector, make it a single patch
                    features = features.unsqueeze(0)
                elif features.shape[0] < self.feature_dim:
                    # Pad with zeros
                    padded = torch.zeros(self.feature_dim, dtype=torch.float32)
                    padded[:features.shape[0]] = features
                    features = padded.unsqueeze(0)
                else:
                    # Reshape to multiple patches if larger than feature_dim
                    features = features.reshape(-1, self.feature_dim)
            
            # Pad or truncate the features to max_patches
            n_patches = features.shape[0]
            if n_patches < self.max_patches:
                # Pad with zeros if fewer patches than max_patches
                padding = torch.zeros(self.max_patches - n_patches, self.feature_dim, 
                                    dtype=features.dtype, device=features.device)
                features = torch.cat([features, padding], dim=0)
            elif n_patches > self.max_patches:
                # Truncate if more patches than max_patches
                features = features[:self.max_patches]
            
            label = torch.tensor(item['label'], dtype=torch.long)
            
            # Get patient identifiers
            patient_id = item.get('patient_id', f"unknown_{idx}")
            center = item.get('center', 'unknown')
            
            # Return a tuple with features, label, and identifiers
            identifiers = {'patient_id': patient_id, 'center': center}
            
            return features, label, identifiers
        except Exception as e:
            logger.error(f"Error in __getitem__ at idx {idx}: {e}", exc_info=True)
            # Return a dummy sample in case of error
            dummy_features = torch.zeros((self.max_patches, self.feature_dim), dtype=torch.float32)
            dummy_label = torch.tensor(0, dtype=torch.long)
            dummy_identifiers = {'patient_id': f"error_{idx}", 'center': 'error'}
            return dummy_features, dummy_label, dummy_identifiers


def create_weighted_sampler(labels, oversample_factor=1.0):
    """
    Create a weighted random sampler for oversampling the minority class.
    
    Args:
        labels (list or array): Class labels
        oversample_factor (float): Factor to multiply minority class weight
                                 (1.0 means balanced, >1.0 means more minority samples)
                                 (0.0 means no oversampling - use uniform sampling)
    
    Returns:
        WeightedRandomSampler or None: Sampler for DataLoader, None if no oversampling
    """
    # If oversample_factor is 0, return None to indicate no oversampling
    if oversample_factor == 0:
        return None
        
    # Count instances per class
    label_counts = Counter(labels)
    
    # Calculate weights per class (inversely proportional to class frequency)
    n_samples = len(labels)
    class_weights = {cls: n_samples / count for cls, count in label_counts.items()}
    
    # Apply oversample factor to minority class
    if 0 in class_weights and 1 in class_weights:
        minority_class = 0 if label_counts[0] < label_counts[1] else 1
        class_weights[minority_class] *= oversample_factor
    
    # Assign weights to each sample
    weights = [class_weights[label] for label in labels]
    
    # Create sampler
    sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
    
    return sampler


def collate_fn(batch):
    """
    Collate function that handles the new structure that includes patient identifiers.
    
    Args:
        batch (list): List of (features, label, identifiers) tuples
        
    Returns:
        torch.Tensor: Batched features
        torch.Tensor: Batched labels
        dict: Dictionary of identifiers lists
    """
    try:
        features = []
        labels = []
        patient_ids = []
        centers = []
        
        for i, item in enumerate(batch):
            try:
                # Unpack the tuple
                if len(item) == 3:
                    feature, label, identifier = item
                    features.append(feature)
                    labels.append(label)
                    patient_ids.append(identifier.get('patient_id', f"unknown_{i}"))
                    centers.append(identifier.get('center', 'unknown'))
                else:
                    # Backward compatibility with old format
                    feature, label = item
                    features.append(feature)
                    labels.append(label)
                    patient_ids.append(f"unknown_{i}")
                    centers.append('unknown')
            except Exception as e:
                print(f"Error processing batch item {i}: {e}")
                # Skip problematic items
                continue
        
        if not features:
            # Return a dummy batch if all items were problematic
            feature_dim = getattr(batch[0][0], 'shape', [0, 512])[1] if batch else 512
            dummy_features = torch.zeros((1, 300, feature_dim), dtype=torch.float32)  # Using default max_patches=300
            dummy_labels = torch.zeros(1, dtype=torch.long)
            dummy_identifiers = {'patient_id': ['error'], 'center': ['error']}
            return dummy_features, dummy_labels, dummy_identifiers
        
        # All tensors should be the same size now, so we can simply stack them
        features_tensor = torch.stack(features)
        labels_tensor = torch.stack(labels)
        
        # Combine identifiers
        identifiers = {
            'patient_id': patient_ids,
            'center': centers
        }
        
        return features_tensor, labels_tensor, identifiers
    except Exception as e:
        print(f"Error in collate_fn: {e}")
        # Return a dummy batch in case of error
        feature_dim = getattr(batch[0][0], 'shape', [0, 512])[1] if batch else 512
        dummy_features = torch.zeros((1, 300, feature_dim), dtype=torch.float32)  # Using default max_patches=300
        dummy_labels = torch.zeros(1, dtype=torch.long)
        dummy_identifiers = {'patient_id': ['error'], 'center': ['error']}
        return dummy_features, dummy_labels, dummy_identifiers


def prepare_dataloaders(data_dir, endpoint='OS_6', batch_size=16, oversample_factor=1.0, 
                        val_size=0.15, test_size=0.15, num_workers=4, seed=42,
                        use_cache=True, cache_dir=None, splitted=False, feature_dim=512):
    """
    Prepare DataLoaders for training, validation, and testing with improved caching.
    
    Args:
        data_dir (str): Directory containing .pkl files
        endpoint (str): Which endpoint to use ('OS_6' or 'OS_24')
        batch_size (int): Batch size
        oversample_factor (float): Factor for oversampling minority class (0 to disable)
        val_size (float): Proportion of data for validation
        test_size (float): Proportion of data for testing
        num_workers (int): Number of workers for data loading
        seed (int): Random seed
        use_cache (bool): Whether to cache data in memory
        cache_dir (str, optional): Directory to cache processed data
        splitted (bool): Whether to use pre-split data files
        feature_dim (int): Expected dimension of features
    
    Returns:
        tuple: (train_loader, val_loader, test_loader, class_weights, split_metrics, max_patches)
    """
    # Create cache directory if specified and doesn't exist
    if cache_dir and not os.path.exists(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)
    
    # Handle pre-split data case
    if splitted:
        splits, metrics, max_patches, class_weights = load_presplit_data(
            data_dir=data_dir,
            endpoint=endpoint,
            cache_dir=cache_dir,
            feature_dim=feature_dim
        )
    else:
        # Compute directory checksum to detect changes
        dir_checksum = compute_data_dir_checksum(data_dir)
        
        # Cached split filename
        cached_split_file = os.path.join(cache_dir, f"splits_{endpoint}_{val_size}_{test_size}_{seed}_{feature_dim}_{dir_checksum}.pkl") if cache_dir else None
        
        # Check if cached splits exist
        if cached_split_file and os.path.exists(cached_split_file) and use_cache:
            print(f"Loading cached splits from {cached_split_file}")
            with open(cached_split_file, 'rb') as f:
                cached_data = pickle.load(f)
                splits = cached_data['splits']
                max_patches = cached_data['max_patches']
                class_weights = cached_data['class_weights']
                metrics = cached_data['metrics']
        else:
            # Process data and create splits
            print(f"Processing data and creating splits (no cached splits found or cache not used)")
            splits, metrics, max_patches, class_weights = create_cached_splits(
                data_dir=data_dir,
                endpoint=endpoint,
                val_size=val_size,
                test_size=test_size,
                seed=seed,
                cache_dir=cache_dir,
                feature_dim=feature_dim
            )
            
            # Cache the splits
            if cached_split_file:
                with open(cached_split_file, 'wb') as f:
                    pickle.dump({
                        'splits': splits,
                        'max_patches': max_patches,
                        'class_weights': class_weights,
                        'metrics': metrics
                    }, f)
    
    # Create datasets using the splits
    train_dataset = CachedDataset(splits['train_data'], max_patches=max_patches, feature_dim=feature_dim)
    val_dataset = CachedDataset(splits['val_data'], max_patches=max_patches, feature_dim=feature_dim)
    test_dataset = CachedDataset(splits['test_data'], max_patches=max_patches, feature_dim=feature_dim)
    
    # Create weighted sampler for handling class imbalance if oversampling is enabled
    train_sampler = None
    if oversample_factor > 0:
        train_sampler = create_weighted_sampler(
            labels=[item['label'] for item in splits['train_data']],
            oversample_factor=oversample_factor
        )
        shuffle = False  # Don't shuffle when using sampler
    else:
        shuffle = True  # Shuffle when not using sampler
    
    # Create data loaders with our collate_fn
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=shuffle if train_sampler is None else False,  # Only shuffle if not using sampler
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return (
        train_loader, 
        val_loader, 
        test_loader, 
        class_weights, 
        metrics, 
        max_patches
    )


def create_cached_splits(data_dir, endpoint='OS_6', val_size=0.15, test_size=0.15, seed=42, cache_dir=None, feature_dim=512):
    """
    Process data files individually and create stratified splits with caching.
    
    Args:
        data_dir (str): Directory containing .pkl files
        endpoint (str): Which endpoint to use ('OS_6' or 'OS_24')
        val_size (float): Proportion of data for validation
        test_size (float): Proportion of data for testing
        seed (int): Random seed
        cache_dir (str, optional): Directory to cache processed data
        feature_dim (int): Expected dimension of features
        
    Returns:
        tuple: (splits, metrics, max_patches, class_weights)
    """
    # Get all .pkl files in the directory
    pkl_files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.pkl')])
    print(f"Found {len(pkl_files)} .pkl files in {data_dir}")
    
    # Initialize splits and counters
    train_data = []
    val_data = []
    test_data = []
    
    train_count = {'total': 0, 0: 0, 1: 0}
    val_count = {'total': 0, 0: 0, 1: 0}
    test_count = {'total': 0, 0: 0, 1: 0}
    
    max_patches = 0
    
    # Process each file individually
    for pkl_file in tqdm(pkl_files, desc="Processing files for splits"):
        # Create a file-specific cache if cache_dir is provided
        file_cache_path = None
        if cache_dir:
            # Using SHA-256 for secure file identification
            file_hash = hashlib.sha256(os.path.basename(pkl_file).encode()).hexdigest()
            file_cache_path = os.path.join(cache_dir, f"{file_hash}_{endpoint}_{feature_dim}.pkl")
            
            # Check if cached file exists
            if os.path.exists(file_cache_path):
                with open(file_cache_path, 'rb') as f:
                    file_data = pickle.load(f)
                    file_max_patches = file_data['max_patches']
                    instances = file_data['instances']
            else:
                # Process the file
                file_max_patches, instances = process_pkl_file(pkl_file, endpoint, feature_dim=feature_dim)
                
                # Cache the processed data
                with open(file_cache_path, 'wb') as f:
                    pickle.dump({
                        'max_patches': file_max_patches,
                        'instances': instances
                    }, f)
        else:
            # Process the file without caching
            file_max_patches, instances = process_pkl_file(pkl_file, endpoint, feature_dim=feature_dim)
        
        # Update max_patches
        max_patches = max(max_patches, file_max_patches)
        
        # Skip if no valid instances were found
        if not instances:
            continue
        
        # Extract labels for stratification
        labels = [instance['label'] for instance in instances]
        
        # Check if stratification is possible (need at least 2 samples per class)
        label_counts = Counter(labels)
        can_stratify = all(count >= 2 for count in label_counts.values())
        
        # Perform split with or without stratification
        if can_stratify:
            # Stratified split for this file
            train_idx, temp_idx = train_test_split(
                range(len(instances)),
                test_size=val_size + test_size,
                random_state=seed,
                stratify=labels
            )
            
            # Check if the second stratification is possible
            temp_labels = [labels[i] for i in temp_idx]
            temp_label_counts = Counter(temp_labels)
            can_stratify_temp = all(count >= 2 for count in temp_label_counts.values())
            
            # Adjust validation size relative to remaining data
            val_test_ratio = val_size / (val_size + test_size)
            
            if can_stratify_temp:
                val_idx, test_idx = train_test_split(
                    temp_idx,
                    test_size=1 - val_test_ratio,
                    random_state=seed,
                    stratify=temp_labels
                )
            else:
                # Fall back to non-stratified split for the second stage
                val_idx, test_idx = train_test_split(
                    temp_idx,
                    test_size=1 - val_test_ratio,
                    random_state=seed
                )
        else:
            # Fall back to non-stratified split
            print(f"Warning: File {os.path.basename(pkl_file)} has insufficient samples per class for stratification (counts: {dict(label_counts)}). Using non-stratified split.")
            train_idx, temp_idx = train_test_split(
                range(len(instances)),
                test_size=val_size + test_size,
                random_state=seed
            )
            
            # Adjust validation size relative to remaining data
            val_test_ratio = val_size / (val_size + test_size)
            val_idx, test_idx = train_test_split(
                temp_idx,
                test_size=1 - val_test_ratio,
                random_state=seed
            )
        
        # Add instances to respective splits
        for idx in train_idx:
            train_data.append(instances[idx])
            train_count['total'] += 1
            train_count[instances[idx]['label']] += 1
        
        for idx in val_idx:
            val_data.append(instances[idx])
            val_count['total'] += 1
            val_count[instances[idx]['label']] += 1
        
        for idx in test_idx:
            test_data.append(instances[idx])
            test_count['total'] += 1
            test_count[instances[idx]['label']] += 1
    
    # Calculate class weights for loss function
    class_counts = {0: train_count[0], 1: train_count[1]}
    total_samples = train_count['total']
    num_classes = len(class_counts)
    class_weights = torch.tensor(
        [total_samples / (num_classes * count) for label, count in sorted(class_counts.items())],
        dtype=torch.float32
    )
    
    # Prepare metrics
    metrics = {
        'train_count': train_count['total'],
        'val_count': val_count['total'],
        'test_count': test_count['total'],
        'train_label_counts': {0: train_count[0], 1: train_count[1]},
        'val_label_counts': {0: val_count[0], 1: val_count[1]},
        'test_label_counts': {0: test_count[0], 1: test_count[1]}
    }
    
    # Print split statistics
    print(f"Split statistics:")
    print(f"  Train: {metrics['train_count']} samples, {metrics['train_label_counts']}")
    print(f"  Validation: {metrics['val_count']} samples, {metrics['val_label_counts']}")
    print(f"  Test: {metrics['test_count']} samples, {metrics['test_label_counts']}")
    print(f"  Max patches: {max_patches}")
    
    # Create the final splits dictionary
    splits = {
        'train_data': train_data,
        'val_data': val_data,
        'test_data': test_data
    }
    
    return splits, metrics, max_patches, class_weights


def load_presplit_data(data_dir, endpoint='OS_6', cache_dir=None, feature_dim=512):
    """
    Load pre-split data from train_set.pkl, val_set.pkl, and test_set.pkl.
    
    Args:
        data_dir (str): Directory containing .pkl files
        endpoint (str): Which endpoint to use ('OS_6' or 'OS_24')
        cache_dir (str, optional): Directory to cache processed data
        feature_dim (int): Expected dimension of features
        
    Returns:
        tuple: (splits, metrics, max_patches, class_weights)
    """
    train_file = os.path.join(data_dir, 'train_set.pkl')
    val_file = os.path.join(data_dir, 'val_set.pkl')
    test_file = os.path.join(data_dir, 'test_set.pkl')
    
    # Check if all files exist
    if not (os.path.exists(train_file) and os.path.exists(val_file) and os.path.exists(test_file)):
        raise FileNotFoundError(f"Pre-split files not found in {data_dir}. Need train_set.pkl, val_set.pkl, and test_set.pkl.")
    
    print("WARNING: Using pre-split data files. Cross-validation is not allowed in this mode.")
    print(f"Loading pre-split data from {data_dir}...")
    
    # Process each file
    train_max_patches, train_data = process_pkl_file(train_file, endpoint, feature_dim=feature_dim)
    val_max_patches, val_data = process_pkl_file(val_file, endpoint, feature_dim=feature_dim)
    test_max_patches, test_data = process_pkl_file(test_file, endpoint, feature_dim=feature_dim)
    
    # Calculate maximum patches across all datasets
    max_patches = max(train_max_patches, val_max_patches, test_max_patches)
    
    # Calculate class weights based on training data
    train_labels = [instance['label'] for instance in train_data]
    class_counts = Counter(train_labels)
    total_samples = len(train_labels)
    num_classes = len(class_counts)
    class_weights = torch.tensor(
        [total_samples / (num_classes * count) for label, count in sorted(class_counts.items())],
        dtype=torch.float32
    )
    
    # Calculate counts
    train_count = {'total': len(train_data), 0: class_counts[0], 1: class_counts[1]}
    
    val_labels = [instance['label'] for instance in val_data]
    val_class_counts = Counter(val_labels)
    val_count = {'total': len(val_data), 0: val_class_counts[0], 1: val_class_counts[1]}
    
    test_labels = [instance['label'] for instance in test_data]
    test_class_counts = Counter(test_labels)
    test_count = {'total': len(test_data), 0: test_class_counts[0], 1: test_class_counts[1]}
    
    # Prepare metrics
    metrics = {
        'train_count': train_count['total'],
        'val_count': val_count['total'],
        'test_count': test_count['total'],
        'train_label_counts': {0: train_count[0], 1: train_count[1]},
        'val_label_counts': {0: val_count[0], 1: val_count[1]},
        'test_label_counts': {0: test_count[0], 1: test_count[1]}
    }
    
    # Print split statistics
    print(f"Using pre-split data:")
    print(f"  Train: {metrics['train_count']} samples, {metrics['train_label_counts']}")
    print(f"  Validation: {metrics['val_count']} samples, {metrics['val_label_counts']}")
    print(f"  Test: {metrics['test_count']} samples, {metrics['test_label_counts']}")
    print(f"  Max patches: {max_patches}")
    
    # Create the final splits dictionary
    splits = {
        'train_data': train_data,
        'val_data': val_data,
        'test_data': test_data
    }
    
    return splits, metrics, max_patches, class_weights