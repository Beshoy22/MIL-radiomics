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
    checksum = hashlib.md5(dir_info_str.encode()).hexdigest()
    
    return checksum


def process_pkl_file(pkl_file, endpoint):
    """
    Process a single .pkl file and extract instances with the specified endpoint.
    Handles feature structure where features can be a list of tensors.
    
    Args:
        pkl_file (str): Path to the .pkl file
        endpoint (str): Which endpoint to use
        
    Returns:
        tuple: (max_patches, instances)
    """
    try:
        instances = []
        max_patches = 0
        
        with open(pkl_file, 'rb') as f:
            instances_list = pickle.load(f)
            center = os.path.basename(pkl_file)  # Use filename as center identifier
            
            for instance in instances_list:
                # Skip instances without the specified endpoint
                if endpoint not in instance:
                    continue
                
                # Get features - these can be a list of tensors
                features = instance['features']
                
                # Handle different feature formats
                if isinstance(features, list):
                    # List of patch embeddings
                    patch_count = len(features)
                    
                    # Process the patches to create a tensor of shape [n_patches, feature_dim]
                    if patch_count > 0:
                        # Check the type of the first patch
                        if isinstance(features[0], torch.Tensor):
                            # If the patches are already tensors, stack them
                            try:
                                processed_features = torch.stack(features)
                            except:
                                # If tensors have different shapes, try to convert each to same shape
                                norm_features = []
                                for feature in features:
                                    # Ensure each feature has shape [512] or convert it
                                    if feature.dim() == 1 and feature.shape[0] == 512:
                                        norm_features.append(feature)
                                    elif feature.dim() == 2:
                                        # If it's 2D, take the first dimension if it's 512
                                        if feature.shape[0] == 512:
                                            norm_features.append(feature[0])
                                        elif feature.shape[1] == 512:
                                            norm_features.append(feature[0])
                                        else:
                                            # Skip this patch if we can't handle it
                                            continue
                                    else:
                                        # Skip this patch if we can't handle it
                                        continue
                                
                                if norm_features:
                                    processed_features = torch.stack(norm_features)
                                else:
                                    # Skip this instance if no valid patches
                                    continue
                        elif isinstance(features[0], np.ndarray):
                            # If the patches are numpy arrays, convert to tensors and stack
                            try:
                                processed_features = torch.tensor(np.stack(features), dtype=torch.float32)
                            except:
                                # Handle different shaped arrays
                                norm_features = []
                                for feature in features:
                                    if feature.shape == (512,) or feature.shape == (1, 512):
                                        norm_features.append(feature.reshape(512))
                                    elif feature.shape == (512, 1):
                                        norm_features.append(feature.reshape(512))
                                    else:
                                        # Skip this patch if we can't handle it
                                        continue
                                
                                if norm_features:
                                    processed_features = torch.tensor(np.stack(norm_features), dtype=torch.float32)
                                else:
                                    # Skip this instance if no valid patches
                                    continue
                        else:
                            # Handle other types (e.g., lists of lists)
                            try:
                                # Try to convert to numpy array and then tensor
                                processed_features = torch.tensor(np.array(features), dtype=torch.float32)
                            except:
                                print(f"Warning: Could not process features in {pkl_file}")
                                continue
                    else:
                        # Skip instances with no patches
                        continue
                elif isinstance(features, torch.Tensor):
                    # Already a tensor, just ensure correct shape
                    processed_features = features
                    
                    # Transpose if in [feature_dim, n_patches] format
                    if processed_features.dim() == 2 and processed_features.shape[0] == 512:
                        processed_features = processed_features.transpose(0, 1)  # Now [n_patches, feature_dim]
                elif isinstance(features, np.ndarray):
                    # Convert numpy array to tensor
                    processed_features = torch.tensor(features, dtype=torch.float32)
                    
                    # Transpose if in [feature_dim, n_patches] format
                    if processed_features.dim() == 2 and processed_features.shape[0] == 512:
                        processed_features = processed_features.transpose(0, 1)  # Now [n_patches, feature_dim]
                else:
                    # Skip instances with unsupported feature types
                    print(f"Warning: Unsupported feature type {type(features)} in {pkl_file}")
                    continue
                
                # Update max_patches
                n_patches = processed_features.shape[0]
                max_patches = max(max_patches, n_patches)
                
                # Get label
                label = instance[endpoint]  # Get the specified endpoint
                
                # Convert to binary if not already
                if not isinstance(label, int):
                    label = 1 if label else 0
                
                instances.append({
                    'features': processed_features,
                    'label': label,
                    'center': center
                })
                
        return max_patches, instances
    except Exception as e:
        print(f"Error processing {pkl_file}: {e}")
        return 0, []


class CachedDataset(Dataset):
    """Dataset that works with pre-processed and cached data"""
    
    def __init__(self, data, transform=None, max_patches=300):
        """
        Args:
            data (list): List of pre-processed instances
            transform (callable, optional): Optional transform to be applied on features
            max_patches (int): Maximum number of patches for padding
        """
        self.data = data
        self.transform = transform
        self.max_patches = max_patches
    
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
                    print(f"Error converting features to tensor at idx {idx}: {e}")
                    # Provide a dummy tensor as fallback
                    features = torch.zeros((self.max_patches, 512), dtype=torch.float32)
            
            # Ensure correct data type
            features = features.float()
            
            # Ensure [n_patches, feature_dim] format
            if features.dim() == 2 and features.shape[1] != 512 and features.shape[0] == 512:
                features = features.transpose(0, 1)  # Transpose to [n_patches, feature_dim]
            
            # Handle case where features might be 1D or have unexpected shape
            if features.dim() == 1:
                if features.shape[0] == 512:
                    # If it's a single 512-dim vector, make it a single patch
                    features = features.unsqueeze(0)
                else:
                    # If it's not 512-dim, create dummy
                    features = torch.zeros((1, 512), dtype=torch.float32)
            elif features.dim() > 2:
                # For higher dimensions, try to reshape intelligently
                if features.shape[-1] == 512:
                    # If the last dimension is 512, reshape to [n, 512]
                    features = features.reshape(-1, 512)
                else:
                    # Otherwise, create a dummy tensor
                    print(f"Warning: Unexpected feature dimension {features.shape} at idx {idx}")
                    features = torch.zeros((1, 512), dtype=torch.float32)
            
            # Pad or truncate the features to max_patches
            n_patches = features.shape[0]
            if n_patches < self.max_patches:
                # Pad with zeros if fewer patches than max_patches
                padding = torch.zeros(self.max_patches - n_patches, 512, 
                                    dtype=features.dtype, device=features.device)
                features = torch.cat([features, padding], dim=0)
            elif n_patches > self.max_patches:
                # Truncate if more patches than max_patches
                features = features[:self.max_patches]
            
            label = torch.tensor(item['label'], dtype=torch.long)
            
            return features, label
        except Exception as e:
            print(f"Error in __getitem__ at idx {idx}: {e}")
            # Return a dummy sample in case of error
            dummy_features = torch.zeros((self.max_patches, 512), dtype=torch.float32)
            dummy_label = torch.tensor(0, dtype=torch.long)
            return dummy_features, dummy_label


def create_cached_splits(data_dir, endpoint='OS_6', val_size=0.15, test_size=0.15, seed=42, cache_dir=None):
    """
    Process data files individually and create stratified splits with caching.
    
    Args:
        data_dir (str): Directory containing .pkl files
        endpoint (str): Which endpoint to use ('OS_6' or 'OS_24')
        val_size (float): Proportion of data for validation
        test_size (float): Proportion of data for testing
        seed (int): Random seed
        cache_dir (str, optional): Directory to cache processed data
        
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
            file_hash = hashlib.md5(os.path.basename(pkl_file).encode()).hexdigest()
            file_cache_path = os.path.join(cache_dir, f"{file_hash}_{endpoint}.pkl")
            
            # Check if cached file exists
            if os.path.exists(file_cache_path):
                with open(file_cache_path, 'rb') as f:
                    file_data = pickle.load(f)
                    file_max_patches = file_data['max_patches']
                    instances = file_data['instances']
            else:
                # Process the file
                file_max_patches, instances = process_pkl_file(pkl_file, endpoint)
                
                # Cache the processed data
                with open(file_cache_path, 'wb') as f:
                    pickle.dump({
                        'max_patches': file_max_patches,
                        'instances': instances
                    }, f)
        else:
            # Process the file without caching
            file_max_patches, instances = process_pkl_file(pkl_file, endpoint)
        
        # Update max_patches
        max_patches = max(max_patches, file_max_patches)
        
        # Skip if no valid instances were found
        if not instances:
            continue
        
        # Extract labels for stratification
        labels = [instance['label'] for instance in instances]
        
        # Stratified split for this file
        train_idx, temp_idx = train_test_split(
            range(len(instances)),
            test_size=val_size + test_size,
            random_state=seed,
            stratify=labels
        )
        
        # Adjust validation size relative to remaining data
        val_test_ratio = val_size / (val_size + test_size)
        val_idx, test_idx = train_test_split(
            temp_idx,
            test_size=1 - val_test_ratio,
            random_state=seed,
            stratify=[labels[i] for i in temp_idx]
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


def prepare_dataloaders(data_dir, endpoint='OS_6', batch_size=16, oversample_factor=1.0, 
                        val_size=0.15, test_size=0.15, num_workers=4, seed=42,
                        use_cache=True, cache_dir=None):
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
    
    Returns:
        tuple: (train_loader, val_loader, test_loader, class_weights, split_metrics, max_patches)
    """
    # Create cache directory if specified and doesn't exist
    if cache_dir and not os.path.exists(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)
    
    # Compute directory checksum to detect changes
    dir_checksum = compute_data_dir_checksum(data_dir)
    
    # Cached split filename
    cached_split_file = os.path.join(cache_dir, f"splits_{endpoint}_{val_size}_{test_size}_{seed}_{dir_checksum}.pkl") if cache_dir else None
    
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
            cache_dir=cache_dir
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
    
    # Create datasets using the cached splits
    train_dataset = CachedDataset(splits['train_data'], max_patches=max_patches)
    val_dataset = CachedDataset(splits['val_data'], max_patches=max_patches)
    test_dataset = CachedDataset(splits['test_data'], max_patches=max_patches)
    
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


# Keeping the existing functions that don't need to be changed
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
    Simplified collate function that stacks the pre-padded tensors.
    Since all tensors are pre-padded to the same size, we don't need custom padding here.
    
    Args:
        batch (list): List of (features, label) tuples
        
    Returns:
        torch.Tensor: Batched features
        torch.Tensor: Batched labels
    """
    try:
        features = []
        labels = []
        
        for i, item in enumerate(batch):
            try:
                feature, label = item
                features.append(feature)
                labels.append(label)
            except Exception as e:
                print(f"Error processing batch item {i}: {e}")
                # Skip problematic items
                continue
        
        if not features:
            # Return a dummy batch if all items were problematic
            dummy_features = torch.zeros((1, 300, 512), dtype=torch.float32)  # Using default max_patches=300
            dummy_labels = torch.zeros(1, dtype=torch.long)
            return dummy_features, dummy_labels
        
        # All tensors should be the same size now, so we can simply stack them
        features_tensor = torch.stack(features)
        labels_tensor = torch.stack(labels)
        
        return features_tensor, labels_tensor
    except Exception as e:
        print(f"Error in collate_fn: {e}")
        # Return a dummy batch in case of error
        dummy_features = torch.zeros((1, 300, 512), dtype=torch.float32)  # Using default max_patches=300
        dummy_labels = torch.zeros(1, dtype=torch.long)
        return dummy_features, dummy_labels