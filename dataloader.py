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
            
            for idx, instance in enumerate(instances_list):
                try:
                    # Skip instances without the specified endpoint
                    if endpoint not in instance:
                        continue
                    
                    # Get features and ensure they're the right shape
                    features = instance['features']
                    
                    # Convert features to torch tensor for manipulation
                    if not isinstance(features, torch.Tensor):
                        try:
                            features = torch.tensor(features, dtype=torch.float32)
                        except Exception as e:
                            print(f"Warning: Could not convert features to tensor in {pkl_file}, instance {idx}: {e}")
                            continue
                    
                    # Transpose if in [feature_dim, n_patches] format
                    if features.dim() == 2 and features.shape[0] == 512:
                        features = features.transpose(0, 1)  # Now [n_patches, feature_dim]
                    
                    # Update max_patches
                    n_patches = features.shape[0]
                    max_patches = max(max_patches, n_patches)
                    
                    # Get label and handle different types carefully
                    label_value = instance[endpoint]  # Get the specified endpoint
                    
                    # Handle different types of label values
                    if isinstance(label_value, (int, bool, np.int32, np.int64)):
                        # For simple types, convert directly
                        label = 1 if label_value else 0
                    elif isinstance(label_value, torch.Tensor):
                        # For tensors, extract a single value if possible
                        if label_value.numel() == 1:
                            # Single element tensor
                            label = 1 if label_value.item() else 0
                        else:
                            # Multi-element tensor, use the first element or skip
                            print(f"Warning: Multi-element label tensor in {pkl_file}, instance {idx}. Using first element.")
                            try:
                                label = 1 if label_value[0].item() else 0
                            except:
                                print(f"Warning: Could not extract label from tensor. Skipping instance.")
                                continue
                    elif isinstance(label_value, np.ndarray):
                        # For numpy arrays, similar approach as tensors
                        if label_value.size == 1:
                            label = 1 if label_value.item() else 0
                        else:
                            print(f"Warning: Multi-element label array in {pkl_file}, instance {idx}. Using first element.")
                            try:
                                label = 1 if label_value.flatten()[0] else 0
                            except:
                                print(f"Warning: Could not extract label from array. Skipping instance.")
                                continue
                    else:
                        # For other types, use truthiness
                        print(f"Warning: Unexpected label type {type(label_value)} in {pkl_file}, instance {idx}.")
                        try:
                            label = 1 if label_value else 0
                        except:
                            print(f"Warning: Could not convert label to binary. Skipping instance.")
                            continue
                    
                    instances.append({
                        'features': features,
                        'label': label,
                        'center': center
                    })
                except Exception as e:
                    print(f"Error processing instance {idx} in {pkl_file}: {e}")
                    continue
                
        if not instances:
            print(f"Warning: No valid instances found in {pkl_file}")
            
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
            
            # Convert to torch tensors if not already
            if not isinstance(features, torch.Tensor):
                try:
                    features = torch.tensor(features, dtype=torch.float32)
                except ValueError as e:
                    print(f"Error converting features to tensor at idx {idx}: {e}")
                    # Provide a dummy tensor as fallback
                    features = torch.zeros((self.max_patches, 512), dtype=torch.float32)
            
            # Ensure correct data type if already a tensor
            features = features.float()
            
            # Ensure [n_patches, feature_dim] format
            if features.dim() == 2 and features.shape[1] != 512 and features.shape[0] == 512:
                features = features.transpose(0, 1)  # Transpose to [n_patches, feature_dim]
            
            # Handle case where features might be 1D or have unexpected shape
            if features.dim() == 1:
                # If it's a 1D tensor, reshape it to 2D with a single feature
                features = features.unsqueeze(0)
                if features.shape[1] != 512:
                    features = torch.zeros((1, 512), dtype=torch.float32)
            elif features.dim() > 2:
                # For higher dimensions, flatten to 2D
                orig_shape = features.shape
                print(f"Warning: Unexpected feature dimension {orig_shape} at idx {idx}, flattening to 2D")
                features = features.reshape(-1, 512)
                if features.shape[1] != 512:
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