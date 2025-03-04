import os
import pickle
from collections import Counter
import time
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split


class PatchEmbeddingsDataset(Dataset):
    """Dataset for loading patch embeddings from CT scans stored in .pkl files"""
    
    def __init__(self, pkl_files, endpoint='OS_6', transform=None, cache_dir=None):
        """
        Args:
            pkl_files (list): List of paths to .pkl files
            endpoint (str): Which endpoint to use, 'OS_6' or 'OS_24'
            transform (callable, optional): Optional transform to be applied on features
            cache_dir (str, optional): Directory to cache processed data
        """
        self.endpoint = endpoint
        self.transform = transform
        self.data = []
        self.centers = []
        self.cache_dir = cache_dir
        self.cache = {}  # Memory cache for faster access
        
        # Create cache directory if specified and doesn't exist
        if cache_dir and not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
            
        # Progress bar for loading files
        print("Loading data from .pkl files...")
        for pkl_file in tqdm(pkl_files, desc="Loading files"):
            try:
                # Try to load from memory cache first
                file_key = os.path.basename(pkl_file)
                
                # If we have a cache directory, check if the file is already cached
                if self.cache_dir:
                    cache_file = os.path.join(self.cache_dir, f"{file_key}.pt")
                    if os.path.exists(cache_file):
                        # Load from disk cache
                        cached_data = torch.load(cache_file)
                        self.data.extend(cached_data['data'])
                        self.centers.extend(cached_data['centers'])
                        continue
                
                # Not in cache, load from pkl file
                with open(pkl_file, 'rb') as f:
                    # Load the list of dictionaries from the pickle file
                    instances_list = pickle.load(f)
                    center = os.path.basename(pkl_file)  # Use filename as center identifier
                    
                    file_data = []
                    file_centers = []
                    
                    for instance in instances_list:
                        # Skip instances without labels
                        if self.endpoint not in instance:
                            continue
                        
                        # Get features and ensure they're the right shape
                        features = instance['features']  # 512 x n_patches embeddings
                        
                        # Get label
                        label = instance[self.endpoint]  # Get the specified endpoint
                        
                        # Convert to binary if not already
                        if not isinstance(label, int):
                            label = 1 if label else 0
                        
                        file_data.append({
                            'features': features,
                            'label': label,
                            'center': center
                        })
                        file_centers.append(center)
                    
                    # Add to dataset
                    self.data.extend(file_data)
                    self.centers.extend(file_centers)
                    
                    # Cache for future use if cache_dir is specified
                    if self.cache_dir:
                        torch.save({
                            'data': file_data,
                            'centers': file_centers
                        }, cache_file)
                        
            except Exception as e:
                print(f"Error loading {pkl_file}: {e}")
                continue
                
        print(f"Loaded {len(self.data)} samples from {len(pkl_files)} files")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        features = item['features']
        
        if self.transform:
            features = self.transform(features)
        
        # Convert to torch tensors
        features = torch.tensor(features, dtype=torch.float32)
        
        # Check if we need to transpose features
        # We want shape [n_patches, feature_dim] for the transformer
        if features.dim() == 2 and features.shape[0] < features.shape[1]:
            features = features.transpose(0, 1)  # Transpose to [n_patches, feature_dim]
            
        label = torch.tensor(item['label'], dtype=torch.long)
        
        return features, label


def stratified_split(pkl_files, endpoint='OS_6', val_size=0.15, test_size=0.15, random_state=42):
    """
    Split the data stratifying by both label and center.
    
    Args:
        pkl_files (list): List of paths to .pkl files
        endpoint (str): Which endpoint to use, 'OS_6' or 'OS_24'
        val_size (float): Proportion of data for validation
        test_size (float): Proportion of data for testing
        random_state (int): Random seed
        
    Returns:
        dict: Dictionary containing split indices and metadata
    """
    # Extract all instances with labels and their centers
    all_instances = []
    centers = []
    labels = []
    
    print("Processing files for stratified split...")
    for pkl_file in tqdm(pkl_files, desc="Processing files"):
        try:
            with open(pkl_file, 'rb') as f:
                # Load the list of dictionaries
                instances_list = pickle.load(f)
                center = os.path.basename(pkl_file)
                
                for i, instance in enumerate(instances_list):
                    # Skip instances without the specified endpoint
                    if endpoint not in instance:
                        continue
                    
                    # Get the label and convert to binary if needed
                    label = instance[endpoint]
                    if not isinstance(label, int):
                        label = 1 if label else 0
                    
                    all_instances.append((pkl_file, i))  # Store file path and index
                    centers.append(center)
                    labels.append(label)
        except Exception as e:
            print(f"Error loading {pkl_file}: {e}")
            continue
    
    print(f"Found {len(all_instances)} valid instances across all files")
    
    # Convert to numpy arrays for scikit-learn
    instances_array = np.array(all_instances, dtype=object)
    centers_array = np.array(centers)
    labels_array = np.array(labels)
    
    # Create a composite stratification variable combining label and center
    strat_var = [f"{l}_{c}" for l, c in zip(labels_array, centers_array)]
    
    # First split: train+val vs test
    train_val_idx, test_idx, _, _ = train_test_split(
        np.arange(len(all_instances)),
        strat_var,
        test_size=test_size,
        random_state=random_state,
        stratify=strat_var
    )
    
    # Second split: train vs val
    strat_var_train_val = [strat_var[i] for i in train_val_idx]
    
    adjusted_val_size = val_size / (1 - test_size)  # Adjust val_size relative to train+val
    train_idx, val_idx, _, _ = train_test_split(
        train_val_idx,
        strat_var_train_val,
        test_size=adjusted_val_size,
        random_state=random_state,
        stratify=strat_var_train_val
    )
    
    # Count instances per split
    train_count = len(train_idx)
    val_count = len(val_idx)
    test_count = len(test_idx)
    
    # Count labels per split
    train_label_counts = Counter(labels_array[train_idx])
    val_label_counts = Counter(labels_array[val_idx])
    test_label_counts = Counter(labels_array[test_idx])
    
    # Extract instances for each split
    train_instances = instances_array[train_idx]
    val_instances = instances_array[val_idx]
    test_instances = instances_array[test_idx]
    
    return {
        'train': train_instances,
        'val': val_instances,
        'test': test_instances,
        'train_labels': labels_array[train_idx],
        'val_labels': labels_array[val_idx],
        'test_labels': labels_array[test_idx],
        'train_centers': centers_array[train_idx],
        'val_centers': centers_array[val_idx],
        'test_centers': centers_array[test_idx],
        'all_instances': all_instances,
        'centers': centers,
        'labels': labels,
        'metrics': {
            'train_count': train_count,
            'val_count': val_count,
            'test_count': test_count,
            'train_label_counts': train_label_counts,
            'val_label_counts': val_label_counts,
            'test_label_counts': test_label_counts
        }
    }


class SplitDataset(Dataset):
    """Dataset for a specific split (train, val, or test)"""
    
    def __init__(self, split_data, split_type, endpoint='OS_6', transform=None, use_cache=True):
        """
        Args:
            split_data (dict): Output from stratified_split function
            split_type (str): 'train', 'val', or 'test'
            endpoint (str): Which endpoint to use, 'OS_6' or 'OS_24'
            transform (callable, optional): Optional transform to be applied on features
            use_cache (bool): Whether to cache data in memory
        """
        self.endpoint = endpoint
        self.transform = transform
        self.instances = split_data[split_type]
        self.labels = split_data[f'{split_type}_labels']
        self.use_cache = use_cache
        self.data_cache = {} if use_cache else None
        
        # Load the actual data
        self.data = []
        print(f"Loading {split_type} data...")
        
        # Use tqdm for progress tracking
        for i, ((pkl_file, idx), label) in enumerate(tqdm(zip(self.instances, self.labels), 
                                                          desc=f"Loading {split_type} set", 
                                                          total=len(self.instances))):
            # Check if this pkl_file is already in cache
            if self.use_cache and pkl_file in self.data_cache:
                instances_list = self.data_cache[pkl_file]
            else:
                with open(pkl_file, 'rb') as f:
                    # Load the list of dictionaries from the pickle file
                    instances_list = pickle.load(f)
                    if self.use_cache:
                        self.data_cache[pkl_file] = instances_list
                
            # Get the specific instance at the given index
            instance = instances_list[idx]
            
            # Extract features
            features = instance['features']
            
            self.data.append({
                'features': features,
                'label': label
            })
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        features = item['features']
        
        if self.transform:
            features = self.transform(features)
        
        # Convert to torch tensors
        features = torch.tensor(features, dtype=torch.float32)
        
        # Check if we need to transpose features
        if features.dim() == 2 and features.shape[0] < features.shape[1]:
            features = features.transpose(0, 1)  # Transpose to [n_patches, feature_dim]
            
        label = torch.tensor(item['label'], dtype=torch.long)
        
        return features, label


def collate_fn(batch):
    """
    Custom collate function to handle variable number of patches.
    
    Args:
        batch (list): List of (features, label) tuples
        
    Returns:
        torch.Tensor: Batched features
        torch.Tensor: Batched labels
    """
    features = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    
    # Check if features need to be transposed (if they're in shape [feature_dim, n_patches])
    # We want shape [n_patches, feature_dim] for the transformer
    transposed_features = []
    for f in features:
        # If the second dimension is larger than the first, it's likely in [feature_dim, n_patches] format
        if f.dim() == 2 and f.shape[0] < f.shape[1]:
            transposed_features.append(f.transpose(0, 1))  # Transpose to [n_patches, feature_dim]
        else:
            transposed_features.append(f)
    
    # Pad features to the same number of patches
    max_patches = max(f.shape[0] for f in transposed_features)
    feature_dim = transposed_features[0].shape[1]
    
    padded_features = []
    for f in transposed_features:
        n_patches = f.shape[0]
        if n_patches < max_patches:
            # Pad with zeros
            padding = torch.zeros(max_patches - n_patches, feature_dim, dtype=f.dtype)
            padded_f = torch.cat([f, padding], dim=0)
        else:
            padded_f = f
        padded_features.append(padded_f)
    
    # Stack features and labels
    features_tensor = torch.stack(padded_features)
    labels_tensor = torch.stack(labels)
    
    return features_tensor, labels_tensor


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


def prepare_dataloaders(data_dir, endpoint='OS_6', batch_size=16, oversample_factor=1.0, 
                        val_size=0.15, test_size=0.15, num_workers=4, seed=42,
                        use_cache=True, cache_dir=None):
    """
    Prepare DataLoaders for training, validation, and testing.
    
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
        tuple: (train_loader, val_loader, test_loader, class_weights, split_metrics)
    """
    # Get all .pkl files in the directory
    pkl_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.pkl')]
    print(f"Found {len(pkl_files)} .pkl files in {data_dir}")
    
    # Split data
    split_data = stratified_split(
        pkl_files=pkl_files,
        endpoint=endpoint,
        val_size=val_size,
        test_size=test_size,
        random_state=seed
    )
    
    # Print split statistics
    metrics = split_data['metrics']
    print(f"Split statistics:")
    print(f"  Train: {metrics['train_count']} samples, {dict(metrics['train_label_counts'])}")
    print(f"  Validation: {metrics['val_count']} samples, {dict(metrics['val_label_counts'])}")
    print(f"  Test: {metrics['test_count']} samples, {dict(metrics['test_label_counts'])}")
    
    # Create datasets for each split
    train_dataset = SplitDataset(split_data, 'train', endpoint=endpoint, use_cache=use_cache)
    val_dataset = SplitDataset(split_data, 'val', endpoint=endpoint, use_cache=use_cache)
    test_dataset = SplitDataset(split_data, 'test', endpoint=endpoint, use_cache=use_cache)
    
    # Create weighted sampler for handling class imbalance if oversampling is enabled
    train_sampler = None
    if oversample_factor > 0:
        train_sampler = create_weighted_sampler(
            labels=split_data['train_labels'],
            oversample_factor=oversample_factor
        )
        shuffle = False  # Don't shuffle when using sampler
    else:
        shuffle = True  # Shuffle when not using sampler
    
    # Create data loaders
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
    
    # Calculate class weights for loss function
    class_counts = Counter(split_data['train_labels'])
    total_samples = len(split_data['train_labels'])
    class_weights = torch.tensor(
        [total_samples / (len(class_counts) * count) for label, count in sorted(class_counts.items())],
        dtype=torch.float32
    )
    
    # Get max patches from training data
    sample_features, _ = train_dataset[0]
    samples = [train_dataset[i] for i in range(min(10, len(train_dataset)))]
    max_patches = max(sample[0].shape[0] for sample in samples)
    print(f"Using max_patches={max_patches} based on training data sample")
    
    return (
        train_loader, 
        val_loader, 
        test_loader, 
        class_weights, 
        metrics, 
        max_patches
    )