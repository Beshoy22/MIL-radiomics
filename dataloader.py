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
    
    def __init__(self, pkl_files, endpoint='OS_6', transform=None, cache_dir=None, max_patches=None):
        """
        Args:
            pkl_files (list): List of paths to .pkl files
            endpoint (str): Which endpoint to use, 'OS_6' or 'OS_24'
            transform (callable, optional): Optional transform to be applied on features
            cache_dir (str, optional): Directory to cache processed data
            max_patches (int, optional): Maximum number of patches for padding
        """
        self.endpoint = endpoint
        self.transform = transform
        self.data = []
        self.centers = []
        self.cache_dir = cache_dir
        self.cache = {}  # Memory cache for faster access
        self.max_patches = max_patches
        
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
                        
                        # Convert features to torch tensor for manipulation
                        if not isinstance(features, torch.Tensor):
                            features = torch.tensor(features, dtype=torch.float32)
                        
                        # Transpose if in [feature_dim, n_patches] format
                        if features.dim() == 2 and features.shape[0] == 512:
                            features = features.transpose(0, 1)  # Now [n_patches, feature_dim]
                        
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
        
        # If max_patches not provided, determine it from the data
        if self.max_patches is None:
            self._determine_max_patches()
    
    def _determine_max_patches(self):
        """Determine maximum number of patches across all samples"""
        max_patches = 0
        for item in self.data:
            features = item['features']
            if isinstance(features, torch.Tensor):
                n_patches = features.shape[0]
            else:
                n_patches = len(features)
            max_patches = max(max_patches, n_patches)
        
        self.max_patches = max_patches
        print(f"Determined max_patches = {self.max_patches}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        features = item['features']
        
        if self.transform:
            features = self.transform(features)
        
        # Convert to torch tensors if not already
        if not isinstance(features, torch.Tensor):
            features = torch.tensor(features, dtype=torch.float32)
        
        # Ensure correct data type if already a tensor
        features = features.float()
        
        # Ensure [n_patches, feature_dim] format
        if features.dim() == 2 and features.shape[1] != 512 and features.shape[0] == 512:
            features = features.transpose(0, 1)  # Transpose to [n_patches, feature_dim]
        
        # Pad the features if needed
        if self.max_patches is not None:
            n_patches = features.shape[0]
            if n_patches < self.max_patches:
                padding = torch.zeros(self.max_patches - n_patches, features.shape[1], 
                                     dtype=features.dtype, device=features.device)
                features = torch.cat([features, padding], dim=0)
            elif n_patches > self.max_patches:
                # If more patches than max_patches, truncate
                features = features[:self.max_patches]
                
        label = torch.tensor(item['label'], dtype=torch.long)
        
        return features, label


class SplitDataset(Dataset):
    """Dataset for a specific split (train, val, or test)"""
    
    def __init__(self, split_data, split_type, endpoint='OS_6', transform=None, 
                use_cache=True, max_patches=None):
        """
        Args:
            split_data (dict): Output from stratified_split function
            split_type (str): 'train', 'val', or 'test'
            endpoint (str): Which endpoint to use, 'OS_6' or 'OS_24'
            transform (callable, optional): Optional transform to be applied on features
            use_cache (bool): Whether to cache data in memory
            max_patches (int, optional): Maximum number of patches for padding
        """
        self.endpoint = endpoint
        self.transform = transform
        self.instances = split_data[split_type]
        self.labels = split_data[f'{split_type}_labels']
        self.use_cache = use_cache
        self.data_cache = {} if use_cache else None
        self.max_patches = max_patches
        
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
            
            # Convert features to torch tensor for manipulation
            if not isinstance(features, torch.Tensor):
                features = torch.tensor(features, dtype=torch.float32)
            
            # Transpose if in [feature_dim, n_patches] format
            if features.dim() == 2 and features.shape[0] == 512:
                features = features.transpose(0, 1)  # Now [n_patches, feature_dim]
            
            self.data.append({
                'features': features,
                'label': label
            })
        
        # If max_patches not provided, determine it from the data
        if self.max_patches is None:
            self._determine_max_patches()
    
    def _determine_max_patches(self):
        """Determine maximum number of patches across all samples"""
        max_patches = 0
        for item in self.data:
            features = item['features']
            if isinstance(features, torch.Tensor):
                n_patches = features.shape[0]
            else:
                n_patches = len(features)
            max_patches = max(max_patches, n_patches)
        
        self.max_patches = max_patches
        print(f"Determined max_patches = {self.max_patches}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        features = item['features']
        
        if self.transform:
            features = self.transform(features)
        
        # Convert to torch tensors if not already
        if not isinstance(features, torch.Tensor):
            features = torch.tensor(features, dtype=torch.float32)
        
        # Ensure correct data type if already a tensor
        features = features.float()
        
        # Ensure [n_patches, feature_dim] format
        if features.dim() == 2 and features.shape[1] != 512 and features.shape[0] == 512:
            features = features.transpose(0, 1)  # Transpose to [n_patches, feature_dim]
        
        # Pad the features if needed
        if self.max_patches is not None:
            n_patches = features.shape[0]
            if n_patches < self.max_patches:
                padding = torch.zeros(self.max_patches - n_patches, features.shape[1], 
                                     dtype=features.dtype, device=features.device)
                features = torch.cat([features, padding], dim=0)
            elif n_patches > self.max_patches:
                # If more patches than max_patches, truncate
                features = features[:self.max_patches]
                
        label = torch.tensor(item['label'], dtype=torch.long)
        
        return features, label


def collate_fn(batch):
    """
    Modified collate function that simply stacks the already padded tensors.
    Since all tensors are pre-padded to the same size, we don't need custom padding here.
    
    Args:
        batch (list): List of (features, label) tuples
        
    Returns:
        torch.Tensor: Batched features
        torch.Tensor: Batched labels
    """
    features = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    
    # For debugging (can be removed in production)
    feature_shapes = [f.shape for f in features]
    print(f"Feature shapes in batch: {feature_shapes}")
    
    # All tensors should be the same size now, so we can simply stack them
    features_tensor = torch.stack(features)
    labels_tensor = torch.stack(labels)
    
    return features_tensor, labels_tensor


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
        tuple: (train_loader, val_loader, test_loader, class_weights, split_metrics, max_patches)
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
    
    # First, determine max_patches from a subset of data
    print("Determining max_patches from a subset of data...")
    max_patches = determine_max_patches(split_data, subset_size=50)
    print(f"Using max_patches = {max_patches} for all datasets")
    
    # Create datasets for each split with the pre-determined max_patches
    train_dataset = SplitDataset(split_data, 'train', endpoint=endpoint, 
                                use_cache=use_cache, max_patches=max_patches)
    val_dataset = SplitDataset(split_data, 'val', endpoint=endpoint, 
                              use_cache=use_cache, max_patches=max_patches)
    test_dataset = SplitDataset(split_data, 'test', endpoint=endpoint, 
                               use_cache=use_cache, max_patches=max_patches)
    
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
    
    # Create data loaders - using our simplified collate_fn since padding is done in datasets
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
    
    return (
        train_loader, 
        val_loader, 
        test_loader, 
        class_weights, 
        metrics, 
        max_patches
    )


def determine_max_patches(split_data, subset_size=50):
    """
    Determine maximum number of patches from a subset of the data.
    
    Args:
        split_data (dict): Output from stratified_split function
        subset_size (int): Number of instances to check
        
    Returns:
        int: Maximum number of patches
    """
    max_patches = 0
    
    # Combine instances from all splits
    all_instances = np.concatenate([
        split_data['train'][:min(subset_size // 3, len(split_data['train']))],
        split_data['val'][:min(subset_size // 3, len(split_data['val']))],
        split_data['test'][:min(subset_size // 3, len(split_data['test']))]
    ])
    
    print(f"Checking {len(all_instances)} instances to determine max_patches...")
    
    for pkl_file, idx in tqdm(all_instances):
        try:
            with open(pkl_file, 'rb') as f:
                instances_list = pickle.load(f)
                instance = instances_list[idx]
                features = instance['features']
                
                # Check shape of features
                if isinstance(features, torch.Tensor):
                    # If already a tensor, check dimensions
                    if features.dim() == 2:
                        if features.shape[0] == 512:  # [feature_dim, n_patches]
                            n_patches = features.shape[1]
                        else:  # [n_patches, feature_dim]
                            n_patches = features.shape[0]
                    else:
                        n_patches = 0  # Skip invalid tensors
                elif isinstance(features, np.ndarray):
                    # If numpy array, check dimensions
                    if features.ndim == 2:
                        if features.shape[0] == 512:  # [feature_dim, n_patches]
                            n_patches = features.shape[1]
                        else:  # [n_patches, feature_dim]
                            n_patches = features.shape[0]
                    else:
                        n_patches = 0  # Skip invalid arrays
                else:
                    # Try to infer dimensionality from other types
                    n_patches = len(features)
                
                max_patches = max(max_patches, n_patches)
        except Exception as e:
            print(f"Error checking instance {idx} in {pkl_file}: {e}")
            continue
    
    # Add a small buffer for safety
    max_patches = int(max_patches * 1.1)
    
    return max_patches