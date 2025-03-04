import os
import pickle
import hashlib
import numpy as np
from tqdm import tqdm
from collections import Counter
from sklearn.model_selection import KFold
import torch
from torch.utils.data import DataLoader

from dataloader import process_pkl_file, CachedDataset, create_weighted_sampler, collate_fn


def create_cached_folds(data_dir, endpoint='OS_6', n_folds=5, seed=42, cache_dir=None):
    """
    Process data files and create n stratified folds with caching.
    
    Args:
        data_dir (str): Directory containing .pkl files
        endpoint (str): Which endpoint to use ('OS_6' or 'OS_24')
        n_folds (int): Number of folds for cross-validation
        seed (int): Random seed
        cache_dir (str, optional): Directory to cache processed data
        
    Returns:
        tuple: (folds, max_patches, class_weights)
    """
    # Get all .pkl files in the directory
    pkl_files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.pkl')])
    print(f"Found {len(pkl_files)} .pkl files in {data_dir}")
    
    # Create cache folder if needed
    if cache_dir and not os.path.exists(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)
    
    # Check if we have cached fold assignments
    folds_cache_path = None
    if cache_dir:
        # Create a hash of sorted filenames to detect changes in dataset
        files_hash = hashlib.md5("".join(sorted([os.path.basename(f) for f in pkl_files])).encode()).hexdigest()
        folds_cache_path = os.path.join(cache_dir, f"folds_{n_folds}_{endpoint}_{seed}_{files_hash}.pkl")
        
        if os.path.exists(folds_cache_path):
            print(f"Loading cached fold assignments from {folds_cache_path}")
            with open(folds_cache_path, 'rb') as f:
                cache_data = pickle.load(f)
                return cache_data['folds'], cache_data['max_patches'], cache_data['class_weights']
    
    # Initialize folds and counters
    folds = [[] for _ in range(n_folds)]
    fold_counts = [{'total': 0, 0: 0, 1: 0} for _ in range(n_folds)]
    max_patches = 0
    
    # Process .pkl files
    file_data = []  # Store file index, labels for stratification
    processed_instances = []  # Store all processed instances
    
    print("Processing files for fold creation...")
    for file_idx, pkl_file in enumerate(tqdm(pkl_files)):
        # Process or load from cache
        file_cache_path = None
        if cache_dir:
            file_hash = hashlib.md5(os.path.basename(pkl_file).encode()).hexdigest()
            file_cache_path = os.path.join(cache_dir, f"{file_hash}_{endpoint}.pkl")
            
            if os.path.exists(file_cache_path):
                with open(file_cache_path, 'rb') as f:
                    file_data_cache = pickle.load(f)
                    file_max_patches = file_data_cache['max_patches']
                    instances = file_data_cache['instances']
            else:
                file_max_patches, instances = process_pkl_file(pkl_file, endpoint)
                
                with open(file_cache_path, 'wb') as f:
                    pickle.dump({
                        'max_patches': file_max_patches,
                        'instances': instances
                    }, f)
        else:
            file_max_patches, instances = process_pkl_file(pkl_file, endpoint)
        
        max_patches = max(max_patches, file_max_patches)
        
        if not instances:
            continue
        
        # Count labels for this file (for stratification)
        label_counts = Counter([instance['label'] for instance in instances])
        
        # Store file info for later fold assignment
        file_data.append({
            'file_idx': file_idx,
            'pkl_file': pkl_file,
            'instances': instances,
            'label_counts': label_counts
        })
        
        # Add to processed instances
        processed_instances.extend(instances)
    
    # Calculate class weights based on all data
    all_labels = [instance['label'] for instance in processed_instances]
    class_counts = Counter(all_labels)
    total_samples = len(all_labels)
    num_classes = len(class_counts)
    class_weights = torch.tensor(
        [total_samples / (num_classes * count) for label, count in sorted(class_counts.items())],
        dtype=torch.float32
    )
    
    # Create stratified folds by distributing files
    # We'll sort files by their positive class ratio to ensure stratification
    file_data.sort(key=lambda x: x['label_counts'].get(1, 0) / (x['label_counts'].get(0, 1) + x['label_counts'].get(1, 0)))
    
    # Distribute files to folds with a "snake" pattern to ensure stratification
    # For example, with 5 folds, files are distributed like: 0,1,2,3,4,4,3,2,1,0,0,1,...
    fold_indices = []
    for i in range(n_folds):
        fold_indices.append(i)
    for i in range(n_folds-2, -1, -1):
        fold_indices.append(i)
    
    fold_idx_cycle = fold_indices * (len(file_data) // len(fold_indices) + 1)
    fold_idx_cycle = fold_idx_cycle[:len(file_data)]
    
    # Assign each file to a fold
    for i, file_info in enumerate(file_data):
        fold_idx = fold_idx_cycle[i]
        for instance in file_info['instances']:
            folds[fold_idx].append(instance)
            fold_counts[fold_idx]['total'] += 1
            fold_counts[fold_idx][instance['label']] += 1
    
    # Print fold statistics
    print(f"Fold statistics:")
    for i, count in enumerate(fold_counts):
        pos_ratio = count.get(1, 0) / count['total'] * 100 if count['total'] > 0 else 0
        print(f"  Fold {i+1}: {count['total']} samples, {count.get(0, 0)} negative, {count.get(1, 0)} positive ({pos_ratio:.1f}% positive)")
    print(f"  Max patches: {max_patches}")
    
    # Save folds to cache
    if folds_cache_path:
        with open(folds_cache_path, 'wb') as f:
            pickle.dump({
                'folds': folds,
                'max_patches': max_patches,
                'class_weights': class_weights
            }, f)
        print(f"Saved fold assignments to cache: {folds_cache_path}")
    
    return folds, max_patches, class_weights


def create_fold_loaders(folds, fold_idx, batch_size=32, oversample_factor=1.0, 
                       max_patches=300, num_workers=4):
    """
    Create dataloaders for a specific fold in cross-validation.
    
    Args:
        folds (list): List of fold datasets
        fold_idx (int): Index of the validation fold
        batch_size (int): Batch size
        oversample_factor (float): Factor for oversampling minority class
        max_patches (int): Maximum number of patches
        num_workers (int): Number of data loading workers
        
    Returns:
        tuple: (train_loader, val_loader)
    """
    # Create train dataset from all folds except current
    train_data = []
    for i in range(len(folds)):
        if i != fold_idx:
            train_data.extend(folds[i])
    
    # Use current fold for validation
    val_data = folds[fold_idx]
    
    # Create datasets
    train_dataset = CachedDataset(train_data, max_patches=max_patches)
    val_dataset = CachedDataset(val_data, max_patches=max_patches)
    
    # Create sampler for training
    train_sampler = None
    if oversample_factor > 0:
        train_sampler = create_weighted_sampler(
            labels=[item['label'] for item in train_data],
            oversample_factor=oversample_factor
        )
        shuffle = False
    else:
        shuffle = True
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=shuffle if train_sampler is None else False,
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
    
    return train_loader, val_loader