import os
import pickle
import argparse
import numpy as np
import torch
from tqdm import tqdm
from collections import Counter
import matplotlib.pyplot as plt


def analyze_dataset(data_dir, endpoint='OS_6'):
    """
    Analyze the entire dataset to determine the maximum number of patches
    across all instances.
    
    Args:
        data_dir (str): Directory containing .pkl files
        endpoint (str): Which endpoint to use ('OS_6' or 'OS_24')
        
    Returns:
        dict: Statistics about patches in the dataset
    """
    # Get all .pkl files in the directory
    pkl_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.pkl')]
    print(f"Found {len(pkl_files)} .pkl files in {data_dir}")
    
    max_patches = 0
    total_instances = 0
    valid_instances = 0
    patch_counts = []
    
    print("Analyzing all instances to find maximum patches...")
    for pkl_file in tqdm(pkl_files, desc="Processing files"):
        try:
            with open(pkl_file, 'rb') as f:
                instances_list = pickle.load(f)
                total_instances += len(instances_list)
                
                for instance in instances_list:
                    # Skip instances without the specified endpoint
                    if endpoint not in instance:
                        continue
                    
                    valid_instances += 1
                    features = instance['features']
                    
                    # Determine number of patches based on data type and shape
                    if isinstance(features, torch.Tensor):
                        if features.dim() == 2:
                            if features.shape[0] == 512:  # [feature_dim, n_patches]
                                n_patches = features.shape[1]
                            else:  # [n_patches, feature_dim]
                                n_patches = features.shape[0]
                        else:
                            n_patches = 0
                    elif isinstance(features, np.ndarray):
                        if features.ndim == 2:
                            if features.shape[0] == 512:  # [feature_dim, n_patches]
                                n_patches = features.shape[1]
                            else:  # [n_patches, feature_dim]
                                n_patches = features.shape[0]
                        else:
                            n_patches = 0
                    else:
                        try:
                            n_patches = len(features)
                        except:
                            n_patches = 0
                            print(f"Warning: Could not determine patches for an instance in {pkl_file}")
                    
                    patch_counts.append(n_patches)
                    max_patches = max(max_patches, n_patches)
        
        except Exception as e:
            print(f"Error processing {pkl_file}: {e}")
    
    # Calculate statistics
    stats = {
        "max_patches": max_patches,
        "total_instances": total_instances,
        "valid_instances": valid_instances,
        "avg_patches": sum(patch_counts) / len(patch_counts) if patch_counts else 0,
        "median_patches": np.median(patch_counts) if patch_counts else 0,
        "patch_counts": patch_counts
    }
    
    return stats


def plot_patch_distribution(patch_counts, max_patches, output_file=None):
    """
    Plot the distribution of patch counts.
    
    Args:
        patch_counts (list): List of patch counts
        max_patches (int): Maximum number of patches
        output_file (str, optional): File to save the plot to
    """
    plt.figure(figsize=(12, 6))
    
    # Create histogram
    plt.hist(patch_counts, bins=50, alpha=0.7, color='blue')
    
    # Add vertical line for max patches
    plt.axvline(x=max_patches, color='red', linestyle='--', 
                label=f'Max Patches: {max_patches}')
    
    # Add vertical line for median
    median = np.median(patch_counts)
    plt.axvline(x=median, color='green', linestyle='-', 
                label=f'Median: {median:.1f}')
    
    # Add vertical line for mean
    mean = np.mean(patch_counts)
    plt.axvline(x=mean, color='orange', linestyle='-', 
                label=f'Mean: {mean:.1f}')
    
    # Add labels and title
    plt.xlabel('Number of Patches')
    plt.ylabel('Frequency')
    plt.title('Distribution of Patch Counts in Dataset')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if output_file:
        plt.savefig(output_file)
    
    plt.show()


def print_statistics(stats):
    """Print statistics about the dataset."""
    print("\n=== Dataset Statistics ===")
    print(f"Total instances: {stats['total_instances']}")
    print(f"Valid instances (with endpoint): {stats['valid_instances']}")
    print(f"Maximum patches: {stats['max_patches']}")
    print(f"Average patches: {stats['avg_patches']:.2f}")
    print(f"Median patches: {stats['median_patches']}")
    
    # Calculate percentiles
    patch_counts = stats['patch_counts']
    percentiles = [50, 75, 90, 95, 99, 99.5, 99.9]
    print("\nPercentiles:")
    for p in percentiles:
        value = np.percentile(patch_counts, p)
        print(f"  {p}th percentile: {value:.1f}")
    
    # Count instances exceeding certain thresholds
    thresholds = [100, 200, 300, 400, 500, 750, 1000]
    print("\nInstances exceeding thresholds:")
    for threshold in thresholds:
        count = sum(1 for p in patch_counts if p > threshold)
        percentage = 100 * count / len(patch_counts)
        print(f"  > {threshold} patches: {count} instances ({percentage:.2f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze dataset to find maximum patches')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing .pkl files')
    parser.add_argument('--endpoint', type=str, default='OS_6', choices=['OS_6', 'OS_24'], 
                        help='Endpoint to use')
    parser.add_argument('--output', type=str, default=None, 
                        help='Output file for the distribution plot')
    
    args = parser.parse_args()
    
    # Analyze dataset
    stats = analyze_dataset(args.data_dir, args.endpoint)
    
    # Print statistics
    print_statistics(stats)
    
    # Plot distribution
    plot_patch_distribution(stats['patch_counts'], stats['max_patches'], args.output)
    
    print(f"\nRecommended max_patches value: {stats['max_patches']}")
    
    # Suggest a practical max_patches value (e.g., 99.5th percentile)
    practical_max = np.percentile(stats['patch_counts'], 99.5)
    print(f"Practical max_patches value (99.5th percentile): {practical_max:.0f}")
    
    # Calculate potential memory savings
    full_size = stats['max_patches'] * 512 * 4 * stats['valid_instances']  # Assuming float32 (4 bytes)
    practical_size = practical_max * 512 * 4 * stats['valid_instances']
    savings = (full_size - practical_size) / 1024 / 1024  # Convert to MB
    
    print(f"Potential memory savings using practical max: {savings:.2f} MB")