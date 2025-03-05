import torch
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
import matplotlib.pyplot as plt
import os
import json
from io import BytesIO

def evaluate_by_center(model, test_loader, device='cuda' if torch.cuda.is_available() else 'cpu', 
                      neptune_run=None):
    """
    Evaluate model performance broken down by center.
    
    Args:
        model (nn.Module): Trained model
        test_loader (DataLoader): Test data loader
        device (str): Device to use for evaluation
        neptune_run: Neptune run object for logging (optional)
        
    Returns:
        dict: Evaluation metrics by center
    """
    model = model.to(device)
    model.eval()
    
    # Collect predictions and centers
    centers = []
    all_labels = []
    all_preds = []
    all_probs = []
    
    # Use tqdm for progress tracking
    with torch.no_grad():
        for batch_idx, (features, labels) in enumerate(tqdm(test_loader, desc="Evaluating by center")):
            features, labels = features.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(features)
            
            # Get predictions and probabilities
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            # Store results
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of positive class
            
            # Get centers from dataset
            batch_centers = []
            for i in range(len(labels)):
                # Get the index in the dataset
                dataset_idx = batch_idx * test_loader.batch_size + i
                if dataset_idx < len(test_loader.dataset):
                    # Get the center from the instance
                    center = test_loader.dataset.data[dataset_idx].get('center', 'unknown')
                    batch_centers.append(center)
                else:
                    batch_centers.append('unknown')
            
            centers.extend(batch_centers)
    
    # Convert to numpy arrays
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    centers = np.array(centers)
    
    # Group by center
    unique_centers = np.unique(centers)
    center_metrics = {}
    
    print(f"\n===== EVALUATION BY CENTER ({len(unique_centers)} centers) =====")
    
    for center in unique_centers:
        center_mask = centers == center
        center_labels = all_labels[center_mask]
        center_preds = all_preds[center_mask]
        center_probs = all_probs[center_mask]
        
        # Calculate metrics
        try:
            accuracy = accuracy_score(center_labels, center_preds)
            precision = precision_score(center_labels, center_preds, zero_division=0)
            recall = recall_score(center_labels, center_preds, zero_division=0)
            f1 = f1_score(center_labels, center_preds, zero_division=0)
            f1_macro = f1_score(center_labels, center_preds, average='macro', zero_division=0)
            
            try:
                auc = roc_auc_score(center_labels, center_probs)
            except:
                auc = 0.0  # In case of only one class
                
            cm = confusion_matrix(center_labels, center_preds)
            
            # Count positive and negative samples
            neg_count = np.sum(center_labels == 0)
            pos_count = np.sum(center_labels == 1)
            total_count = len(center_labels)
            pos_ratio = pos_count / total_count if total_count > 0 else 0
            
            metrics = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'f1_macro': f1_macro,
                'auc': auc,
                'confusion_matrix': cm,
                'sample_count': total_count,
                'positive_count': pos_count,
                'negative_count': neg_count,
                'positive_ratio': pos_ratio,
                'center_labels': center_labels,
                'center_preds': center_preds,
                'center_probs': center_probs
            }
            
            center_metrics[center] = metrics
            
            # Print center metrics
            print(f"\nCenter: {center}")
            print(f"  Samples: {total_count} ({pos_count} positive, {neg_count} negative, {pos_ratio:.1%} positive ratio)")
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  F1 Macro: {f1_macro:.4f}")
            print(f"  AUC: {auc:.4f}")
            print(f"  Confusion Matrix:\n{cm}")
            
            # Log metrics to Neptune
            if neptune_run:
                for metric_name in ['accuracy', 'precision', 'recall', 'f1', 'f1_macro', 'auc']:
                    neptune_run[f"evaluation_by_center/{center}_{metric_name}"] = metrics[metric_name]
        except Exception as e:
            print(f"Error calculating metrics for center {center}: {e}")
            center_metrics[center] = {
                'error': str(e),
                'sample_count': np.sum(center_mask)
            }
    
    # Calculate overall weighted averages for key metrics
    total_samples = sum(metrics['sample_count'] for metrics in center_metrics.values() if 'sample_count' in metrics)
    
    weighted_metrics = {}
    for metric_name in ['accuracy', 'f1_macro', 'auc']:
        weighted_sum = sum(
            metrics[metric_name] * metrics['sample_count'] 
            for center, metrics in center_metrics.items() 
            if metric_name in metrics and 'sample_count' in metrics
        )
        weighted_metrics[f'weighted_{metric_name}'] = weighted_sum / total_samples if total_samples > 0 else 0
    
    print("\nWeighted averages across centers:")
    for metric_name, value in weighted_metrics.items():
        print(f"  {metric_name}: {value:.4f}")
        if neptune_run:
            neptune_run[f"evaluation_by_center/{metric_name}"] = value
    
    # Return overall results
    result = {
        'center_metrics': center_metrics,
        'weighted_metrics': weighted_metrics,
        'all_labels': all_labels,
        'all_preds': all_preds,
        'all_probs': all_probs,
        'centers': centers
    }
    
    return result

def plot_center_metrics(center_metrics, key_metrics=['f1_macro', 'auc'], 
                        output_dir=None, neptune_run=None, min_samples=10):
    """
    Create bar charts comparing center performance metrics.
    
    Args:
        center_metrics (dict): Dictionary with metrics for each center
        key_metrics (list): List of metrics to visualize
        output_dir (str, optional): Directory to save plots
        neptune_run: Neptune run object for logging (optional)
        min_samples (int): Minimum number of samples for a center to be included
    """
    if not center_metrics or 'center_metrics' not in center_metrics:
        print("No center metrics available to plot.")
        return
    
    # Get centers with enough samples
    valid_centers = []
    counts = []
    pos_ratios = []
    
    for center, metrics in center_metrics['center_metrics'].items():
        if 'sample_count' in metrics and metrics['sample_count'] >= min_samples:
            valid_centers.append(center)
            counts.append(metrics['sample_count'])
            pos_ratios.append(metrics.get('positive_ratio', 0))
    
    if not valid_centers:
        print(f"No centers with at least {min_samples} samples found.")
        return
    
    # Sort centers by sample count
    sorted_indices = np.argsort(counts)[::-1]  # Descending order
    sorted_centers = [valid_centers[i] for i in sorted_indices]
    sorted_counts = [counts[i] for i in sorted_indices]
    sorted_pos_ratios = [pos_ratios[i] for i in sorted_indices]
    
    # Create figure based on number of metrics to plot
    n_metrics = len(key_metrics) + 1  # +1 for sample counts
    fig_height = 4 * n_metrics
    plt.figure(figsize=(14, fig_height))
    
    # Plot sample counts first
    plt.subplot(n_metrics, 1, 1)
    bars = plt.bar(sorted_centers, sorted_counts)
    
    # Add positive ratio as text
    for i, (bar, ratio) in enumerate(zip(bars, sorted_pos_ratios)):
        plt.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 5,
            f"{ratio:.1%} pos",
            ha='center', 
            va='bottom',
            rotation=0,
            fontsize=8
        )
    
    plt.title('Sample Count by Center')
    plt.ylabel('Number of Samples')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    # Plot each requested metric
    for i, metric in enumerate(key_metrics):
        plt.subplot(n_metrics, 1, i+2)
        
        metric_values = []
        for center in sorted_centers:
            center_data = center_metrics['center_metrics'][center]
            if metric in center_data:
                metric_values.append(center_data[metric])
            else:
                metric_values.append(0)
        
        # Add weighted average reference line
        weighted_avg = center_metrics['weighted_metrics'].get(f'weighted_{metric}', None)
        
        # Plot bars
        bars = plt.bar(sorted_centers, metric_values, 
                      color='lightblue' if metric == 'f1_macro' else 'lightgreen')
        
        # Add horizontal line for weighted average
        if weighted_avg is not None:
            plt.axhline(y=weighted_avg, color='red', linestyle='--', 
                       label=f'Weighted Avg: {weighted_avg:.3f}')
            plt.legend()
        
        # Add value labels on top of bars
        for bar in bars:
            plt.text(
                bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.01,
                f"{bar.get_height():.3f}",
                ha='center', 
                va='bottom',
                rotation=0,
                fontsize=8
            )
        
        plt.title(f'{metric.upper()} by Center')
        plt.ylabel(metric.replace('_', ' ').title())
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, 1.05)  # For metrics that range from 0 to 1
        plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    fig = plt.gcf()
    
    # Save figure locally if output_dir is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, 'center_metrics.png')
        fig.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Center metrics visualization saved to {output_file}")
    
    # Log figure to Neptune
    if neptune_run:
        try:
            from neptune_utils import log_figure
            log_figure(neptune_run, fig, "center_metrics_comparison")
        except Exception as e:
            print(f"Warning: Failed to log figure to Neptune: {e}")
    
    plt.close(fig)