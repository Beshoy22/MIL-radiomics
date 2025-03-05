"""
Example file showing how to implement custom metrics and visualizations with Neptune.ai
This file is meant as a reference for extending the current Neptune implementation.
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from sklearn.manifold import TSNE
import seaborn as sns
from neptune_utils import log_figure


def log_feature_importance(neptune_run, model, input_size=512, n_features=20):
    """
    Log feature importance based on model weights.
    This is a simple example - the actual implementation will depend on the model architecture.
    
    Args:
        neptune_run: Neptune run object
        model: Trained model
        input_size: Size of input features
        n_features: Number of top features to visualize
    """
    if neptune_run is None:
        return
    
    # This is just an example - adapt to your model architecture
    if hasattr(model, 'feature_proj'):
        weight = model.feature_proj.weight.detach().cpu().numpy()
    else:
        print("Model doesn't have feature_proj layer - skipping feature importance")
        return
    
    # Calculate feature importance (L1 norm of weights)
    importance = np.abs(weight).sum(axis=0)
    
    # Get top N features
    top_indices = importance.argsort()[-n_features:][::-1]
    top_importance = importance[top_indices]
    
    # Plot feature importance
    plt.figure(figsize=(12, 6))
    plt.bar(range(n_features), top_importance)
    plt.xlabel('Feature index')
    plt.ylabel('Importance')
    plt.title('Top Feature Importance')
    plt.xticks(range(n_features), top_indices)
    plt.tight_layout()
    
    # Log to Neptune
    log_figure(neptune_run, plt.gcf(), "feature_importance")
    plt.close()


def log_tsne_embeddings(neptune_run, features, labels, perplexity=30, n_iter=1000):
    """
    Log t-SNE visualization of feature embeddings.
    
    Args:
        neptune_run: Neptune run object
        features: Feature embeddings
        labels: Class labels
        perplexity: t-SNE perplexity parameter
        n_iter: Number of iterations
    """
    if neptune_run is None:
        return
    
    # Perform t-SNE dimensionality reduction
    print("Computing t-SNE embeddings...")
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42)
    features_2d = tsne.fit_transform(features)
    
    # Plot t-SNE embeddings
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], 
                          c=labels, cmap='viridis', alpha=0.8, s=50)
    plt.colorbar(scatter, label='Class')
    plt.title('t-SNE visualization of feature embeddings')
    plt.xlabel('t-SNE dimension 1')
    plt.ylabel('t-SNE dimension 2')
    plt.tight_layout()
    
    # Log to Neptune
    log_figure(neptune_run, plt.gcf(), "tsne_embeddings")
    plt.close()


def log_precision_recall_curves(neptune_run, labels, probs, name="precision_recall_curve"):
    """
    Log precision-recall curves with average precision.
    
    Args:
        neptune_run: Neptune run object
        labels: True labels
        probs: Predicted probabilities
        name: Name for the plot in Neptune
    """
    if neptune_run is None:
        return
    
    # Calculate precision and recall
    precision, recall, thresholds = precision_recall_curve(labels, probs)
    avg_precision = np.mean(precision)
    
    # Calculate area under PR curve
    pr_auc = auc(recall, precision)
    
    # Plot precision-recall curve
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, lw=2, 
             label=f'PR curve (AP={avg_precision:.2f}, AUC={pr_auc:.2f})')
    plt.fill_between(recall, precision, alpha=0.2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='best')
    plt.grid(alpha=0.3)
    
    # Log to Neptune
    log_figure(neptune_run, plt.gcf(), name)
    
    # Also log numerical metrics
    neptune_run[f"evaluation/pr_auc"] = pr_auc
    neptune_run[f"evaluation/avg_precision"] = avg_precision
    
    plt.close()


def log_learning_curve(neptune_run, history, name="learning_curve"):
    """
    Log learning curve with train/validation metrics.
    
    Args:
        neptune_run: Neptune run object
        history: Training history dictionary
        name: Name for the plot in Neptune
    """
    if neptune_run is None:
        return
    
    metrics = []
    if 'train_loss' in history and 'val_loss' in history:
        metrics.append(('loss', 'train_loss', 'val_loss', 'Loss'))
    
    if 'train_acc' in history and 'val_acc' in history:
        metrics.append(('accuracy', 'train_acc', 'val_acc', 'Accuracy (%)'))
    
    if 'val_f1_macro' in history and 'val_f1_weighted' in history:
        metrics.append(('f1', 'val_f1_macro', 'val_f1_weighted', 'F1 Score'))
    
    n_metrics = len(metrics)
    if n_metrics == 0:
        return
    
    plt.figure(figsize=(12, 4 * n_metrics))
    
    for i, (metric_name, train_key, val_key, ylabel) in enumerate(metrics):
        plt.subplot(n_metrics, 1, i+1)
        
        if train_key in history:
            plt.plot(history[train_key], label=f'Train {ylabel}')
        
        if val_key in history:
            plt.plot(history[val_key], label=f'Validation {ylabel}')
        
        plt.xlabel('Epoch')
        plt.ylabel(ylabel)
        plt.title(f'Training and Validation {ylabel}')
        plt.legend()
        plt.grid(alpha=0.3)
    
    plt.tight_layout()
    
    # Log to Neptune
    log_figure(neptune_run, plt.gcf(), name)
    plt.close()


def log_patch_distribution(neptune_run, patch_counts, name="patch_distribution"):
    """
    Log histogram of patch counts.
    
    Args:
        neptune_run: Neptune run object
        patch_counts: List of patch counts per instance
        name: Name for the plot in Neptune
    """
    if neptune_run is None:
        return
    
    plt.figure(figsize=(10, 6))
    sns.histplot(patch_counts, bins=50, kde=True)
    
    # Add statistics
    mean = np.mean(patch_counts)
    median = np.median(patch_counts)
    min_val = np.min(patch_counts)
    max_val = np.max(patch_counts)
    
    plt.axvline(x=mean, color='red', linestyle='-', label=f'Mean: {mean:.1f}')
    plt.axvline(x=median, color='green', linestyle='--', label=f'Median: {median:.1f}')
    
    plt.xlabel('Number of Patches')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of Patch Counts (min={min_val}, max={max_val})')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Log to Neptune
    log_figure(neptune_run, plt.gcf(), name)
    plt.close()
    
    # Log statistics as metrics
    neptune_run["data/patch_count/mean"] = mean
    neptune_run["data/patch_count/median"] = median
    neptune_run["data/patch_count/min"] = min_val
    neptune_run["data/patch_count/max"] = max_val
    
    # Log percentiles
    percentiles = [25, 50, 75, 90, 95, 99]
    for p in percentiles:
        value = np.percentile(patch_counts, p)
        neptune_run[f"data/patch_count/percentile_{p}"] = value


def log_classification_threshold_analysis(neptune_run, labels, probs, name="threshold_analysis"):
    """
    Log threshold analysis for binary classification.
    
    Args:
        neptune_run: Neptune run object
        labels: True labels
        probs: Predicted probabilities
        name: Name for the plot in Neptune
    """
    if neptune_run is None:
        return
    
    from sklearn.metrics import precision_recall_fscore_support
    
    thresholds = np.linspace(0.1, 0.9, 9)
    precision = []
    recall = []
    f1 = []
    
    for threshold in thresholds:
        preds = (probs >= threshold).astype(int)
        p, r, f, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=0)
        precision.append(p)
        recall.append(r)
        f1.append(f)
    
    # Find optimal threshold for F1
    max_f1_idx = np.argmax(f1)
    optimal_threshold = thresholds[max_f1_idx]
    
    # Plot metrics vs threshold
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, precision, 'b-', label='Precision')
    plt.plot(thresholds, recall, 'g-', label='Recall')
    plt.plot(thresholds, f1, 'r-', label='F1 Score')
    plt.axvline(x=optimal_threshold, color='k', linestyle='--', 
                label=f'Optimal Threshold: {optimal_threshold:.2f}')
    
    plt.xlabel('Classification Threshold')
    plt.ylabel('Score')
    plt.title('Precision, Recall, and F1 Score vs Classification Threshold')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Log to Neptune
    log_figure(neptune_run, plt.gcf(), name)
    plt.close()
    
    # Log optimal threshold
    neptune_run["evaluation/optimal_threshold"] = optimal_threshold
    neptune_run["evaluation/optimal_threshold/f1"] = f1[max_f1_idx]
    neptune_run["evaluation/optimal_threshold/precision"] = precision[max_f1_idx]
    neptune_run["evaluation/optimal_threshold/recall"] = recall[max_f1_idx]


def log_model_graph(neptune_run, model, input_shape=(1, 300, 512)):
    """
    Log model graph and architecture summary.
    
    Args:
        neptune_run: Neptune run object
        model: PyTorch model
        input_shape: Input shape for the model
    """
    if neptune_run is None:
        return
    
    # Create a string representation of the model
    model_str = str(model)
    
    # Log model summary
    neptune_run["model/summary"] = model_str
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Log parameter counts
    neptune_run["model/parameters/total"] = total_params
    neptune_run["model/parameters/trainable"] = trainable_params
    neptune_run["model/parameters/non_trainable"] = total_params - trainable_params
    
    # Try to log model graph using torchviz if available
    try:
        from torchviz import make_dot
        
        # Create dummy input
        dummy_input = torch.zeros(input_shape, device=next(model.parameters()).device)
        
        # Generate graph
        output = model(dummy_input)
        dot = make_dot(output, params=dict(model.named_parameters()))
        
        # Save graph to file
        dot.format = 'png'
        dot.render('model_graph', cleanup=True)
        
        # Upload to Neptune
        neptune_run["model/graph"].upload('model_graph.png')
        
        # Clean up
        if os.path.exists('model_graph.png'):
            os.remove('model_graph.png')
            
    except ImportError:
        print("torchviz not available. Skipping model graph visualization.")
    except Exception as e:
        print(f"Error generating model graph: {e}")