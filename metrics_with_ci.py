import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix
)
from io import BytesIO


def predict(model, dataloader, return_attention=False, 
           device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Make predictions using a trained model.
    
    Args:
        model (nn.Module): Trained model
        dataloader (DataLoader): DataLoader with data
        return_attention (bool): Whether to return attention weights
        device (str): Device to use for inference
        
    Returns:
        tuple: (labels, predictions, probabilities, attention_weights) if return_attention=True
               (labels, predictions, probabilities) otherwise
    """
    model = model.to(device)
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []
    all_attns = []
    
    with torch.no_grad():
        for features, labels in tqdm(dataloader, desc="Predicting"):
            features, labels = features.to(device), labels.to(device)
            
            # Forward pass with or without attention weights
            if return_attention and hasattr(model, 'forward') and 'return_attn' in model.forward.__code__.co_varnames:
                outputs, attn_weights = model(features, return_attn=True)
                all_attns.extend(attn_weights.cpu().numpy())
            else:
                outputs = model(features)
            
            # Get predictions and probabilities
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of positive class
    
    if return_attention:
        return np.array(all_labels), np.array(all_preds), np.array(all_probs), all_attns
    else:
        return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def bootstrap_metric(labels, preds, probs, metric_fn, n_bootstrap=1000, confidence=0.95):
    """
    Calculate a metric with bootstrap confidence intervals.
    
    Args:
        labels (np.array): True labels
        preds (np.array): Predicted labels
        probs (np.array): Predicted probabilities
        metric_fn (callable): Function to calculate the metric
        n_bootstrap (int): Number of bootstrap samples
        confidence (float): Confidence level (0-1)
        
    Returns:
        tuple: (metric_value, confidence_interval)
    """
    n_samples = len(labels)
    
    # Calculate metric on full dataset
    metric_value = metric_fn(labels, preds, probs)
    
    # Calculate bootstrap confidence interval
    bootstrap_values = []
    
    for _ in range(n_bootstrap):
        # Sample with replacement
        indices = np.random.choice(n_samples, n_samples, replace=True)
        bootstrap_labels = labels[indices]
        bootstrap_preds = preds[indices]
        bootstrap_probs = probs[indices]
        
        # Calculate metric on bootstrap sample
        try:
            bootstrap_values.append(metric_fn(bootstrap_labels, bootstrap_preds, bootstrap_probs))
        except:
            # In case the bootstrap sample has only one class
            bootstrap_values.append(0.0)
    
    # Sort bootstrap values
    bootstrap_values = np.array(bootstrap_values)
    bootstrap_values = np.sort(bootstrap_values)
    
    # Calculate confidence interval
    alpha = 1.0 - confidence
    lower_idx = int(n_bootstrap * alpha / 2)
    upper_idx = int(n_bootstrap * (1.0 - alpha / 2))
    
    # Handle edge cases
    if lower_idx < 0:
        lower_idx = 0
    if upper_idx >= n_bootstrap:
        upper_idx = n_bootstrap - 1
    
    return metric_value, [float(bootstrap_values[lower_idx]), float(bootstrap_values[upper_idx])]


def evaluate_model_with_ci(model, dataloader, device='cuda', n_bootstrap=1000, confidence=0.95, neptune_run=None):
    """
    Evaluate model with confidence intervals for all metrics.
    
    Args:
        model (nn.Module): Trained model
        dataloader (DataLoader): Data loader with samples
        device (str): Device to use
        n_bootstrap (int): Number of bootstrap samples
        confidence (float): Confidence level (0-1)
        neptune_run: Neptune run object for logging (optional)
        
    Returns:
        dict: Evaluation metrics with confidence intervals
    """
    # Get predictions
    labels, preds, probs = predict(model, dataloader, device=device)
    
    # Calculate confusion matrix
    cm = confusion_matrix(labels, preds)
    
    # Define metric functions
    def accuracy_fn(y_true, y_pred, _):
        return accuracy_score(y_true, y_pred)
    
    def precision_fn(y_true, y_pred, _):
        return precision_score(y_true, y_pred, zero_division=0)
    
    def recall_fn(y_true, y_pred, _):
        return recall_score(y_true, y_pred, zero_division=0)
    
    def f1_fn(y_true, y_pred, _):
        return f1_score(y_true, y_pred, zero_division=0)
    
    def f1_macro_fn(y_true, y_pred, _):
        return f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    def f1_weighted_fn(y_true, y_pred, _):
        return f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    def auc_fn(y_true, _, y_prob):
        # Handle single-class edge case
        if len(np.unique(y_true)) < 2:
            return 0.5
        return roc_auc_score(y_true, y_prob)
    
    # Calculate metrics with confidence intervals
    print("Calculating metrics with confidence intervals...")
    accuracy, accuracy_ci = bootstrap_metric(labels, preds, probs, accuracy_fn, n_bootstrap, confidence)
    precision, precision_ci = bootstrap_metric(labels, preds, probs, precision_fn, n_bootstrap, confidence)
    recall, recall_ci = bootstrap_metric(labels, preds, probs, recall_fn, n_bootstrap, confidence)
    f1, f1_ci = bootstrap_metric(labels, preds, probs, f1_fn, n_bootstrap, confidence)
    f1_macro, f1_macro_ci = bootstrap_metric(labels, preds, probs, f1_macro_fn, n_bootstrap, confidence)
    f1_weighted, f1_weighted_ci = bootstrap_metric(labels, preds, probs, f1_weighted_fn, n_bootstrap, confidence)
    
    # AUC might fail if only one class is present in some bootstrap samples
    try:
        auc, auc_ci = bootstrap_metric(labels, preds, probs, auc_fn, n_bootstrap, confidence)
    except:
        auc, auc_ci = 0.5, [0.0, 1.0]
    
    # Print metrics with confidence intervals
    print("\n===== EVALUATION METRICS WITH CONFIDENCE INTERVALS =====")
    print(f"Accuracy: {accuracy:.4f} (95% CI: {accuracy_ci[0]:.4f}-{accuracy_ci[1]:.4f})")
    print(f"Precision: {precision:.4f} (95% CI: {precision_ci[0]:.4f}-{precision_ci[1]:.4f})")
    print(f"Recall: {recall:.4f} (95% CI: {recall_ci[0]:.4f}-{recall_ci[1]:.4f})")
    print(f"F1: {f1:.4f} (95% CI: {f1_ci[0]:.4f}-{f1_ci[1]:.4f})")
    print(f"F1 Macro: {f1_macro:.4f} (95% CI: {f1_macro_ci[0]:.4f}-{f1_macro_ci[1]:.4f})")
    print(f"F1 Weighted: {f1_weighted:.4f} (95% CI: {f1_weighted_ci[0]:.4f}-{f1_weighted_ci[1]:.4f})")
    print(f"AUC: {auc:.4f} (95% CI: {auc_ci[0]:.4f}-{auc_ci[1]:.4f})")
    print(f"Confusion Matrix:\n{cm}")
    
    # Log metrics with confidence intervals to Neptune
    if neptune_run:
        # Create dictionary with metric values and confidence intervals
        metrics_ci = {
            'accuracy': {'value': accuracy, 'ci_low': accuracy_ci[0], 'ci_high': accuracy_ci[1]},
            'precision': {'value': precision, 'ci_low': precision_ci[0], 'ci_high': precision_ci[1]},
            'recall': {'value': recall, 'ci_low': recall_ci[0], 'ci_high': recall_ci[1]},
            'f1': {'value': f1, 'ci_low': f1_ci[0], 'ci_high': f1_ci[1]},
            'f1_macro': {'value': f1_macro, 'ci_low': f1_macro_ci[0], 'ci_high': f1_macro_ci[1]},
            'f1_weighted': {'value': f1_weighted, 'ci_low': f1_weighted_ci[0], 'ci_high': f1_weighted_ci[1]},
            'auc': {'value': auc, 'ci_low': auc_ci[0], 'ci_high': auc_ci[1]}
        }
        
        # Log each metric with its confidence intervals using flattened structure
        for metric_name, metric_values in metrics_ci.items():
            neptune_run[f"evaluation/test_{metric_name}_value"] = metric_values['value']
            neptune_run[f"evaluation/test_{metric_name}_ci_low"] = metric_values['ci_low']
            neptune_run[f"evaluation/test_{metric_name}_ci_high"] = metric_values['ci_high']
            
        # Log confusion matrix
        try:
            from neptune_utils import log_confusion_matrix
            log_confusion_matrix(neptune_run, cm, name="test_confusion_matrix_with_ci")
        except (ImportError, AttributeError):
            # Fallback to logging confusion matrix as an array
            neptune_run["evaluation/test_confusion_matrix"] = cm.tolist()
    
    # Return dictionary with all metrics
    return {
        'accuracy': accuracy,
        'accuracy_ci': accuracy_ci,
        'precision': precision,
        'precision_ci': precision_ci,
        'recall': recall,
        'recall_ci': recall_ci,
        'f1': f1,
        'f1_ci': f1_ci,
        'f1_macro': f1_macro,
        'f1_macro_ci': f1_macro_ci,
        'f1_weighted': f1_weighted,
        'f1_weighted_ci': f1_weighted_ci,
        'auc': auc,
        'auc_ci': auc_ci,
        'confusion_matrix': cm.tolist(),
        'all_labels': labels,
        'all_preds': preds,
        'all_probs': probs
    }


def plot_metrics_with_ci(metrics, output_dir=None, neptune_run=None):
    """
    Plot metrics with confidence intervals.
    
    Args:
        metrics (dict): Metrics dictionary with confidence intervals
        output_dir (str, optional): Directory to save plot
        neptune_run: Neptune run object for logging (optional)
    """
    import matplotlib.pyplot as plt
    import os
    
    # Metrics to plot
    metric_keys = ['accuracy', 'precision', 'recall', 'f1', 'f1_macro', 'f1_weighted', 'auc']
    
    plt.figure(figsize=(12, 6))
    
    # Extract values and confidence intervals
    values = [metrics[key] for key in metric_keys]
    ci_low = [metrics[f'{key}_ci'][0] for key in metric_keys]
    ci_high = [metrics[f'{key}_ci'][1] for key in metric_keys]
    
    # Calculate yerr for error bars
    yerr_low = [values[i] - ci_low[i] for i in range(len(values))]
    yerr_high = [ci_high[i] - values[i] for i in range(len(values))]
    yerr = [yerr_low, yerr_high]
    
    # Create bar chart
    plt.bar(range(len(metric_keys)), values, color='skyblue', alpha=0.8)
    plt.errorbar(range(len(metric_keys)), values, yerr=yerr, fmt='none', color='navy', capsize=5)
    
    # Add labels and formatting
    plt.xticks(range(len(metric_keys)), [key.capitalize() for key in metric_keys])
    plt.xlabel('Metric')
    plt.ylabel('Value')
    plt.title('Evaluation Metrics with 95% Confidence Intervals')
    plt.grid(axis='y', alpha=0.3)
    plt.ylim(0, 1.05)
    
    # Add value labels on bars
    for i, v in enumerate(values):
        plt.text(i, v + 0.02, f"{v:.3f}", ha='center')
    
    plt.tight_layout()
    fig = plt.gcf()
    
    # Save figure locally if output_dir is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, 'metrics_with_ci.png')
        fig.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Metrics with CI plot saved to {output_file}")
    
    # Log figure to Neptune
    if neptune_run:
        try:
            from neptune_utils import log_figure
            log_figure(neptune_run, fig, "metrics_with_confidence_intervals")
        except Exception as e:
            print(f"Warning: Failed to log figure to Neptune: {e}")
    
    plt.close(fig)  # Close the figure to avoid displaying when not needed