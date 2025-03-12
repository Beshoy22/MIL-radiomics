import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix
)
from io import BytesIO
from verbose_utils import logger
import time

def predict(model, dataloader, return_attention=False, 
           device='cuda' if torch.cuda.is_available() else 'cpu',
           verbose=False):
    """
    Make predictions using a trained model.
    
    Args:
        model (nn.Module): Trained model
        dataloader (DataLoader): DataLoader with data
        return_attention (bool): Whether to return attention weights
        device (str): Device to use for inference
        verbose (bool): Whether to print verbose debugging information
        
    Returns:
        tuple: (labels, predictions, probabilities, patient_ids, centers, attention_weights) if return_attention=True
               (labels, predictions, probabilities, patient_ids, centers) otherwise
    """
    model = model.to(device)
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []
    all_attns = []
    all_patient_ids = []
    all_centers = []
    
    if verbose:
        logger.log(f"Making predictions on {len(dataloader.dataset)} samples")
        logger.log(f"Model: {type(model).__name__}")
        logger.log(f"Return attention: {return_attention}")
    
    batch_times = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Predicting")):
            batch_start = time.time()
            
            # Handle different return formats from the dataloader
            if len(batch) == 3:  # New format with identifiers
                features, labels, identifiers = batch
                patient_ids = identifiers.get('patient_id', [f"unknown_{i}" for i in range(len(labels))])
                centers = identifiers.get('center', ['unknown'] * len(labels))
            else:  # Old format without identifiers
                features, labels = batch
                patient_ids = [f"unknown_{i}" for i in range(len(labels))]
                centers = ['unknown'] * len(labels)
            
            if verbose and batch_idx == 0:  # Only log the first batch
                logger.tensor_info("Input features", features)
                logger.tensor_info("Labels", labels)
                
            features, labels = features.to(device), labels.to(device)
            
            # Forward pass with or without attention weights
            if return_attention and hasattr(model, 'forward') and 'return_attn' in model.forward.__code__.co_varnames:
                outputs, attn_weights = model(features, return_attn=True)
                all_attns.extend(attn_weights.cpu().numpy())
                
                if verbose and batch_idx == 0:  # Only log the first batch
                    logger.tensor_info("Attention weights", attn_weights)
            else:
                outputs = model(features)
            
            if verbose and batch_idx == 0:  # Only log the first batch
                logger.tensor_info("Model outputs", outputs)
            
            # Get predictions and probabilities
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of positive class
            all_patient_ids.extend(patient_ids)
            all_centers.extend(centers)
            
            batch_end = time.time()
            batch_times.append(batch_end - batch_start)
    
    if verbose:
        avg_batch_time = sum(batch_times) / len(batch_times) if batch_times else 0
        logger.log(f"Prediction completed:")
        logger.log(f"  Average batch processing time: {avg_batch_time:.4f}s")
        logger.log(f"  Total samples processed: {len(all_labels)}")
        logger.log(f"  Unique centers: {len(set(all_centers))}")
        
        # Log class distribution
        labels_array = np.array(all_labels)
        class_counts = {label: np.sum(labels_array == label) for label in np.unique(labels_array)}
        logger.log(f"  Class distribution: {class_counts}")
    
    if return_attention:
        return np.array(all_labels), np.array(all_preds), np.array(all_probs), all_patient_ids, all_centers, all_attns
    else:
        return np.array(all_labels), np.array(all_preds), np.array(all_probs), all_patient_ids, all_centers


def bootstrap_metric(labels, preds, probs, metric_fn, n_bootstrap=1000, confidence=0.95, verbose=False):
    """
    Calculate a metric with bootstrap confidence intervals.
    
    Args:
        labels (np.array): True labels
        preds (np.array): Predicted labels
        probs (np.array): Predicted probabilities
        metric_fn (callable): Function to calculate the metric
        n_bootstrap (int): Number of bootstrap samples
        confidence (float): Confidence level (0-1)
        verbose (bool): Whether to print verbose debugging information
        
    Returns:
        tuple: (metric_value, confidence_interval)
    """
    n_samples = len(labels)
    
    if verbose:
        logger.log(f"Bootstrap metric calculation:")
        logger.log(f"  Samples: {n_samples}")
        logger.log(f"  Bootstrap iterations: {n_bootstrap}")
        logger.log(f"  Confidence level: {confidence}")
        logger.log(f"  Metric function: {metric_fn.__name__}")
    
    # Calculate metric on full dataset
    metric_value = metric_fn(labels, preds, probs)
    
    if verbose:
        logger.log(f"  Full dataset metric value: {metric_value:.4f}")
    
    # Calculate bootstrap confidence interval
    bootstrap_values = []
    
    bootstrap_progress = tqdm(range(n_bootstrap), desc=f"Bootstrap {metric_fn.__name__}") if verbose else range(n_bootstrap)
    
    for _ in bootstrap_progress:
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
    
    ci = [float(bootstrap_values[lower_idx]), float(bootstrap_values[upper_idx])]
    
    if verbose:
        logger.log(f"  Confidence interval: [{ci[0]:.4f}, {ci[1]:.4f}]")
        logger.log(f"  Bootstrap mean: {np.mean(bootstrap_values):.4f}")
        logger.log(f"  Bootstrap std: {np.std(bootstrap_values):.4f}")
    
    return metric_value, ci


def evaluate_model_with_ci(model, dataloader, device='cuda', n_bootstrap=1000, confidence=0.95, 
                          neptune_run=None, verbose=False, dataset_name='test'):
    """
    Evaluate model with confidence intervals for all metrics.
    
    Args:
        model (nn.Module): Trained model
        dataloader (DataLoader): Data loader with samples
        device (str): Device to use
        n_bootstrap (int): Number of bootstrap samples
        confidence (float): Confidence level (0-1)
        neptune_run: Neptune run object for logging (optional)
        verbose (bool): Whether to print verbose debugging information
        dataset_name (str): Name of the dataset being evaluated (default: 'test')
        
    Returns:
        dict: Evaluation metrics with confidence intervals
    """
    if verbose:
        logger.log("Starting model evaluation with confidence intervals")
        logger.log(f"  Bootstrap samples: {n_bootstrap}")
        logger.log(f"  Confidence level: {confidence}")
    
    # Get predictions
    with logger.section("Making predictions"):
        labels, preds, probs, patient_ids, centers = predict(
            model, dataloader, device=device, verbose=verbose
        )
    
    # Calculate confusion matrix
    cm = confusion_matrix(labels, preds)
    
    if verbose:
        logger.log("Confusion matrix:")
        logger.log(f"{cm}")
    
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
    if verbose:
        logger.log("Calculating metrics with confidence intervals...")
    
    with logger.section("Calculating accuracy"):
        accuracy, accuracy_ci = bootstrap_metric(labels, preds, probs, accuracy_fn, n_bootstrap, confidence, verbose)
    
    with logger.section("Calculating precision"):
        precision, precision_ci = bootstrap_metric(labels, preds, probs, precision_fn, n_bootstrap, confidence, verbose)
    
    with logger.section("Calculating recall"):
        recall, recall_ci = bootstrap_metric(labels, preds, probs, recall_fn, n_bootstrap, confidence, verbose)
    
    with logger.section("Calculating F1"):
        f1, f1_ci = bootstrap_metric(labels, preds, probs, f1_fn, n_bootstrap, confidence, verbose)
    
    with logger.section("Calculating F1 Macro"):
        f1_macro, f1_macro_ci = bootstrap_metric(labels, preds, probs, f1_macro_fn, n_bootstrap, confidence, verbose)
    
    with logger.section("Calculating F1 Weighted"):
        f1_weighted, f1_weighted_ci = bootstrap_metric(labels, preds, probs, f1_weighted_fn, n_bootstrap, confidence, verbose)
    
    # AUC might fail if only one class is present in some bootstrap samples
    try:
        with logger.section("Calculating AUC"):
            auc, auc_ci = bootstrap_metric(labels, preds, probs, auc_fn, n_bootstrap, confidence, verbose)
    except Exception as e:
        if verbose:
            logger.log(f"Error calculating AUC: {e}")
        auc, auc_ci = 0.5, [0.0, 1.0]
    
    # Print metrics with confidence intervals
    print(f"\n===== EVALUATION METRICS WITH CONFIDENCE INTERVALS ({dataset_name.upper()} SET) =====")
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
            neptune_run[f"evaluation/{dataset_name}_{metric_name}_value"] = metric_values['value']
            neptune_run[f"evaluation/{dataset_name}_{metric_name}_ci_low"] = metric_values['ci_low']
            neptune_run[f"evaluation/{dataset_name}_{metric_name}_ci_high"] = metric_values['ci_high']
            
        # Log confusion matrix
        try:
            from neptune_utils import log_confusion_matrix
            log_confusion_matrix(neptune_run, cm, name=f"{dataset_name}_confusion_matrix_with_ci")
        except (ImportError, AttributeError):
            # Fallback to logging confusion matrix as an array
            neptune_run[f"evaluation/{dataset_name}_confusion_matrix"] = cm.tolist()
    
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
        'all_probs': probs,
        'patient_ids': patient_ids,
        'centers': centers,
        'dataset_name': dataset_name  # Add dataset name to metrics
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
    
    # Add dataset name to title if available
    dataset_name = metrics.get('dataset_name', 'test')
    plt.title(f'Evaluation Metrics with 95% Confidence Intervals ({dataset_name.upper()} SET)')
    
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