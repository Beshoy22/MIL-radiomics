import os
import json
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from sklearn.manifold import TSNE
from io import BytesIO

def set_seed(seed=42):
    """
    Set seed for reproducibility.
    
    Args:
        seed (int): Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Random seed set to {seed}")


def save_model_and_results(model, metrics, history, output_dir):
    """
    Save model, metrics, and training history.
    
    Args:
        model (nn.Module): Trained model
        metrics (dict): Evaluation metrics
        history (dict): Training history
        output_dir (str): Directory to save results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model
    torch.save(model.state_dict(), os.path.join(output_dir, 'model.pt'))
    
    # Process metrics for each dataset for JSON saving
    metrics_json = {}
    
    # Handle the case of both the legacy format and the new comprehensive format
    if 'all_datasets' in metrics:
        # New format with multiple datasets
        for dataset_name, dataset_metrics in metrics['all_datasets'].items():
            dataset_metrics_json = {}
            for k, v in dataset_metrics.items():
                if k in ['confusion_matrix', 'all_labels', 'all_preds', 'all_probs']:
                    continue
                if isinstance(v, np.ndarray):
                    dataset_metrics_json[k] = v.tolist()
                elif isinstance(v, np.float64) or isinstance(v, np.float32):
                    dataset_metrics_json[k] = float(v)
                else:
                    dataset_metrics_json[k] = v
            metrics_json[dataset_name] = dataset_metrics_json
    
    # Also save the legacy top-level metrics for backward compatibility
    for k, v in metrics.items():
        if k in ['confusion_matrix', 'all_labels', 'all_preds', 'all_probs', 'all_datasets']:
            continue
        if isinstance(v, np.ndarray):
            metrics_json[k] = v.tolist()
        elif isinstance(v, np.float64) or isinstance(v, np.float32):
            metrics_json[k] = float(v)
        else:
            metrics_json[k] = v
    
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics_json, f, indent=2)
    
    # Save history - updated to include F1 metrics if available
    history_json = {
        'train_loss': [float(v) for v in history['train_loss']],
        'val_loss': [float(v) for v in history['val_loss']],
        'train_acc': [float(v) for v in history['train_acc']],
        'val_acc': [float(v) for v in history['val_acc']]
    }
    
    # Add F1 scores if available
    if 'val_f1_macro' in history:
        history_json['val_f1_macro'] = [float(v) for v in history['val_f1_macro']]
    if 'val_f1_weighted' in history:
        history_json['val_f1_weighted'] = [float(v) for v in history['val_f1_weighted']]
    
    with open(os.path.join(output_dir, 'history.json'), 'w') as f:
        json.dump(history_json, f, indent=2)
    
    print(f"Model and results saved to {output_dir}")


def load_model(model_class, model_path, model_config=None, device=None):
    """
    Load a trained model from disk.
    
    Args:
        model_class: Model class (e.g., MILTransformer)
        model_path (str): Path to the saved model
        model_config (dict): Configuration parameters for the model
        device (str): Device to load the model onto
        
    Returns:
        nn.Module: Loaded model
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if model_config is None:
        model_config = {}
    
    model = model_class(**model_config)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    return model


# Here are the fixed versions of the visualization functions that properly save files

def plot_training_curves(history, output_dir=None, neptune_run=None):
    """
    Plot training and validation curves.
    
    Args:
        history (dict): Training history
        output_dir (str, optional): Directory to save plots
        neptune_run: Neptune run object for logging (optional)
    """
    import matplotlib.pyplot as plt
    import os
    
    # Determine number of plots needed - we need a third plot if F1 metrics are available
    has_f1_metrics = 'val_f1_macro' in history
    num_plots = 3 if has_f1_metrics else 2
    
    plt.figure(figsize=(15, 5 if has_f1_metrics else 10))
    
    # Plot loss
    plt.subplot(1, num_plots, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot accuracy
    plt.subplot(1, num_plots, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Plot F1 metrics if available
    if has_f1_metrics:
        plt.subplot(1, num_plots, 3)
        plt.plot(history['val_f1_macro'], label='F1 Macro')
        plt.plot(history['val_f1_weighted'], label='F1 Weighted')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.title('Validation F1 Scores')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    fig = plt.gcf()
    
    # Save figure locally if output_dir is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, 'training_curves.png')
        fig.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to {output_file}")
    
    # Log figure to Neptune
    if neptune_run:
        try:
            from neptune_utils import log_figure
            log_figure(neptune_run, fig, "training_curves")
        except Exception as e:
            print(f"Warning: Failed to log figure to Neptune: {e}")
    
    plt.close(fig)  # Close the figure to avoid displaying when not needed


def plot_confusion_matrix(y_true, y_pred, output_dir=None, neptune_run=None):
    """
    Plot confusion matrix.
    
    Args:
        y_true (array): True labels
        y_pred (array): Predicted labels
        output_dir (str, optional): Directory to save plot
        neptune_run: Neptune run object for logging (optional)
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    fig = plt.gcf()
    
    # Save figure locally if output_dir is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, 'confusion_matrix.png')
        fig.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {output_file}")
    
    # Log figure to Neptune
    if neptune_run:
        try:
            from neptune_utils import log_figure
            log_figure(neptune_run, fig, "confusion_matrix")
        except Exception as e:
            print(f"Warning: Failed to log figure to Neptune: {e}")
    
    plt.close(fig)  # Close the figure to avoid displaying when not needed


def plot_roc_curve(labels, probs, output_dir=None, neptune_run=None):
    """
    Plot ROC curve.
    
    Args:
        labels (array): True labels
        probs (array): Predicted probabilities
        output_dir (str, optional): Directory to save plot
        neptune_run: Neptune run object for logging (optional)
    """
    import matplotlib.pyplot as plt
    import os
    from sklearn.metrics import roc_curve, auc
    
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    fig = plt.gcf()
    
    # Save figure locally if output_dir is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, 'roc_curve.png')
        fig.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"ROC curve saved to {output_file}")
    
    # Log figure to Neptune
    if neptune_run:
        try:
            from neptune_utils import log_figure
            log_figure(neptune_run, fig, "roc_curve")
        except Exception as e:
            print(f"Warning: Failed to log figure to Neptune: {e}")
    
    plt.close(fig)  # Close the figure to avoid displaying when not needed

def plot_pr_curve(labels, probs, output_dir=None, neptune_run=None):
    """
    Plot Precision-Recall curve.
    
    Args:
        labels (array): True labels
        probs (array): Predicted probabilities
        output_dir (str, optional): Directory to save plot
        neptune_run: Neptune run object for logging (optional)
    """
    precision, recall, _ = precision_recall_curve(labels, probs)
    avg_precision = np.mean(precision)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (avg precision = {avg_precision:.2f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')
    plt.grid(True)
    fig = plt.gcf()
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'pr_curve.png'))
    
    # Log figure to Neptune
    if neptune_run:
        try:
            from neptune_utils import log_figure
            log_figure(neptune_run, fig, "pr_curve")
        except ImportError:
            buffer = BytesIO()
            fig.savefig(buffer, format='png')
            buffer.seek(0)
            neptune_run["visualizations/pr_curve"].upload(buffer)
    
    plt.show()


def plot_confusion_matrix(y_true, y_pred, output_dir=None, neptune_run=None):
    """
    Plot confusion matrix.
    
    Args:
        y_true (array): True labels
        y_pred (array): Predicted labels
        output_dir (str, optional): Directory to save plot
        neptune_run: Neptune run object for logging (optional)
    """
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    fig = plt.gcf()
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    
    # Log figure to Neptune
    if neptune_run:
        try:
            from neptune_utils import log_figure
            log_figure(neptune_run, fig, "confusion_matrix")
        except ImportError:
            buffer = BytesIO()
            fig.savefig(buffer, format='png')
            buffer.seek(0)
            neptune_run["visualizations/confusion_matrix"].upload(buffer)
    
    plt.show()


def visualize_attention(model, dataloader, num_samples=5, output_dir=None, 
                        device='cuda' if torch.cuda.is_available() else 'cpu',
                        neptune_run=None):
    """
    Visualize attention weights for a few samples.
    
    Args:
        model (nn.Module): Trained model
        dataloader (DataLoader): DataLoader with data
        num_samples (int): Number of samples to visualize
        output_dir (str, optional): Directory to save the plots
        device (str): Device to use for inference
        neptune_run: Neptune run object for logging (optional)
    """
    model.eval()
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    samples_visualized = 0
    
    with torch.no_grad():
        for features, labels in dataloader:
            if samples_visualized >= num_samples:
                break
                
            features, labels = features.to(device), labels.to(device)
            
            # Forward pass with attention weights
            _, attn_weights = model(features, return_attn=True)
            
            # Convert to numpy for visualization
            features_np = features.cpu().numpy()
            labels_np = labels.cpu().numpy()
            attn_weights_np = attn_weights.cpu().numpy()
            
            # Visualize each sample in the batch
            for i in range(min(features.size(0), num_samples - samples_visualized)):
                # Get number of actual patches (non-zero)
                n_patches = features_np[i].shape[0]
                non_zero_patches = np.sum(np.any(features_np[i] != 0, axis=1))
                
                # Get attention weights for this sample
                sample_attn = attn_weights_np[i, 0, :non_zero_patches]
                
                # Normalize attention weights for visualization
                normalized_attn = (sample_attn - sample_attn.min()) / (sample_attn.max() - sample_attn.min() + 1e-8)
                
                # Plot attention weights
                plt.figure(figsize=(10, 4))
                plt.bar(range(non_zero_patches), normalized_attn)
                plt.title(f'Sample {samples_visualized+1}, Label: {labels_np[i]}')
                plt.xlabel('Patch index')
                plt.ylabel('Normalized attention weight')
                plt.tight_layout()
                fig = plt.gcf()
                
                if output_dir:
                    plt.savefig(os.path.join(output_dir, f'attention_sample_{samples_visualized+1}.png'))
                
                # Log figure to Neptune
                if neptune_run:
                    try:
                        from neptune_utils import log_figure
                        log_figure(neptune_run, fig, f"attention_sample_{samples_visualized+1}")
                    except ImportError:
                        buffer = BytesIO()
                        fig.savefig(buffer, format='png')
                        buffer.seek(0)
                        neptune_run[f"visualizations/attention_sample_{samples_visualized+1}"].upload(buffer)
                
                plt.show()
                samples_visualized += 1
                
                if samples_visualized >= num_samples:
                    break