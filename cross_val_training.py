import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from transformer_mil_model import create_model
from lstm_mil_model import create_lstm_model
from conv_mil_model import create_conv_model
from lightweight_conv_mil_model import create_lightweight_conv_model
from model_train import setup_training, train_model
from utils import save_model_and_results, plot_training_curves, plot_confusion_matrix, plot_roc_curve


def run_cross_validation(args, folds, max_patches, class_weights, device='cuda'):
    """
    Run k-fold cross-validation training and evaluation.
    
    Args:
        args: Command line arguments
        folds (list): List of fold datasets
        max_patches (int): Maximum number of patches
        class_weights (torch.Tensor): Class weights for loss function
        device (str): Device to use for training
        
    Returns:
        tuple: (best_model, fold_metrics, fold_histories)
    """
    n_folds = len(folds)
    fold_metrics = []
    fold_histories = []
    fold_models = []
    
    # For keeping track of best model
    best_f1_macro = -1  # Changed from f1_weighted to f1_macro
    best_model = None
    best_model_fold = -1
    
    # Run training for each fold
    for fold_idx in range(n_folds):
        print(f"\n{'='*20} FOLD {fold_idx+1}/{n_folds} {'='*20}")
        
        # Create dataloaders for this fold
        from cross_validation import create_fold_loaders
        train_loader, val_loader = create_fold_loaders(
            folds=folds,
            fold_idx=fold_idx,
            batch_size=args.batch_size,
            oversample_factor=args.oversample_factor,
            max_patches=max_patches,
            num_workers=args.num_workers
        )
        
        print(f"Fold {fold_idx+1} - Training samples: {len(train_loader.dataset)}, "
              f"Validation samples: {len(val_loader.dataset)}")
        
        # Create model based on model type
        if args.model_type == 'transformer':
            model = create_model(
                feature_dim=args.feature_dim,
                hidden_dim=args.hidden_dim,
                num_heads=args.num_heads,
                num_layers=args.num_layers,
                dropout=args.dropout,
                num_classes=len(class_weights),
                max_patches=max_patches,
                device=device
            )
        elif args.model_type == 'lstm':
            model = create_lstm_model(
                feature_dim=args.feature_dim,
                hidden_dim=args.hidden_dim,
                num_layers=args.num_layers,
                dropout=args.dropout,
                bidirectional=args.bidirectional,
                num_classes=len(class_weights),
                max_patches=max_patches,
                use_attention=args.use_attention,
                device=device
            )
        elif args.model_type == 'conv':
            model = create_conv_model(
                feature_dim=args.feature_dim,
                hidden_dim=args.hidden_dim,
                dropout=args.dropout,
                num_classes=len(class_weights),
                max_patches=max_patches,
                num_groups=args.num_groups,
                device=device
            )
        elif args.model_type == 'lightweight_conv':
            model = create_lightweight_conv_model(
                feature_dim=args.feature_dim,
                hidden_dim=args.hidden_dim,
                num_blocks=args.num_blocks,
                dropout=args.dropout,
                num_classes=len(class_weights),
                max_patches=max_patches,
                num_groups=args.num_groups,
                device=device
            )
        
        # Set up training components
        criterion, optimizer, scheduler = setup_training(
            model=model,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            class_weights=class_weights,
            device=device
        )
        
        # Train model with specified selection metric
        print(f"Training {args.model_type} model for fold {fold_idx+1} (using {args.selection_metric} for model selection)...")
        model, history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            num_epochs=args.num_epochs,
            early_stopping_patience=args.patience,
            device=device,
            selection_metric=args.selection_metric
        )
        
        # Evaluate model on validation fold
        print(f"Evaluating {args.model_type} model on fold {fold_idx+1}...")
        
        from metrics_with_ci import evaluate_model_with_ci
        metrics = evaluate_model_with_ci(
            model=model,
            dataloader=val_loader,
            device=device
        )
        
        # Store results
        fold_metrics.append(metrics)
        fold_histories.append(history)
        fold_models.append(model)
        
        # Save fold-specific results
        fold_output_dir = os.path.join(args.output_dir, f'fold_{fold_idx+1}')
        os.makedirs(fold_output_dir, exist_ok=True)
        
        save_model_and_results(
            model=model,
            metrics=metrics,
            history=history,
            output_dir=fold_output_dir
        )
        
        # Plot fold-specific results
        plot_training_curves(history, fold_output_dir)
        plot_confusion_matrix(metrics['all_labels'], metrics['all_preds'], fold_output_dir)
        plot_roc_curve(metrics['all_labels'], metrics['all_probs'], fold_output_dir)
        
        # Check if this is the best model so far
        # Changed from f1_weighted to f1_macro
        if metrics['f1_macro'] > best_f1_macro:
            best_f1_macro = metrics['f1_macro']
            best_model = model
            best_model_fold = fold_idx
    
    print(f"\nCross-validation complete. Best model from fold {best_model_fold+1} with F1-macro: {best_f1_macro:.4f}")
    
    # Save the best model separately
    torch.save(best_model.state_dict(), os.path.join(args.output_dir, 'best_model.pt'))
    
    # Aggregate and save fold metrics
    aggregate_and_save_cv_metrics(fold_metrics, args.output_dir)
    
    return best_model, fold_metrics, fold_histories


def aggregate_and_save_cv_metrics(fold_metrics, output_dir):
    """
    Aggregate metrics across folds and save results.
    
    Args:
        fold_metrics (list): List of metrics for each fold
        output_dir (str): Directory to save results
    """
    # Keys to aggregate
    metric_keys = ['accuracy', 'precision', 'recall', 'f1', 'f1_macro', 'f1_weighted', 'auc']
    
    # Aggregate metrics
    aggregated = {}
    for key in metric_keys:
        values = [metrics[key] for metrics in fold_metrics]
        ci_values = [metrics[f'{key}_ci'] for metrics in fold_metrics]
        
        aggregated[key] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'values': [float(v) for v in values],
            'ci_values': ci_values
        }
    
    # Save aggregated metrics
    with open(os.path.join(output_dir, 'cv_metrics.json'), 'w') as f:
        json.dump(aggregated, f, indent=2)
    
    # Create visualization of cross-validation results
    plot_cv_metrics(aggregated, output_dir)
    
    # Print summary
    print("\nCross-Validation Summary:")
    for key in metric_keys:
        mean = aggregated[key]['mean']
        std = aggregated[key]['std']
        print(f"  {key.capitalize()}: {mean:.4f} ± {std:.4f}")


def plot_cv_metrics(aggregated, output_dir):
    """
    Plot cross-validation metrics with error bars.
    
    Args:
        aggregated (dict): Aggregated metrics
        output_dir (str): Directory to save plot
    """
    metric_keys = ['accuracy', 'precision', 'recall', 'f1', 'f1_macro', 'f1_weighted', 'auc']
    
    plt.figure(figsize=(12, 6))
    
    # Extract means and stds for plotting
    means = [aggregated[key]['mean'] for key in metric_keys]
    stds = [aggregated[key]['std'] for key in metric_keys]
    
    # Create bar plot
    x = np.arange(len(metric_keys))
    width = 0.7
    
    plt.bar(x, means, width, yerr=stds, capsize=10, color='skyblue', edgecolor='navy', alpha=0.8)
    
    # Add labels and title
    plt.xlabel('Metric')
    plt.ylabel('Value')
    plt.title('Cross-Validation Results with Standard Deviation')
    plt.xticks(x, [key.capitalize() for key in metric_keys])
    plt.grid(axis='y', alpha=0.3)
    
    # Format y-axis to show percentages up to 100%
    plt.ylim(0, 1.05)
    
    # Add mean values on top of bars
    for i, mean in enumerate(means):
        plt.text(i, mean + 0.02, f"{mean:.3f}", ha='center')
    
    plt.tight_layout()
    
    # Save figure
    plt.savefig(os.path.join(output_dir, 'cv_metrics.png'))
    plt.close()