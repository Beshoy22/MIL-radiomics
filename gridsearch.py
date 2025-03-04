import os
import json
import argparse
import itertools
import time
import heapq
from collections import defaultdict
from datetime import datetime

import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from dataloader import prepare_dataloaders
from transformer_mil_model import create_model
from lstm_mil_model import create_lstm_model
from conv_mil_model import create_conv_model
from lightweight_conv_mil_model import create_lightweight_conv_model
from model_train import setup_training, train_model, evaluate_model
from utils import set_seed, save_model_and_results, plot_training_curves, plot_comparison_metrics
from metrics_with_ci import evaluate_model_with_ci


def get_parameter_grid():
    """
    Define the parameter grid for each model type.
    
    Returns:
        dict: Parameter grid for each model type
    """
    # Common parameters across models
    common_params = {
        'feature_dim': [512],  # Fixed based on input data
        'hidden_dim': [64, 128, 256],
        'dropout': [0.2, 0.3, 0.4],
        'lr': [1e-4, 5e-4, 1e-3],
        'weight_decay': [1e-5, 1e-4, 1e-3],
        'batch_size': [16, 32, 64]
    }
    
    # Model-specific parameters
    transformer_params = {
        **common_params,
        'num_heads': [2, 4, 8],
        'num_layers': [1, 2, 3]
    }
    
    lstm_params = {
        **common_params,
        'num_layers': [1, 2, 3],
        'bidirectional': [True, False],
        'use_attention': [True, False]
    }
    
    conv_params = {
        **common_params,
        'num_groups': [5, 10, 15]
    }
    
    lightweight_conv_params = {
        **common_params,
        'num_groups': [5, 10, 15],
        'num_blocks': [1, 2, 3]
    }
    
    # Final parameter grid
    return {
        'transformer': transformer_params,
        'lstm': lstm_params,
        'conv': conv_params,
        'lightweight_conv': lightweight_conv_params
    }


def generate_parameter_combinations(params_grid, max_combinations=None):
    """
    Generate all parameter combinations for a model type, optionally limited to a maximum number.
    
    Args:
        params_grid (dict): Parameter grid for a model type
        max_combinations (int, optional): Maximum number of combinations to generate
        
    Returns:
        list: List of parameter combinations
    """
    # Get all parameter names and values
    param_names = list(params_grid.keys())
    param_values = list(params_grid.values())
    
    # Calculate total combinations
    total_combinations = 1
    for values in param_values:
        total_combinations *= len(values)
    
    print(f"Total number of possible combinations: {total_combinations}")
    
    # If max_combinations is specified and less than total, sample randomly
    if max_combinations and max_combinations < total_combinations:
        print(f"Limiting to {max_combinations} random combinations")
        np.random.seed(42)  # For reproducibility
        
        # Generate combinations
        combinations = []
        for _ in range(max_combinations):
            combo = {}
            for name, values in params_grid.items():
                combo[name] = np.random.choice(values)
            combinations.append(combo)
        
        return combinations
    else:
        # Generate all combinations
        combinations = []
        for values in itertools.product(*param_values):
            combo = {name: value for name, value in zip(param_names, values)}
            combinations.append(combo)
        
        return combinations


def create_model_instance(model_type, params, max_patches, num_classes, device):
    """
    Create a model instance based on model type and parameters.
    
    Args:
        model_type (str): Type of model ('transformer', 'lstm', 'conv', 'lightweight_conv')
        params (dict): Model parameters
        max_patches (int): Maximum number of patches
        num_classes (int): Number of output classes
        device (torch.device): Device to place the model on
        
    Returns:
        nn.Module: Initialized model
    """
    if model_type == 'transformer':
        model = create_model(
            feature_dim=params['feature_dim'],
            hidden_dim=params['hidden_dim'],
            num_heads=params['num_heads'],
            num_layers=params['num_layers'],
            dropout=params['dropout'],
            num_classes=num_classes,
            max_patches=max_patches,
            device=device
        )
    elif model_type == 'lstm':
        model = create_lstm_model(
            feature_dim=params['feature_dim'],
            hidden_dim=params['hidden_dim'],
            num_layers=params['num_layers'],
            dropout=params['dropout'],
            bidirectional=params['bidirectional'],
            num_classes=num_classes,
            max_patches=max_patches,
            use_attention=params['use_attention'],
            device=device
        )
    elif model_type == 'conv':
        model = create_conv_model(
            feature_dim=params['feature_dim'],
            hidden_dim=params['hidden_dim'],
            dropout=params['dropout'],
            num_classes=num_classes,
            max_patches=max_patches,
            num_groups=params['num_groups'],
            device=device
        )
    elif model_type == 'lightweight_conv':
        model = create_lightweight_conv_model(
            feature_dim=params['feature_dim'],
            hidden_dim=params['hidden_dim'],
            num_blocks=params['num_blocks'],
            dropout=params['dropout'],
            num_classes=num_classes,
            max_patches=max_patches,
            num_groups=params['num_groups'],
            device=device
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    return model


def train_and_evaluate(model_type, params, train_loader, val_loader, test_loader, 
                      num_classes, max_patches, device, 
                      num_epochs=50, patience=5, result_dir=None):
    """
    Train and evaluate a model with given parameters.
    
    Args:
        model_type (str): Type of model
        params (dict): Model and training parameters
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        test_loader (DataLoader): Test data loader
        num_classes (int): Number of output classes
        max_patches (int): Maximum number of patches
        device (torch.device): Device for training
        num_epochs (int): Maximum number of epochs
        patience (int): Patience for early stopping
        result_dir (str, optional): Directory to save results
        
    Returns:
        tuple: (metrics, history, model)
    """
    try:
        # Create model
        model = create_model_instance(
            model_type=model_type,
            params=params,
            max_patches=max_patches,
            num_classes=num_classes,
            device=device
        )
        
        # Calculate trainable parameters
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Set up training components
        criterion, optimizer, scheduler = setup_training(
            model=model,
            learning_rate=params['lr'],
            weight_decay=params['weight_decay'],
            device=device
        )
        
        # Train model with timing
        start_time = time.time()
        model, history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            num_epochs=num_epochs,
            early_stopping_patience=patience,
            device=device
        )
        training_time = time.time() - start_time
        
        # Evaluate model on all datasets
        metrics = evaluate_model(
            model=model,
            test_loader=test_loader,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            device=device
        )
        
        # Add additional information to metrics
        metrics['num_params'] = num_params
        metrics['training_time'] = training_time
        metrics['early_stopping_epoch'] = len(history['train_loss'])
        
        # Save best model if a result directory is provided
        if result_dir:
            config_str = f"{model_type}"
            for key, value in sorted(params.items()):
                if key in ['feature_dim', 'batch_size']:  # Skip some common parameters
                    continue
                config_str += f"_{key}-{value}"
            
            # Truncate if too long
            if len(config_str) > 100:
                config_str = config_str[:100]
            
            model_dir = os.path.join(result_dir, config_str)
            os.makedirs(model_dir, exist_ok=True)
            
            # Save model and results
            save_model_and_results(
                model=model,
                metrics=metrics,
                history=history,
                output_dir=model_dir
            )
            
            # Plot training curves
            plot_training_curves(history, model_dir)
        
        return metrics, history, model
    
    except Exception as e:
        print(f"Error training model: {e}")
        return None, None, None


def run_grid_search(args):
    """
    Run grid search over all model types and parameter combinations.
    
    Args:
        args: Command line arguments
        
    Returns:
        tuple: (results_df, best_models)
    """
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"Using device: {device}")
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(args.output_dir, f"gridsearch_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    
    # Prepare data once for all models
    print("Preparing data loaders...")
    train_loader, val_loader, test_loader, class_weights, metrics, max_patches = prepare_dataloaders(
        data_dir=args.data_dir,
        endpoint=args.endpoint,
        batch_size=args.batch_size,  # Default batch size, will be overridden based on model params
        oversample_factor=args.oversample_factor,
        val_size=args.val_size,
        test_size=args.test_size,
        num_workers=args.num_workers,
        seed=args.seed,
        use_cache=args.use_cache,
        cache_dir=args.cache_dir
    )
    
    # Get parameter grid
    param_grid = get_parameter_grid()
    
    # Store all results
    all_results = []
    
    # Priority queue to keep track of top models based on validation F1 macro
    best_models_heap = []
    best_models_info = {}  # Store full information about best models
    
    # Iterate through each model type
    for model_type in args.model_types:
        print(f"\n{'='*20} Grid search for {model_type} {'='*20}")
        
        # Get parameter combinations for this model type
        model_params = param_grid[model_type]
        combinations = generate_parameter_combinations(model_params, args.max_combinations_per_model)
        
        print(f"Running grid search with {len(combinations)} parameter combinations for {model_type}")
        
        # Create temporary dataloaders dictionary to reuse for different batch sizes
        dataloader_cache = {}
        
        # Iterate through parameter combinations
        for i, params in enumerate(tqdm(combinations, desc=f"Searching {model_type} params")):
            print(f"\nCombination {i+1}/{len(combinations)}: {model_type}")
            for k, v in params.items():
                print(f"  {k}: {v}")
            
            # Check if we need to create new dataloaders for this batch size
            batch_size = params['batch_size']
            if batch_size not in dataloader_cache:
                print(f"Creating data loaders for batch size {batch_size}...")
                train_loader_bs, val_loader_bs, test_loader_bs, _, _, _ = prepare_dataloaders(
                    data_dir=args.data_dir,
                    endpoint=args.endpoint,
                    batch_size=batch_size,
                    oversample_factor=args.oversample_factor,
                    val_size=args.val_size,
                    test_size=args.test_size,
                    num_workers=args.num_workers,
                    seed=args.seed,
                    use_cache=args.use_cache,
                    cache_dir=args.cache_dir
                )
                dataloader_cache[batch_size] = (train_loader_bs, val_loader_bs, test_loader_bs)
            else:
                train_loader_bs, val_loader_bs, test_loader_bs = dataloader_cache[batch_size]
            
            # Train and evaluate model
            metrics, history, model = train_and_evaluate(
                model_type=model_type,
                params=params,
                train_loader=train_loader_bs,
                val_loader=val_loader_bs,
                test_loader=test_loader_bs,
                num_classes=len(class_weights),
                max_patches=max_patches,
                device=device,
                num_epochs=args.num_epochs,
                patience=args.patience,
                result_dir=results_dir if args.save_all_models else None
            )
            
            # If training was successful
            if metrics is not None:
                # Create a result entry
                result = {
                    'model_type': model_type,
                    **params,
                    'val_f1_macro': metrics['all_datasets']['val']['f1_macro'],
                    'val_f1_weighted': metrics['all_datasets']['val']['f1_weighted'],
                    'val_accuracy': metrics['all_datasets']['val']['accuracy'],
                    'val_auc': metrics['all_datasets']['val']['auc'],
                    'test_f1_macro': metrics['all_datasets']['test']['f1_macro'],
                    'test_f1_weighted': metrics['all_datasets']['test']['f1_weighted'],
                    'test_accuracy': metrics['all_datasets']['test']['accuracy'],
                    'test_auc': metrics['all_datasets']['test']['auc'],
                    'train_f1_macro': metrics['all_datasets']['train']['f1_macro'],
                    'train_f1_weighted': metrics['all_datasets']['train']['f1_weighted'],
                    'train_accuracy': metrics['all_datasets']['train']['accuracy'],
                    'train_auc': metrics['all_datasets']['train']['auc'],
                    'num_params': metrics['num_params'],
                    'training_time': metrics['training_time'],
                    'epochs': metrics['early_stopping_epoch']
                }
                
                all_results.append(result)
                
                # Add to best models heap (negative val_f1_macro for max-heap)
                val_f1_macro = metrics['all_datasets']['val']['f1_macro']
                
                # Create a unique identifier for this model
                model_id = f"{model_type}_"
                for k, v in sorted(params.items()):
                    model_id += f"{k}-{v}_"
                model_id = model_id[:-1]  # Remove the last underscore
                
                # If heap is not full or this model is better than the worst in the heap
                if len(best_models_heap) < args.top_k or -val_f1_macro < best_models_heap[0][0]:
                    # If heap is full, remove the worst model
                    if len(best_models_heap) == args.top_k:
                        heapq.heappop(best_models_heap)
                    
                    # Add this model to the heap
                    heapq.heappush(best_models_heap, (-val_f1_macro, model_id))
                    
                    # Save model info
                    best_models_info[model_id] = {
                        'model': model,
                        'metrics': metrics,
                        'history': history,
                        'params': params,
                        'model_type': model_type
                    }
                    
                    print(f"New top-{args.top_k} model: {model_type} with val_f1_macro = {val_f1_macro:.4f}")
                
                # Save results after each model
                df = pd.DataFrame(all_results)
                df.to_csv(os.path.join(results_dir, 'all_results.csv'), index=False)
    
    # Convert results to DataFrame for analysis
    results_df = pd.DataFrame(all_results)
    
    # Sort by validation F1 macro score
    results_df = results_df.sort_values('val_f1_macro', ascending=False)
    
    # Save all results
    results_df.to_csv(os.path.join(results_dir, 'all_results.csv'), index=False)
    
    # Extract top models
    top_models = []
    for _, model_id in sorted(best_models_heap):
        model_info = best_models_info[model_id]
        top_models.append({
            'model_id': model_id,
            'model': model_info['model'],
            'metrics': model_info['metrics'],
            'history': model_info['history'],
            'params': model_info['params'],
            'model_type': model_info['model_type']
        })
    
    # Save top models details
    with open(os.path.join(results_dir, 'top_models_details.json'), 'w') as f:
        top_models_details = []
        for model_info in top_models:
            metrics = model_info['metrics']
            params = model_info['params']
            
            # Extract relevant metrics
            details = {
                'model_id': model_info['model_id'],
                'model_type': model_info['model_type'],
                'params': params,
                'val_f1_macro': metrics['all_datasets']['val']['f1_macro'],
                'val_f1_weighted': metrics['all_datasets']['val']['f1_weighted'],
                'val_accuracy': metrics['all_datasets']['val']['accuracy'],
                'val_auc': metrics['all_datasets']['val']['auc'],
                'test_f1_macro': metrics['all_datasets']['test']['f1_macro'],
                'test_f1_weighted': metrics['all_datasets']['test']['f1_weighted'],
                'test_accuracy': metrics['all_datasets']['test']['accuracy'],
                'test_auc': metrics['all_datasets']['test']['auc'],
                'train_f1_macro': metrics['all_datasets']['train']['f1_macro'],
                'train_f1_weighted': metrics['all_datasets']['train']['f1_weighted'],
                'train_accuracy': metrics['all_datasets']['train']['accuracy'],
                'train_auc': metrics['all_datasets']['train']['auc'],
                'num_params': metrics['num_params'],
                'training_time': metrics['training_time'],
                'epochs': metrics['early_stopping_epoch']
            }
            
            top_models_details.append(details)
        
        json.dump(top_models_details, f, indent=2)
    
    # Analyze overall grid search results
    analyze_grid_search_results(results_df, results_dir)
    
    # Save each top model separately
    top_models_dir = os.path.join(results_dir, 'top_models')
    os.makedirs(top_models_dir, exist_ok=True)
    
    for i, model_info in enumerate(top_models):
        rank = len(top_models) - i  # Reverse the order for rank 1 to be the best
        model_dir = os.path.join(top_models_dir, f"rank_{rank}_{model_info['model_type']}")
        
        save_model_and_results(
            model=model_info['model'],
            metrics=model_info['metrics'],
            history=model_info['history'],
            output_dir=model_dir
        )
        
        # Plot training curves
        plot_training_curves(model_info['history'], model_dir)
    
    # Create performance comparison plots for top models
    plot_top_models_comparison(top_models, results_dir)
    
    return results_df, top_models


def plot_top_models_comparison(top_models, output_dir):
    """
    Create comparison plots for top models.
    
    Args:
        top_models (list): List of top model information
        output_dir (str): Directory to save plots
    """
    # Extract data for plotting
    model_names = []
    val_f1_macro = []
    val_f1_weighted = []
    val_accuracy = []
    val_auc = []
    test_f1_macro = []
    test_f1_weighted = []
    test_accuracy = []
    test_auc = []
    train_f1_macro = []
    train_f1_weighted = []
    train_accuracy = []
    train_auc = []
    num_params = []
    
    for i, model_info in enumerate(top_models):
        rank = len(top_models) - i  # Reverse for rank 1 to be best
        metrics = model_info['metrics']
        model_type = model_info['model_type']
        
        # Create a shortened name
        short_name = f"#{rank} {model_type}"
        model_names.append(short_name)
        
        # Extract metrics
        val_f1_macro.append(metrics['all_datasets']['val']['f1_macro'])
        val_f1_weighted.append(metrics['all_datasets']['val']['f1_weighted'])
        val_accuracy.append(metrics['all_datasets']['val']['accuracy'])
        val_auc.append(metrics['all_datasets']['val']['auc'])
        
        test_f1_macro.append(metrics['all_datasets']['test']['f1_macro'])
        test_f1_weighted.append(metrics['all_datasets']['test']['f1_weighted'])
        test_accuracy.append(metrics['all_datasets']['test']['accuracy'])
        test_auc.append(metrics['all_datasets']['test']['auc'])
        
        train_f1_macro.append(metrics['all_datasets']['train']['f1_macro'])
        train_f1_weighted.append(metrics['all_datasets']['train']['f1_weighted'])
        train_accuracy.append(metrics['all_datasets']['train']['accuracy'])
        train_auc.append(metrics['all_datasets']['train']['auc'])
        
        num_params.append(metrics['num_params'])
    
    # Flip the order for plotting (best model first)
    model_names.reverse()
    val_f1_macro.reverse()
    val_f1_weighted.reverse()
    val_accuracy.reverse()
    val_auc.reverse()
    test_f1_macro.reverse()
    test_f1_weighted.reverse()
    test_accuracy.reverse()
    test_auc.reverse()
    train_f1_macro.reverse()
    train_f1_weighted.reverse()
    train_accuracy.reverse()
    train_auc.reverse()
    num_params.reverse()
    
    # Create performance comparison plot
    plt.figure(figsize=(15, 10))
    
    # Set up the plot
    x = np.arange(len(model_names))
    width = 0.2
    
    # Plot F1 Macro
    plt.subplot(2, 2, 1)
    plt.bar(x - width, train_f1_macro, width, label='Train', color='skyblue')
    plt.bar(x, val_f1_macro, width, label='Validation', color='orange')
    plt.bar(x + width, test_f1_macro, width, label='Test', color='green')
    plt.xlabel('Model')
    plt.ylabel('F1 Macro')
    plt.title('F1 Macro Score Comparison')
    plt.xticks(x, model_names, rotation=45, ha='right')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Plot F1 Weighted
    plt.subplot(2, 2, 2)
    plt.bar(x - width, train_f1_weighted, width, label='Train', color='skyblue')
    plt.bar(x, val_f1_weighted, width, label='Validation', color='orange')
    plt.bar(x + width, test_f1_weighted, width, label='Test', color='green')
    plt.xlabel('Model')
    plt.ylabel('F1 Weighted')
    plt.title('F1 Weighted Score Comparison')
    plt.xticks(x, model_names, rotation=45, ha='right')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Plot Accuracy
    plt.subplot(2, 2, 3)
    plt.bar(x - width, train_accuracy, width, label='Train', color='skyblue')
    plt.bar(x, val_accuracy, width, label='Validation', color='orange')
    plt.bar(x + width, test_accuracy, width, label='Test', color='green')
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Comparison')
    plt.xticks(x, model_names, rotation=45, ha='right')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Plot AUC
    plt.subplot(2, 2, 4)
    plt.bar(x - width, train_auc, width, label='Train', color='skyblue')
    plt.bar(x, val_auc, width, label='Validation', color='orange')
    plt.bar(x + width, test_auc, width, label='Test', color='green')
    plt.xlabel('Model')
    plt.ylabel('AUC')
    plt.title('AUC Comparison')
    plt.xticks(x, model_names, rotation=45, ha='right')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(os.path.join(output_dir, 'top_models_comparison.png'), bbox_inches='tight')
    plt.close()
    
    # Create a detailed table for top models
    fig, ax = plt.subplots(figsize=(15, len(model_names) * 0.8 + 2))
    ax.axis('off')
    
    table_data = []
    for i, name in enumerate(model_names):
        table_data.append([
            name,
            f"{val_f1_macro[i]:.4f}",
            f"{val_f1_weighted[i]:.4f}",
            f"{val_accuracy[i]:.4f}",
            f"{val_auc[i]:.4f}",
            f"{test_f1_macro[i]:.4f}",
            f"{test_f1_weighted[i]:.4f}",
            f"{test_accuracy[i]:.4f}",
            f"{test_auc[i]:.4f}",
            f"{num_params[i]:,}"
        ])
    
    column_headers = [
        "Model",
        "Val F1 Macro",
        "Val F1 Weighted",
        "Val Accuracy",
        "Val AUC",
        "Test F1 Macro",
        "Test F1 Weighted",
        "Test Accuracy",
        "Test AUC",
        "Params"
    ]
    
    table = ax.table(
        cellText=table_data,
        colLabels=column_headers,
        loc='center',
        cellLoc='center',
        colColours=['#f0f0f0'] * len(column_headers)
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    plt.title('Top Models Detailed Comparison', fontsize=16, pad=20)
    plt.tight_layout()
    
    # Save the table
    plt.savefig(os.path.join(output_dir, 'top_models_table.png'), bbox_inches='tight')
    plt.close()
    
    # Create parameter analysis plots
    for model_info in top_models:
        model_type = model_info['model_type']
        # Only create these plots if we have enough models of this type
        models_of_type = [m for m in top_models if m['model_type'] == model_type]
        if len(models_of_type) > 1:
            create_parameter_analysis_plots(models_of_type, model_type, output_dir)


def create_parameter_analysis_plots(models, model_type, output_dir):
    """
    Create plots to analyze the effect of different parameters for a specific model type.
    
    Args:
        models (list): List of models of the same type
        model_type (str): Model type
        output_dir (str): Directory to save plots
    """
    # Extract parameters and performance metrics
    params_effect = defaultdict(list)
    
    for model_info in models:
        params = model_info['params']
        val_f1_macro = model_info['metrics']['all_datasets']['val']['f1_macro']
        
        for param, value in params.items():
            if param in ['feature_dim', 'batch_size']:  # Skip some common parameters
                continue
            
            params_effect[param].append((value, val_f1_macro))
    
    # Create plots for each parameter
    for param, values in params_effect.items():
        if len(set([v[0] for v in values])) > 1:  # Only plot if we have different values
            plt.figure(figsize=(10, 6))
            
            # Sort by parameter value
            values.sort(key=lambda x: x[0])
            
            # Extract parameter values and F1 scores
            param_values = [v[0] for v in values]
            f1_scores = [v[1] for v in values]
            
            # For boolean parameters
            if all(isinstance(v, bool) for v in param_values):
                plt.bar([str(v) for v in param_values], f1_scores, color='skyblue')
            else:
                # Try to convert to numeric for line plot
                try:
                    x_values = [float(v) for v in param_values]
                    plt.plot(x_values, f1_scores, marker='o', linestyle='-', color='royalblue')
                except (ValueError, TypeError):
                    # For non-numeric parameters
                    plt.bar([str(v) for v in param_values], f1_scores, color='skyblue')
            
            plt.xlabel(param)
            plt.ylabel('Validation F1 Macro')
            plt.title(f'Effect of {param} on {model_type} Performance')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Save the plot
            plt.savefig(os.path.join(output_dir, f'{model_type}_{param}_analysis.png'), bbox_inches='tight')
            plt.close()


def analyze_grid_search_results(results_df, output_dir):
    """
    Analyze the grid search results and create summary visualizations.
    
    Args:
        results_df (pd.DataFrame): DataFrame with all grid search results
        output_dir (str): Directory to save visualizations
    """
    # Create visualization directory
    viz_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    
    # Analyze performance by model type
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='model_type', y='val_f1_macro', data=results_df)
    plt.title('Validation F1 Macro Score by Model Type')
    plt.xlabel('Model Type')
    plt.ylabel('F1 Macro Score')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(viz_dir, 'model_type_performance.png'), bbox_inches='tight')
    plt.close()
    
    # Create a table with summary statistics for each model type
    summary_stats = results_df.groupby('model_type').agg({
        'val_f1_macro': ['mean', 'std', 'min', 'max'],
        'val_accuracy': ['mean', 'std', 'min', 'max'],
        'test_f1_macro': ['mean', 'std', 'min', 'max'],
        'test_accuracy': ['mean', 'std', 'min', 'max'],
        'num_params': ['mean', 'min', 'max'],
        'training_time': ['mean', 'min', 'max']
    }).round(4)
    
    # Save summary stats
    summary_stats.to_csv(os.path.join(output_dir, 'model_type_summary.csv'))
    
    # Create parameter effect visualizations for each model type
    for model_type in results_df['model_type'].unique():
        model_df = results_df[results_df['model_type'] == model_type]
        
        # Get parameters specific to this model type
        param_columns = [col for col in model_df.columns if col not in [
            'model_type', 'val_f1_macro', 'val_f1_weighted', 'val_accuracy', 'val_auc',
            'test_f1_macro', 'test_f1_weighted', 'test_accuracy', 'test_auc',
            'train_f1_macro', 'train_f1_weighted', 'train_accuracy', 'train_auc',
            'num_params', 'training_time', 'epochs'
        ]]
        
        for param in param_columns:
            if len(model_df[param].unique()) > 1:  # Only analyze if we have different values
                plt.figure(figsize=(10, 6))
                
                # For boolean parameters
                if model_df[param].dtype == bool:
                    sns.boxplot(x=param, y='val_f1_macro', data=model_df)
                    plt.title(f'Effect of {param} on {model_type} Performance')
                # For numeric parameters
                elif pd.api.types.is_numeric_dtype(model_df[param]):
                    # Create grouped boxplot
                    sns.boxplot(x=param, y='val_f1_macro', data=model_df)
                    plt.title(f'Effect of {param} on {model_type} Performance')
                # For other parameters
                else:
                    sns.boxplot(x=param, y='val_f1_macro', data=model_df)
                    plt.title(f'Effect of {param} on {model_type} Performance')
                
                plt.xlabel(param)
                plt.ylabel('Validation F1 Macro')
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                plt.savefig(os.path.join(viz_dir, f'{model_type}_{param}_effect.png'), bbox_inches='tight')
                plt.close()
        
        # Training time vs. performance
        plt.figure(figsize=(10, 6))
        plt.scatter(model_df['training_time'], model_df['val_f1_macro'], alpha=0.7)
        plt.xlabel('Training Time (seconds)')
        plt.ylabel('Validation F1 Macro')
        plt.title(f'{model_type} - Training Time vs. Performance')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(viz_dir, f'{model_type}_time_performance.png'), bbox_inches='tight')
        plt.close()
        
        # Model size vs. performance
        plt.figure(figsize=(10, 6))
        plt.scatter(model_df['num_params'], model_df['val_f1_macro'], alpha=0.7)
        plt.xlabel('Number of Parameters')
        plt.ylabel('Validation F1 Macro')
        plt.title(f'{model_type} - Model Size vs. Performance')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(viz_dir, f'{model_type}_size_performance.png'), bbox_inches='tight')
        plt.close()