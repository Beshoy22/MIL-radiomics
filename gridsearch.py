import os
import itertools
import json
import time
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from datetime import datetime
import copy
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from verbose_utils import logger, set_verbose_mode

def run_grid_search(args, param_grid, top_k=5, main_func=None):
    """
    Run grid search with specified parameter grid.
    
    Args:
        args: Command-line arguments (will be modified for each combination)
        param_grid (dict): Dictionary of parameter names and lists of values
        top_k (int): Number of top models to return
        main_func (callable): Main function to call for each combination
        
    Returns:
        tuple: (top_k_models, grid_search_dir)
    """
    # Enable verbose mode if specified
    if hasattr(args, 'verbose') and args.verbose:
        set_verbose_mode(True)
        logger.log("Running grid search with verbose mode enabled")
    
    if main_func is None:
        # Import here to avoid circular imports
        from main import main as main_func
    
    # Generate all combinations of parameters
    param_names = list(param_grid.keys())
    param_values = list(itertools.product(*[param_grid[name] for name in param_names]))
    
    total_combinations = len(param_values)
    if hasattr(args, 'verbose') and args.verbose:
        logger.log(f"Grid search parameters:")
        for name in param_names:
            logger.log(f"  {name}: {param_grid[name]}")
        logger.log(f"Total combinations: {total_combinations}")
    
    print(f"Running grid search with {total_combinations} combinations")
    
    # Create a directory to store grid search results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    grid_search_dir = os.path.join(args.output_dir, f"grid_search_{timestamp}")
    os.makedirs(grid_search_dir, exist_ok=True)
    
    # Save parameter grid for reference
    with open(os.path.join(grid_search_dir, "param_grid.json"), "w") as f:
        json.dump(param_grid, f, indent=2)
    
    # Results storage
    results = []
    
    # Run training for each parameter combination
    for i, param_combination in enumerate(tqdm(param_values, desc="Grid search progress", position=0)):
        # Create a copy of arguments and update with current parameters
        current_args = copy.deepcopy(args)
        
        # Mark this run as part of a grid search to handle tqdm properly
        current_args.is_grid_search = True
        
        # Update args with current parameter values
        for name, value in zip(param_names, param_combination):
            setattr(current_args, name, value)
        
        # Create a directory for this combination
        combination_id = f"combination_{i+1:04d}"
        combination_dir = os.path.join(grid_search_dir, combination_id)
        current_args.output_dir = combination_dir
        os.makedirs(combination_dir, exist_ok=True)
        
        # Save current parameters
        with open(os.path.join(combination_dir, "params.json"), "w") as f:
            params_dict = {name: getattr(current_args, name) for name in param_names}
            json.dump(params_dict, f, indent=2)
        
        try:
            print(f"\n{'-'*30} Running combination {i+1}/{total_combinations} {'-'*30}")
            # Display the current parameter combination
            current_params_str = ", ".join([f"{name}={getattr(current_args, name)}" for name in param_names])
            print(f"Parameters: {current_params_str}")
            
            # Remember start time
            start_time = time.time()
            
            # Run training with current parameters - disable nested progress bars
            with tqdm.external_write_mode():
                if current_args.cv_folds > 1:
                    # Cross-validation mode
                    model, fold_metrics, history = main_func(current_args)
                    metrics = fold_metrics[0]  # Use the first fold for simplicity in this case
                else:
                    # Standard mode
                    model, metrics, history, _ = main_func(current_args)
            
            # Calculate elapsed time
            elapsed_time = time.time() - start_time
            
            # Extract key metrics
            result = {
                "combination_id": combination_id,
                "output_dir": combination_dir,
                "elapsed_time": elapsed_time
            }
            
            # Add parameters
            for name, value in zip(param_names, param_combination):
                result[name] = value
            
            # Add metrics
            # If standard train/val/test mode was used
            if "all_datasets" in metrics:
                for dataset in ["train", "val", "test"]:
                    if dataset in metrics["all_datasets"]:
                        dataset_metrics = metrics["all_datasets"][dataset]
                        for metric_name in ["accuracy", "precision", "recall", "f1", "f1_macro", "f1_weighted", "auc"]:
                            if metric_name in dataset_metrics:
                                result[f"{dataset}_{metric_name}"] = dataset_metrics[metric_name]
            
            # If confidence intervals were computed
            for metric_name in ["accuracy", "precision", "recall", "f1", "f1_macro", "f1_weighted", "auc"]:
                ci_key = f"{metric_name}_ci"
                if ci_key in metrics:
                    result[f"test_{metric_name}_ci_low"] = metrics[ci_key][0]
                    result[f"test_{metric_name}_ci_high"] = metrics[ci_key][1]
            
            # Store results
            results.append(result)
            
            # Save current results to interim CSV to keep track of progress
            results_df = pd.DataFrame(results)
            results_df.to_csv(os.path.join(grid_search_dir, "interim_results.csv"), index=False)
            
            # Print a summary of the results so far
            print(f"\nCombination {i+1}/{total_combinations} completed in {elapsed_time/60:.2f} minutes")
            if "val_f1_macro" in result:
                print(f"  Val F1 Macro: {result['val_f1_macro']:.4f}")
            if "test_f1_macro" in result:
                print(f"  Test F1 Macro: {result['test_f1_macro']:.4f}")
            
        except Exception as e:
            print(f"Error in combination {combination_id}: {e}")
            # Log the error
            with open(os.path.join(combination_dir, "error.log"), "w") as f:
                f.write(str(e))
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save all results
    results_df.to_csv(os.path.join(grid_search_dir, "all_results.csv"), index=False)
    
    # Sort by validation F1 macro and select top k
    if len(results_df) > 0:
        if "val_f1_macro" in results_df.columns:
            top_k_models = results_df.sort_values("val_f1_macro", ascending=False).head(top_k)
        else:
            print("Warning: val_f1_macro not found in results. Using available metrics for sorting.")
            # Try to find an alternative metric
            for metric in ["val_f1", "val_accuracy", "test_f1_macro", "test_f1"]:
                if metric in results_df.columns:
                    top_k_models = results_df.sort_values(metric, ascending=False).head(top_k)
                    break
            else:
                # If no alternative found, just take the first k
                top_k_models = results_df.head(min(top_k, len(results_df)))
        
        # Save top k models
        top_k_models.to_csv(os.path.join(grid_search_dir, "top_k_models.csv"), index=False)
        
        # Create a summary report
        create_summary_report(top_k_models, grid_search_dir)
        
        # Visualize results
        visualize_grid_search_results(results_df, grid_search_dir)
    else:
        print("No successful grid search runs to analyze")
        top_k_models = pd.DataFrame()
    
    return top_k_models, grid_search_dir

def create_summary_report(top_k_models, output_dir):
    """
    Create a detailed summary report of the top k models.
    
    Args:
        top_k_models (DataFrame): DataFrame with top k models
        output_dir (str): Directory to save the report
    """
    with open(os.path.join(output_dir, "summary_report.txt"), "w") as f:
        f.write("=" * 80 + "\n")
        f.write("GRID SEARCH SUMMARY REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Total models evaluated: {len(top_k_models)}\n")
        f.write(f"Report generation time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("TOP PERFORMING MODELS\n")
        f.write("-" * 80 + "\n\n")
        
        for i, (_, model) in enumerate(top_k_models.iterrows()):
            f.write(f"Rank {i+1}: {model['combination_id']}\n")
            
            # Write hyperparameters
            f.write("  Hyperparameters:\n")
            for col in model.index:
                if col not in ["combination_id", "output_dir", "elapsed_time"] and not col.startswith("train_") and not col.startswith("val_") and not col.startswith("test_"):
                    f.write(f"    {col}: {model[col]}\n")
            
            # Write performance metrics
            f.write("\n  Performance Metrics:\n")
            
            # Training metrics
            train_metrics = {col[6:]: model[col] for col in model.index if col.startswith("train_")}
            if train_metrics:
                f.write("    Training:\n")
                for metric, value in train_metrics.items():
                    f.write(f"      {metric}: {value:.4f}\n")
            
            # Validation metrics
            val_metrics = {col[4:]: model[col] for col in model.index if col.startswith("val_")}
            if val_metrics:
                f.write("    Validation:\n")
                for metric, value in val_metrics.items():
                    f.write(f"      {metric}: {value:.4f}\n")
            
            # Test metrics
            test_metrics = {col[5:]: model[col] for col in model.index if col.startswith("test_") and not col.endswith("_ci_low") and not col.endswith("_ci_high")}
            test_ci_low = {col[5:-7]: model[col] for col in model.index if col.endswith("_ci_low")}
            test_ci_high = {col[5:-8]: model[col] for col in model.index if col.endswith("_ci_high")}
            
            if test_metrics:
                f.write("    Test:\n")
                for metric, value in test_metrics.items():
                    if metric in test_ci_low and metric in test_ci_high:
                        f.write(f"      {metric}: {value:.4f} (95% CI: {test_ci_low[metric]:.4f}-{test_ci_high[metric]:.4f})\n")
                    else:
                        f.write(f"      {metric}: {value:.4f}\n")
            
            # Training time
            if "elapsed_time" in model:
                elapsed_minutes = model["elapsed_time"] / 60
                f.write(f"\n  Training Time: {elapsed_minutes:.2f} minutes\n")
            
            f.write("\n" + "-" * 80 + "\n\n")
        
        f.write("\nCOMPARISON OF TOP MODELS\n")
        f.write("-" * 80 + "\n\n")
        
        # Compare key metrics across models
        key_metrics = ["f1_macro", "accuracy", "auc"]
        for metric in key_metrics:
            if f"val_{metric}" in top_k_models.columns:
                f.write(f"{metric.upper()} comparison:\n")
                for i, (_, model) in enumerate(top_k_models.iterrows()):
                    val_value = model.get(f"val_{metric}", float('nan'))
                    test_value = model.get(f"test_{metric}", float('nan'))
                    
                    ci_low = model.get(f"test_{metric}_ci_low", float('nan'))
                    ci_high = model.get(f"test_{metric}_ci_high", float('nan'))
                    
                    if not np.isnan(ci_low) and not np.isnan(ci_high):
                        f.write(f"  Rank {i+1}: Val: {val_value:.4f}, Test: {test_value:.4f} (95% CI: {ci_low:.4f}-{ci_high:.4f})\n")
                    else:
                        f.write(f"  Rank {i+1}: Val: {val_value:.4f}, Test: {test_value:.4f}\n")
                f.write("\n")
        
        f.write("\nCONCLUSION\n")
        f.write("-" * 80 + "\n\n")
        
        # Best hyperparameter values
        f.write("Best hyperparameter values based on top models:\n")
        param_stats = defaultdict(list)
        
        # Identify hyperparameters (columns that are not metrics or IDs)
        hyperparams = [col for col in top_k_models.columns 
                       if col not in ["combination_id", "output_dir", "elapsed_time"] 
                       and not col.startswith("train_") 
                       and not col.startswith("val_") 
                       and not col.startswith("test_")]
        
        for param in hyperparams:
            values = top_k_models[param].values
            if all(isinstance(v, (int, float)) for v in values):
                f.write(f"  {param}: mean={np.mean(values):.4f}, median={np.median(values):.4f}, range={min(values)}-{max(values)}\n")
            else:
                # For categorical parameters, count occurrences
                value_counts = top_k_models[param].value_counts()
                most_common = value_counts.index[0]
                f.write(f"  {param}: most common={most_common} (in {value_counts[most_common]}/{len(top_k_models)} models)\n")
        
        f.write("\nRecommended configuration for best performance:\n")
        best_model = top_k_models.iloc[0]
        for param in hyperparams:
            f.write(f"  {param}: {best_model[param]}\n")

def visualize_grid_search_results(results_df, output_dir):
    """
    Create visualizations of grid search results.
    
    Args:
        results_df (DataFrame): DataFrame with grid search results
        output_dir (str): Directory to save visualizations
    """
    # Create visualizations directory
    viz_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    
    # Identify parameters and metrics
    params = [col for col in results_df.columns 
              if col not in ["combination_id", "output_dir", "elapsed_time"] 
              and not col.startswith("train_") 
              and not col.startswith("val_") 
              and not col.startswith("test_")]
    
    # Key metrics to visualize
    metrics = ["val_f1_macro", "test_f1_macro", "val_accuracy", "test_accuracy", "val_auc", "test_auc"]
    metrics = [m for m in metrics if m in results_df.columns]
    
    if not metrics:
        print("No suitable metrics found for visualization")
        return
    
    # 1. Parameter impact on key metrics
    for param in params:
        # Skip parameters with only one value
        if len(results_df[param].unique()) <= 1:
            continue
        
        # Check parameter type
        param_values = results_df[param].values
        is_numeric = all(isinstance(v, (int, float)) for v in param_values)
        
        for metric in metrics:
            plt.figure(figsize=(10, 6))
            
            if is_numeric:
                # For numeric parameters, use scatter plot with regression line
                sns.regplot(x=param, y=metric, data=results_df, scatter_kws={'alpha':0.6})
            else:
                # For categorical parameters, use box plot
                sns.boxplot(x=param, y=metric, data=results_df)
            
            plt.title(f"Impact of {param} on {metric}")
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, f"{param}_{metric}.png"))
            plt.close()
    
    # 2. Pair plot for parameters and key metrics (if we have enough numeric columns)
    numeric_cols = []
    for col in params + metrics:
        if col in results_df.columns and pd.api.types.is_numeric_dtype(results_df[col]):
            numeric_cols.append(col)
    
    if len(numeric_cols) > 1:  # Need at least 2 columns for a pair plot
        plt.figure(figsize=(12, 10))
        g = sns.pairplot(results_df[numeric_cols], kind="scatter", diag_kind="kde")
        g.fig.suptitle("Parameter Relationships", y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, "parameter_relationships.png"))
        plt.close()
    
    # 3. Top models comparison
    if "val_f1_macro" in results_df.columns:
        top_k = min(5, len(results_df))
        top_models = results_df.sort_values("val_f1_macro", ascending=False).head(top_k)
        
        # Select key metrics for comparison
        comparison_metrics = [m for m in metrics if m.startswith("val_") or m.startswith("test_")]
        comparison_metrics = comparison_metrics[:6]  # Limit to 6 metrics for readability
        
        if comparison_metrics and len(top_models) > 0:
            plt.figure(figsize=(14, 8))
            
            # Reshape for plotting
            plot_data = []
            for i, (_, model) in enumerate(top_models.iterrows()):
                for metric in comparison_metrics:
                    if metric in model:
                        plot_data.append({
                            "Model": f"Rank {i+1}",
                            "Metric": metric,
                            "Value": model[metric]
                        })
            
            if plot_data:
                plot_df = pd.DataFrame(plot_data)
                
                # Create grouped bar chart
                sns.barplot(x="Model", y="Value", hue="Metric", data=plot_df)
                plt.title("Comparison of Top Models")
                plt.legend(title="Metric")
                plt.grid(alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(viz_dir, "top_models_comparison.png"))
                plt.close()

def load_grid_search_config(config_file):
    """
    Load grid search configuration from a JSON file.
    
    Args:
        config_file (str): Path to the configuration file
        
    Returns:
        dict: Grid search parameter configuration
    """
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        # Validate the configuration
        if not isinstance(config, dict):
            raise ValueError("Grid search configuration must be a dictionary")
        
        # Ensure all values are lists
        for param, values in config.items():
            if not isinstance(values, list):
                config[param] = [values]
                
        return config
    except Exception as e:
        print(f"Error loading grid search configuration: {e}")
        print("Using default parameter grid instead")
        return None

def get_default_param_grid(model_type):
    """
    Get default parameter grid based on model type.
    
    Args:
        model_type (str): Type of model ('transformer', 'lstm', 'conv', 'lightweight_conv')
        
    Returns:
        dict: Default parameter grid
    """
    # Base parameters for all models
    param_grid = {
        'lr': [1e-5, 1e-4, 5e-4],
        'dropout': [0.1, 0.3, 0.5],
        'hidden_dim': [64, 128, 256]
    }
    
    # Add model-specific parameters
    if model_type == 'transformer':
        param_grid['num_heads'] = [2, 4, 8]
        param_grid['num_layers'] = [1, 2, 3]
    elif model_type == 'lstm':
        param_grid['num_layers'] = [1, 2, 3]
        param_grid['bidirectional'] = [True, False]
        param_grid['use_attention'] = [True, False]
    elif model_type == 'conv':
        param_grid['num_groups'] = [5, 10, 20]
    elif model_type == 'lightweight_conv':
        param_grid['num_groups'] = [5, 10, 20]
        param_grid['num_blocks'] = [1, 2, 3]
    
    return param_grid