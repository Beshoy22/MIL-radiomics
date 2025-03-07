import os
import argparse
import torch
import json
import pandas as pd

from verbose_utils import set_verbose_mode, configure_tqdm_for_grid_search, logger
from dataloader import prepare_dataloaders
from transformer_mil_model import create_model
from lstm_mil_model import create_lstm_model
from conv_mil_model import create_conv_model
from lightweight_conv_mil_model import create_lightweight_conv_model
from model_train import setup_training, train_model, evaluate_model
from utils import set_seed, save_model_and_results, plot_training_curves, plot_confusion_matrix, plot_roc_curve, plot_comparison_metrics
from utils import save_predictions_to_csv
from cross_validation import create_cached_folds, create_fold_loaders
from cross_val_training import run_cross_validation
from metrics_with_ci import evaluate_model_with_ci, plot_metrics_with_ci
from neptune_utils import init_neptune_run, log_model
from center_evaluation import evaluate_by_center, plot_center_metrics


def main(args):
    """
    Main function to train and evaluate the MIL model.
    
    Args:
        args: Command line arguments
        
    Returns:
        tuple: (model, metrics, history) or 
               (model, metrics, history, center_metrics) if not using cross-validation,
               (best_model, fold_metrics, fold_histories) if using cross-validation
    """
    # Configure verbose mode
    verbose_logger = set_verbose_mode(args.verbose)
    
    # Configure tqdm for grid search mode
    configure_tqdm_for_grid_search(args.is_grid_search)
    
    # Log execution info in verbose mode
    if args.verbose:
        verbose_logger.header("MIL FRAMEWORK EXECUTION")
        verbose_logger.log(f"Running with arguments:")
        for arg, value in vars(args).items():
            verbose_logger.log(f"  {arg}: {value}")
    
    # Initialize Neptune logging if enabled
    neptune_run = None
    if args.use_neptune:
        verbose_logger.log("Initializing Neptune logging...")
        neptune_run = init_neptune_run(args)
        if neptune_run:
            verbose_logger.log("Neptune logging initialized successfully")
    
    # Set seed for reproducibility
    verbose_logger.log("Setting random seed for reproducibility...")
    set_seed(args.seed)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    verbose_logger.log(f"Using device: {device}")
    
    # Check for incompatible options
    if args.cv_folds > 1 and args.splitted:
        raise ValueError("Cross-validation is not allowed when using pre-split data files. Please set --cv_folds to 1.")
    
    if args.cv_folds > 1:
        # Cross-validation mode
        verbose_logger.subheader(f"STARTING {args.cv_folds}-FOLD CROSS-VALIDATION")
        
        # Create folds
        with verbose_logger.section("Creating cross-validation folds"):
            folds, max_patches, class_weights = create_cached_folds(
                data_dir=args.data_dir,
                endpoint=args.endpoint,
                n_folds=args.cv_folds,
                seed=args.seed,
                cache_dir=args.cache_dir
            )
            
            if args.verbose:
                verbose_logger.log(f"Created {len(folds)} folds with max_patches={max_patches}")
                verbose_logger.log(f"Class weights: {class_weights}")
        
        # Run cross-validation
        verbose_logger.log("Starting cross-validation training...")
        best_model, fold_metrics, fold_histories = run_cross_validation(
            args=args,
            folds=folds,
            max_patches=max_patches,
            class_weights=class_weights,
            device=device,
            neptune_run=neptune_run
        )
        
        # Save CV predictions to CSV
        fold_predictions = []
        for fold_idx, metrics in enumerate(fold_metrics):
            if 'patient_ids' in metrics and 'all_labels' in metrics:
                for i, patient_id in enumerate(metrics['patient_ids']):
                    fold_predictions.append({
                        'fold': fold_idx + 1,
                        'patient_id': patient_id,
                        'ground_truth': metrics['all_labels'][i],
                        'prediction': metrics['all_preds'][i],
                        'probability': metrics['all_probs'][i],
                        'center': metrics.get('centers', ['unknown'] * len(metrics['patient_ids']))[i]
                    })
        
        # Save all CV predictions to CSV
        if fold_predictions:
            df = pd.DataFrame(fold_predictions)
            cv_csv_path = os.path.join(args.output_dir, 'cv_predictions.csv')
            df.to_csv(cv_csv_path, index=False)
            print(f"Cross-validation predictions saved to {cv_csv_path}")
        
        # Log final model to Neptune
        if neptune_run:
            log_model(neptune_run, best_model, name="final_model")
            neptune_run.stop()
        
        return best_model, fold_metrics, fold_histories
        
    else:
        # Standard train/val/test mode
        print("Using standard train/val/test split")
        
        # Prepare data loaders
        train_loader, val_loader, test_loader, class_weights, metrics, max_patches = prepare_dataloaders(
            data_dir=args.data_dir,
            endpoint=args.endpoint,
            batch_size=args.batch_size,
            oversample_factor=args.oversample_factor,
            val_size=args.val_size,
            test_size=args.test_size,
            num_workers=args.num_workers,
            seed=args.seed,
            use_cache=args.use_cache,
            cache_dir=args.cache_dir,
            splitted=args.splitted
        )
        print(f"Data loaders ready")
        
        # Set output directory on dataset objects to allow saving predictions
        if hasattr(test_loader.dataset, '__dict__'):
            test_loader.dataset.output_dir = args.output_dir
        
        # Log dataset sizes to Neptune
        if neptune_run:
            neptune_run["data/train_samples"] = len(train_loader.dataset)
            neptune_run["data/val_samples"] = len(val_loader.dataset)
            neptune_run["data/test_samples"] = len(test_loader.dataset)
            neptune_run["data/max_patches"] = max_patches
        
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
            print(f"Transformer model ready")
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
            print(f"LSTM model ready (with {'attention' if args.use_attention else 'pooling'})")
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
            print(f"Convolutional model ready")
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
            print(f"Lightweight convolutional model ready")
        else:
            raise ValueError(f"Unsupported model type: {args.model_type}")
        
        # Print model architecture summary
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model has {num_params:,} trainable parameters")
        
        # Log model params to Neptune
        if neptune_run:
            neptune_run["model/trainable_parameters"] = num_params
        
        # Set up training components
        criterion, optimizer, scheduler = setup_training(
            model=model,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            class_weights=class_weights,
            device=device
        )
        
        # Train model with specified selection metric
        print(f"Training {args.model_type} model (using {args.selection_metric} for model selection)...")
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
            selection_metric=args.selection_metric,
            neptune_run=neptune_run
        )
        
        # Evaluate model with confidence intervals
        print(f"Evaluating {args.model_type} model with confidence intervals...")
        metrics = evaluate_model_with_ci(
            model=model,
            dataloader=test_loader,
            device=device,
            n_bootstrap=args.bootstrap_samples,
            neptune_run=neptune_run
        )
        
        # Save predictions to CSV
        if 'patient_ids' in metrics and 'all_labels' in metrics:
            save_predictions_to_csv(
                patient_ids=metrics['patient_ids'],
                labels=metrics['all_labels'],
                predictions=metrics['all_preds'],
                probabilities=metrics['all_probs'],
                centers=metrics.get('centers'),
                output_dir=args.output_dir
            )
        
        # Also get standard metrics on all datasets for comparison
        standard_metrics = evaluate_model(
            model=model,
            test_loader=test_loader,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
            neptune_run=neptune_run
        )
        
        # Center-based evaluation
        print(f"Evaluating {args.model_type} model by center...")
        center_metrics = evaluate_by_center(
            model=model,
            test_loader=test_loader,
            device=device,
            neptune_run=neptune_run
        )
        
        # Plot center-based metrics
        print(f"Plotting center-based metrics...")
        plot_center_metrics(
            center_metrics=center_metrics,
            key_metrics=['f1_macro', 'auc'],
            output_dir=args.output_dir,
            neptune_run=neptune_run,
            min_samples=args.min_center_samples
        )
        
        # Save model and results
        save_model_and_results(
            model=model,
            metrics=metrics,
            history=history,
            output_dir=args.output_dir,
            center_metrics=center_metrics
        )
        
        # Plot results
        plot_training_curves(history, args.output_dir, neptune_run)
        plot_confusion_matrix(metrics['all_labels'], metrics['all_preds'], args.output_dir, neptune_run)
        plot_roc_curve(metrics['all_labels'], metrics['all_probs'], args.output_dir, neptune_run)
        plot_comparison_metrics(standard_metrics, args.output_dir, neptune_run)
        plot_metrics_with_ci(metrics, args.output_dir, neptune_run)
        
        # Log final model to Neptune
        if neptune_run:
            log_model(neptune_run, model, name="final_model")
            neptune_run.stop()
        
        return model, metrics, history, center_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train MIL model for CT patch embeddings')
    
    # Add verbose flag to parser
    parser.add_argument('--verbose', action='store_true', 
                        help='Enable verbose output with detailed information for debugging')
    
    # Model type
    parser.add_argument('--model_type', type=str, default='transformer', 
                        choices=['transformer', 'lstm', 'conv', 'lightweight_conv'], 
                        help='Type of model to train')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing .pkl files')
    parser.add_argument('--endpoint', type=str, default='OS_6', choices=['OS_6', 'OS_24'], 
                        help='Endpoint to use')
    parser.add_argument('--oversample_factor', type=float, default=1.0, 
                        help='Factor for oversampling minority class (0 to disable)')
    parser.add_argument('--val_size', type=float, default=0.15, help='Validation set size')
    parser.add_argument('--test_size', type=float, default=0.15, help='Test set size')
    parser.add_argument('--splitted', action='store_true', 
                        help='Use pre-splitted train/val/test pkl files (train_set.pkl, val_set.pkl, test_set.pkl)')
    
    # Cross-validation arguments
    parser.add_argument('--cv_folds', type=int, default=1, 
                        help='Number of folds for cross-validation (1 for no CV)')
    parser.add_argument('--bootstrap_samples', type=int, default=1000,
                        help='Number of bootstrap samples for confidence intervals')
    
    # Caching arguments
    parser.add_argument('--use_cache', action='store_true', help='Use caching for faster loading')
    parser.add_argument('--cache_dir', type=str, default=None, help='Directory to store cached data files')
    
    # Model arguments (common)
    parser.add_argument('--feature_dim', type=int, default=512, help='Dimension of input features')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension in model')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of transformer/LSTM layers')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    
    # Transformer-specific arguments
    parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads (transformer only)')
    
    # LSTM-specific arguments
    parser.add_argument('--bidirectional', action='store_true', help='Use bidirectional LSTM (LSTM only)')
    parser.add_argument('--use_attention', action='store_true', 
                        help='Use attention mechanism in LSTM (if false, uses average pooling)')
    
    # Conv-specific arguments
    parser.add_argument('--num_groups', type=int, default=10, 
                        help='Number of groups for patch aggregation (conv models only)')
    parser.add_argument('--num_blocks', type=int, default=2,
                        help='Number of convolutional blocks (lightweight_conv only)')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--num_epochs', type=int, default=100, help='Maximum number of epochs')
    parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--selection_metric', type=str, default='f1_macro', 
                        choices=['f1_macro', 'val_loss'], 
                        help='Metric to use for model selection during training')
    
    # Center evaluation arguments
    parser.add_argument('--min_center_samples', type=int, default=10,
                        help='Minimum number of samples for a center to be included in visualization')
    
    # Output arguments
    parser.add_argument('--export_predictions', action='store_true', help='Export patient-level predictions to CSV')
    parser.add_argument('--predictions_filename', type=str, default='predictions.csv',
                        help='Filename for exported predictions')
    
    # Grid search arguments
    parser.add_argument('--grid_search', action='store_true', help='Enable grid search')
    parser.add_argument('--grid_search_top_k', type=int, default=5, help='Number of top models to keep from grid search')
    parser.add_argument('--grid_search_config', type=str, default=None, 
                        help='Path to grid search configuration JSON file')
    parser.add_argument('--create_sample_grid_config', action='store_true',
                        help='Create a sample grid search configuration file and exit')
    
    # Neptune logging argument
    parser.add_argument('--use_neptune', action='store_true', help='Enable Neptune logging')
    
    # Other arguments
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output_dir', type=str, default=None, 
                       help='Output directory for results (defaults to ./outputs/model_type)')
    parser.add_argument('--cpu', action='store_true', help='Use CPU even if GPU is available')
    
    args = parser.parse_args()
    
    # Create sample grid search config if requested
    if args.create_sample_grid_config:
        from gridsearch import create_sample_grid_search_config
        config_path = create_sample_grid_search_config()
        print(f"Sample grid search configuration created at: {config_path}")
        print("You can use this as a starting point and modify it for your needs.")
        print(f"To use it, run with: --grid_search --grid_search_config {config_path}")
        exit(0)
    
    # Set default output directory if not specified
    if args.output_dir is None:
        if args.cv_folds > 1:
            args.output_dir = f'./outputs/{args.model_type}_cv{args.cv_folds}'
        else:
            args.output_dir = f'./outputs/{args.model_type}'
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create cache directory if specified and doesn't exist
    if args.cache_dir:
        os.makedirs(args.cache_dir, exist_ok=True)
    
    # Run grid search if enabled
    if args.grid_search:
        from gridsearch import run_grid_search, load_grid_search_config, get_default_param_grid
        
        # Load or create parameter grid
        if args.grid_search_config:
            param_grid = load_grid_search_config(args.grid_search_config)
            if param_grid is None:
                # If loading fails, use default grid
                param_grid = get_default_param_grid(args.model_type)
        else:
            # Use default parameter grid
            param_grid = get_default_param_grid(args.model_type)
        
        print(f"Running grid search with the following parameter grid:")
        print(json.dumps(param_grid, indent=2))
        
        # Run grid search
        top_models, grid_search_dir = run_grid_search(
            args=args,
            param_grid=param_grid,
            top_k=args.grid_search_top_k,
            main_func=main
        )
        
        print(f"\nGrid search completed. Results saved to {grid_search_dir}")
        print("\nTop models:")
        for i, (_, model) in enumerate(top_models.iterrows()):
            print(f"Rank {i+1}: Validation F1 Macro: {model.get('val_f1_macro', 'N/A')}")
            if 'test_f1_macro' in model and 'test_f1_macro_ci_low' in model and 'test_f1_macro_ci_high' in model:
                print(f"  Test F1 Macro: {model['test_f1_macro']:.4f} (95% CI: {model['test_f1_macro_ci_low']:.4f}-{model['test_f1_macro_ci_high']:.4f})")
            print(f"  Directory: {model['output_dir']}")
            
            # Print hyperparameters of this model
            print("  Hyperparameters:")
            for param in param_grid.keys():
                if param in model:
                    print(f"    {param}: {model[param]}")
            print()
            
            # Create a combined predictions CSV file for all grid search models
            combined_predictions = []
            for j, (_, model_info) in enumerate(top_models.iterrows()):
                model_dir = model_info['output_dir']
                predictions_path = os.path.join(model_dir, 'predictions.csv')
                if os.path.exists(predictions_path):
                    df = pd.read_csv(predictions_path)
                    df['model_rank'] = j + 1
                    for param in param_grid.keys():
                        if param in model_info:
                            df[f'param_{param}'] = model_info[param]
                    combined_predictions.append(df)
            
            if combined_predictions:
                combined_df = pd.concat(combined_predictions, ignore_index=True)
                combined_path = os.path.join(grid_search_dir, 'all_model_predictions.csv')
                combined_df.to_csv(combined_path, index=False)
                print(f"Combined predictions from all models saved to {combined_path}")
    else:
        # Run main function for a single training
        main(args)