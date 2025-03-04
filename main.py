import os
import argparse
import torch

from dataloader import prepare_dataloaders
from transformer_mil_model import create_model
from lstm_mil_model import create_lstm_model
from model_train import setup_training, train_model, evaluate_model
from utils import set_seed, save_model_and_results, plot_training_curves, plot_confusion_matrix, plot_roc_curve, plot_comparison_metrics


def main(args):
    """
    Main function to train and evaluate the MIL model.
    
    Args:
        args: Command line arguments
    """
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"Using device: {device}")
    
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
        cache_dir=args.cache_dir
    )
    print(f"Data loaders ready")
    
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
            device=device
        )
        print(f"LSTM model ready")
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")
    
    # Print model architecture summary
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {num_params:,} trainable parameters")
    
    # Set up training components
    criterion, optimizer, scheduler = setup_training(
        model=model,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        class_weights=class_weights,
        device=device
    )
    
    # Train model
    print(f"Training {args.model_type} model...")
    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=args.num_epochs,
        early_stopping_patience=args.patience,
        device=device
    )
    
    # Evaluate model on all datasets
    print(f"Evaluating {args.model_type} model on all datasets...")
    metrics = evaluate_model(
        model=model,
        test_loader=test_loader,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        device=device
    )
    
    # Save model and results
    save_model_and_results(
        model=model,
        metrics=metrics,
        history=history,
        output_dir=args.output_dir
    )
    
    # Plot results
    plot_training_curves(history, args.output_dir)
    plot_confusion_matrix(metrics['all_labels'], metrics['all_preds'], args.output_dir)
    plot_roc_curve(metrics['all_labels'], metrics['all_probs'], args.output_dir)
    # Plot comparison metrics across datasets
    plot_comparison_metrics(metrics, args.output_dir)
    
    return model, metrics, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train MIL model for CT patch embeddings')
    
    # Model type
    parser.add_argument('--model_type', type=str, default='transformer', 
                        choices=['transformer', 'lstm'], help='Type of model to train')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing .pkl files')
    parser.add_argument('--endpoint', type=str, default='OS_6', choices=['OS_6', 'OS_24'], 
                        help='Endpoint to use')
    parser.add_argument('--oversample_factor', type=float, default=1.0, 
                        help='Factor for oversampling minority class (0 to disable)')
    parser.add_argument('--val_size', type=float, default=0.15, help='Validation set size')
    parser.add_argument('--test_size', type=float, default=0.15, help='Test set size')
    
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
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--num_epochs', type=int, default=100, help='Maximum number of epochs')
    parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    
    # Other arguments
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output_dir', type=str, default=None, 
                       help='Output directory for results (defaults to ./outputs/model_type)')
    parser.add_argument('--cpu', action='store_true', help='Use CPU even if GPU is available')
    
    args = parser.parse_args()
    
    # Set default output directory if not specified
    if args.output_dir is None:
        args.output_dir = f'./outputs/{args.model_type}'
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create cache directory if specified and doesn't exist
    if args.cache_dir:
        os.makedirs(args.cache_dir, exist_ok=True)
    
    # Run main function
    main(args)