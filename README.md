# Multiple Instance Learning (MIL) Framework with Neptune Logging

This framework provides a comprehensive solution for training and evaluating Multiple Instance Learning (MIL) models on patch-based data, with integrated Neptune.ai logging for experiment tracking.

## Features

- **Multiple MIL architectures**:
  - Transformer-based MIL
  - LSTM-based MIL
  - Convolutional MIL
  - Lightweight Convolutional MIL
- **Comprehensive data handling**:
  - Automatic caching for faster loading
  - Support for pre-split datasets
  - Configurable data preprocessing
- **Robust evaluation**:
  - Bootstrap confidence intervals
  - Cross-validation support
  - Multiple evaluation metrics
- **Visualization tools**:
  - Training curves
  - Confusion matrices
  - ROC and PR curves
  - Attention visualization
- **Neptune.ai integration**:
  - Experiment tracking
  - Metric logging
  - Visualization logging
  - Model artifact tracking

## Installation

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd mil-framework
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure Neptune logging (optional):
   - Create a `.env` file in the project root with your Neptune API key and project name:
   ```
   NEPTUNE_API_KEY=your_api_key_here
   NEPTUNE_PROJECT=your_workspace/your_project_name
   ```

## Usage

### Basic Training

To train a model with default parameters:

```bash
python main.py --data_dir /path/to/data --model_type transformer --use_neptune
```

### Cross-Validation

To train with cross-validation:

```bash
python main.py --data_dir /path/to/data --model_type transformer --cv_folds 5 --use_neptune
```

### Model Types

Choose from different MIL architectures:

```bash
# Transformer-based MIL
python main.py --data_dir /path/to/data --model_type transformer --use_neptune

# LSTM-based MIL
python main.py --data_dir /path/to/data --model_type lstm --use_neptune

# Convolutional MIL
python main.py --data_dir /path/to/data --model_type conv --use_neptune

# Lightweight Convolutional MIL
python main.py --data_dir /path/to/data --model_type lightweight_conv --use_neptune
```

### Key Arguments

- `--data_dir`: Path to the directory containing data files
- `--model_type`: Type of model architecture (`transformer`, `lstm`, `conv`, or `lightweight_conv`)
- `--endpoint`: Endpoint to use for classification (`OS_6` or `OS_24`)
- `--cv_folds`: Number of folds for cross-validation (default: 1, meaning no cross-validation)
- `--use_neptune`: Enable Neptune.ai logging
- `--output_dir`: Directory to save outputs (default: `./outputs/{model_type}`)
- `--batch_size`: Batch size for training
- `--lr`: Learning rate
- `--num_epochs`: Maximum number of training epochs
- `--patience`: Patience for early stopping
- `--seed`: Random seed for reproducibility

See all available options:

```bash
python main.py --help
```

## Data Format

The framework expects data in Python pickle (.pkl) files with the following structure:

- Each .pkl file contains a list of instances
- Each instance is a dictionary containing:
  - `features`: Patch embeddings (tensor or array) of shape [n_patches, feature_dim]
  - `OS_6` or `OS_24`: Binary label (0 or 1) for the endpoint
  - Other metadata (optional)

## Neptune.ai Integration

Neptune.ai is used for experiment tracking and visualization. When enabled with `--use_neptune`, the framework will log:

- Model parameters and hyperparameters
- Training and validation metrics (loss, accuracy, F1 scores)
- Learning rate changes
- Best model checkpoint
- Evaluation metrics with confidence intervals
- Visualizations (training curves, confusion matrices, ROC curves)
- Cross-validation results

Access the Neptune dashboard to view and compare experiments.

## Output Structure

The framework saves outputs to the specified directory (or a default one):

```
outputs/
    ├── {model_type}/                # Standard training output
    │   ├── model.pt                 # Trained model weights
    │   ├── metrics.json             # Evaluation metrics
    │   ├── history.json             # Training history
    │   ├── training_curves.png      # Training curves plot
    │   ├── confusion_matrix.png     # Confusion matrix plot
    │   ├── roc_curve.png            # ROC curve plot
    │   └── metrics_with_ci.png      # Metrics with confidence intervals
    │
    └── {model_type}_cv{n}/          # Cross-validation output
        ├── best_model.pt            # Best model weights
        ├── cv_metrics.json          # Aggregated CV metrics
        ├── cv_metrics.png           # CV metrics plot
        ├── fold_1/                  # Fold 1 outputs
        ├── fold_2/                  # Fold 2 outputs
        └── ...
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.