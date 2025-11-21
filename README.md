# Multiple Instance Learning (MIL) for Radiomics

This framework provides a comprehensive solution for training and evaluating Multiple Instance Learning (MIL) models on patch-based data, with integrated Neptune.ai logging for experiment tracking.

## Features

- **Multiple MIL architectures**:
  - Transformer-based MIL with multi-head attention
  - LSTM-based MIL with optional attention mechanism
  - Convolutional MIL with patch grouping
  - Lightweight Convolutional MIL for resource efficiency
  - Dense MIL with configurable layer architecture
- **Comprehensive data handling**:
  - Automatic caching for faster loading
  - Support for pre-split datasets
  - Configurable data preprocessing
  - Handling of complex patch feature structures
- **Robust evaluation**:
  - Bootstrap confidence intervals
  - Cross-validation support
  - Multiple evaluation metrics
  - Center-based performance analysis
- **Advanced training tools**:
  - Grid search for hyperparameter optimization
  - Class imbalance handling via oversampling
  - Early stopping with customizable metrics
  - Learning rate scheduling
- **Visualization tools**:
  - Training curves
  - Confusion matrices
  - ROC and PR curves
  - Attention visualization
  - Center performance comparison
- **Neptune.ai integration**:
  - Experiment tracking
  - Metric logging
  - Visualization logging
  - Model artifact tracking

## Installation

### Standard Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/Beshoy22/MIL-radiomics.git
   cd MIL-radiomics
   ```

2. Install the package:
   ```bash
   # Basic installation
   pip install -r requirements.txt

   # Or install as a package (recommended)
   pip install -e .
   ```

### Development Installation

For contributors and developers:

```bash
# Install with development dependencies
pip install -e .
pip install -r requirements-dev.txt

# Or use the dev extras
pip install -e ".[dev]"
```

### Testing

Run the test suite to verify installation:

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=. --cov-report=html

# Run specific test file
pytest tests/test_dataloader.py -v
```

### Configuration

1. **Neptune.ai logging** (optional):
   Create a `.env` file in the project root with your Neptune credentials:
   ```env
   NEPTUNE_API_KEY=your_api_key_here
   NEPTUNE_PROJECT=your_workspace/your_project_name
   ```

2. **Logging Level** (optional):
   Set the logging verbosity:
   ```env
   LOG_LEVEL=DEBUG  # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL
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

# Dense MIL
python main.py --data_dir /path/to/data --model_type dense --hidden_dims 256,128,64 --use_neptune
```

### Grid Search

To perform hyperparameter optimization:

```bash
# Use default parameter grid
python main.py --data_dir /path/to/data --model_type transformer --grid_search --use_neptune

# Use custom parameter grid
python main.py --data_dir /path/to/data --model_type transformer --grid_search --grid_search_config config.json --use_neptune

# Create a sample grid search configuration
python main.py --create_sample_grid_config
```

### Center-Based Evaluation

To analyze model performance across different centers:

```bash
python main.py --data_dir /path/to/data --model_type transformer --min_center_samples 15 --use_neptune
```

### Key Arguments

#### Data Arguments
- `--data_dir`: Path to the directory containing data files
- `--endpoint`: Endpoint to use for classification (`OS_6` or `OS_24`)
- `--oversample_factor`: Factor for oversampling minority class (default: 1.0, 0 to disable)
- `--val_size`: Proportion of data for validation (default: 0.15)
- `--test_size`: Proportion of data for testing (default: 0.15)
- `--splitted`: Use pre-split data files (train_set.pkl, val_set.pkl, test_set.pkl)

#### Model Arguments
- `--model_type`: Type of model architecture (`transformer`, `lstm`, `conv`, `lightweight_conv`, or `dense`)
- `--feature_dim`: Dimension of input features (default: 512)
- `--hidden_dim`: Hidden dimension in the model (default: 128)
- `--num_layers`: Number of transformer/LSTM layers (default: 2)
- `--dropout`: Dropout rate (default: 0.3)

#### Model-Specific Arguments
- Transformer: `--num_heads` (default: 4)
- LSTM: `--bidirectional`, `--use_attention`
- Conv/Lightweight Conv: `--num_groups` (default: 10), `--num_blocks` (default: 2), `--top-k`
- Dense: `--hidden_dims` (comma-separated list, default: '256,128,64'), `--num_groups` (default: 10), `--top-k`, `--no_batch_norm`, `--no_residual`, `--activation`

#### Training Arguments
- `--batch_size`: Batch size (default: 32)
- `--lr`: Learning rate (default: 1e-4)
- `--num_epochs`: Maximum number of training epochs (default: 100)
- `--patience`: Patience for early stopping (default: 10)
- `--selection_metric`: Metric for model selection (`f1_macro` or `val_loss`)

#### Evaluation Arguments
- `--cv_folds`: Number of folds for cross-validation (default: 1, meaning no cross-validation)
- `--bootstrap_samples`: Number of bootstrap samples for confidence intervals (default: 1000)
- `--min_center_samples`: Minimum samples per center for visualization (default: 10)

#### Other Arguments
- `--use_neptune`: Enable Neptune.ai logging
- `--output_dir`: Directory to save outputs (default: `./outputs/{model_type}`)
- `--seed`: Random seed for reproducibility (default: 42)
- `--verbose`: Enable detailed logging for debugging

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
  - Other metadata (optional): `patient_id`, `center`

Alternatively, you can use pre-split data with `--splitted` flag, providing:
- `train_set.pkl`: Training data
- `val_set.pkl`: Validation data
- `test_set.pkl`: Test data

## Output Directory Structure

When you run the framework, it creates an output directory with the following structure:

```
outputs/
    ├── {model_type}/                # Standard training output
    │   ├── model.pt                 # Trained model weights
    │   ├── metrics.json             # Evaluation metrics with confidence intervals
    │   ├── history.json             # Training history
    │   ├── center_metrics.json      # Center-specific evaluation metrics
    │   ├── predictions.csv          # Patient-level predictions
    │   ├── training_curves.png      # Training curves plot
    │   ├── confusion_matrix.png     # Confusion matrix plot
    │   ├── roc_curve.png            # ROC curve plot
    │   ├── metrics_with_ci.png      # Metrics with confidence intervals
    │   ├── dataset_comparison.png   # Comparison of metrics across datasets
    │   └── center_metrics.png       # Performance by center
    │
    ├── {model_type}_cv{n}/          # Cross-validation output
    │   ├── best_model.pt            # Best model weights
    │   ├── cv_metrics.json          # Aggregated CV metrics
    │   ├── cv_metrics.png           # CV metrics plot
    │   ├── cv_predictions.csv       # Combined predictions across folds
    │   ├── fold_1/                  # Fold 1 outputs
    │   ├── fold_2/                  # Fold 2 outputs
    │   └── ...
    │
    └── grid_search_{timestamp}/     # Grid search output
        ├── param_grid.json          # Parameter grid configuration
        ├── all_results.csv          # Results for all parameter combinations
        ├── top_k_models.csv         # Results for top-performing models
        ├── summary_report.txt       # Detailed analysis of grid search results
        ├── all_model_predictions.csv # Combined predictions from all models
        ├── visualizations/          # Grid search visualizations
        ├── combination_0001/        # Results for specific parameter combination
        ├── combination_0002/        # Results for specific parameter combination
        └── ...
```

## Evaluation Metrics

The `metrics.json` file contains comprehensive evaluation metrics:

- **Accuracy**: Overall classification accuracy (higher is better)
- **Precision**: Precision score for positive class (higher is better)
- **Recall**: Recall score for positive class (higher is better)
- **F1**: F1 score for positive class (higher is better)
- **F1 Macro**: Average F1 score across all classes (higher is better)
- **F1 Weighted**: Class-weighted F1 score (higher is better)
- **AUC**: Area Under the ROC Curve (higher is better)

Each metric includes its confidence interval (`_ci`), which represents the statistical uncertainty of the result, calculated using bootstrap resampling.

## Center-Based Evaluation

The `center_metrics.json` file and `center_metrics.png` visualization provide performance breakdowns by center:

- **Sample counts**: Number of samples from each center
- **Performance metrics**: Metrics calculated for each center
- **Weighted averages**: Overall metrics weighted by sample count

Analyze this data to:
- Identify centers where the model performs differently
- Detect potential data distribution issues
- Assess model generalization across different data sources

## Grid Search

The grid search functionality allows for systematic hyperparameter optimization:

1. Create a grid search configuration file:
   ```json
   {
     "lr": [1e-5, 5e-5, 1e-4, 5e-4],
     "dropout": [0.1, 0.3, 0.5],
     "hidden_dim": [64, 128, 256]
   }
   ```

2. Run grid search:
   ```bash
   python main.py --data_dir /path/to/data --model_type transformer --grid_search --grid_search_config config.json
   ```

3. Analyze results:
   - `top_k_models.csv`: Summary of top-performing models
   - `summary_report.txt`: Detailed analysis of grid search results
   - `visualizations/`: Visualizations of parameter effects on performance

## Neptune.ai Integration

When enabled with `--use_neptune`, the framework logs:

- Model parameters and hyperparameters
- Training and validation metrics (loss, accuracy, F1 scores)
- Learning rate changes
- Best model checkpoint
- Evaluation metrics with confidence intervals
- Visualizations (training curves, confusion matrices, ROC curves)
- Cross-validation results
- Center-based performance metrics

Access the Neptune dashboard to view and compare experiments.

## Advanced Model Configuration

### Dense MIL Model

The Dense MIL model offers high flexibility with configurable architecture and patch processing:

```bash
python main.py --data_dir /path/to/data --model_type dense \
  --hidden_dims 256,128,64 \
  --activation gelu \
  --dropout 0.3 \
  --num_groups 15 \
  --top-k
```

- `--hidden_dims`: Comma-separated list of hidden dimensions
- `--activation`: Activation function (relu, gelu, leaky_relu, tanh)
- `--no_batch_norm`: Disable batch normalization
- `--no_residual`: Disable residual connections
- `--num_groups`: Number of patch groups for aggregation
- `--top-k`: Use top-k attention-weighted patches instead of group aggregation

The `--num_groups` parameter determines how patches are grouped, while `--top-k` enables selecting the most informative patches based on attention weights. These options can significantly affect model performance depending on your data characteristics.

### LSTM MIL Model

Control the LSTM behavior with options:

```bash
python main.py --data_dir /path/to/data --model_type lstm \
  --bidirectional \
  --use_attention \
  --num_layers 2
```

### Convolutional MIL Models

Configure patch processing strategy:

```bash
python main.py --data_dir /path/to/data --model_type conv \
  --num_groups 15 \
  --top-k \
  --num_blocks 3
```

- `--num_groups`: Number of patch groups for aggregation
- `--top-k`: Use top-k attention-weighted patches
- `--num_blocks`: Number of convolutional blocks

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.