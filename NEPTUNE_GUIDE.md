# Neptune.ai Integration Guide

This guide explains how to use Neptune.ai for experiment tracking with the MIL framework.

## What is Neptune.ai?

Neptune.ai is a metadata store for MLOps, built for teams that run a lot of experiments. It gives you a central place to log, store, display, organize, compare, and query all your model-building metadata.

## Setup

### 1. Create a Neptune.ai Account

If you don't have a Neptune account, sign up at [neptune.ai](https://neptune.ai/).

### 2. Install Neptune

The Neptune library is included in the `requirements.txt` file. Make sure it's installed by running:

```bash
pip install -r requirements.txt
```

### 3. Configure API Key and Project

Create a `.env` file in the project root directory with your Neptune API key and project name:

```
NEPTUNE_API_KEY=your_api_key_here
NEPTUNE_PROJECT=your_workspace/your_project_name
```

You can find your API key in your Neptune account settings.

## Using Neptune.ai with MIL Framework

To enable Neptune logging, simply add the `--use_neptune` flag when running the framework:

```bash
python main.py --data_dir /path/to/data --model_type transformer --use_neptune
```

## What Gets Logged?

The framework logs the following information to Neptune:

### Parameters

- Model type and architecture parameters
- Training parameters (learning rate, batch size, etc.)
- Data parameters (endpoint, data split, etc.)

### Metrics

- Training metrics (loss, accuracy)
- Validation metrics (loss, accuracy, F1 scores)
- Test metrics (with confidence intervals)
- Cross-validation metrics (if enabled)

### Artifacts

- Training curves
- Confusion matrices
- ROC curves
- PR curves
- Attention visualizations
- Model weights

### Training Process

- Learning rate changes
- Early stopping events
- Best model information
- Training time

## Viewing Experiments

1. Go to your Neptune.ai dashboard
2. Navigate to your project
3. View the list of experiments
4. Click on an experiment to see detailed information

## Comparing Experiments

Neptune allows you to compare different experiments:

1. Select multiple experiments from the list
2. Click "Compare" to see a comparison view
3. Analyze differences in parameters, metrics, and charts

## Common Use Cases

### Hyperparameter Tuning

Run multiple experiments with different hyperparameters and compare them in Neptune:

```bash
python main.py --data_dir /path/to/data --model_type transformer --lr 0.001 --use_neptune
python main.py --data_dir /path/to/data --model_type transformer --lr 0.0001 --use_neptune
```

### Model Architecture Comparison

Compare different model architectures:

```bash
python main.py --data_dir /path/to/data --model_type transformer --use_neptune
python main.py --data_dir /path/to/data --model_type lstm --use_neptune
python main.py --data_dir /path/to/data --model_type conv --use_neptune
```

### Cross-Validation Analysis

Analyze cross-validation results:

```bash
python main.py --data_dir /path/to/data --model_type transformer --cv_folds 5 --use_neptune
```

## Advanced Usage

### Tagging Runs

You can add custom tags to your Neptune runs by modifying `neptune_utils.py`:

```python
def init_neptune_run(args, tags=None):
    # Add your custom tags to the tags list
    if not tags:
        tags = []
    tags.append("your_custom_tag")
    # ...
```

### Custom Metrics

To log custom metrics, you can use the Neptune run object:

```python
# In your code
if neptune_run:
    neptune_run["your/custom/metric"] = value
```

### Troubleshooting

If Neptune logging is not working:

1. Check that the `.env` file exists and has the correct format
2. Verify that your API key is valid
3. Ensure your internet connection is working
4. Check Neptune service status at [status.neptune.ai](https://status.neptune.ai)

## Resources

- [Neptune Documentation](https://docs.neptune.ai/)
- [Neptune Python API Reference](https://docs.neptune.ai/api-reference/neptune)
- [Neptune Community](https://community.neptune.ai/)