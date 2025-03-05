import os
import neptune
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from io import BytesIO
import torch
import numpy as np

def load_neptune_config():
    """
    Load Neptune configuration from .env file
    
    Returns:
        dict: Neptune configuration with api_key and project
    """
    load_dotenv()
    
    api_key = os.getenv('NEPTUNE_API_KEY')
    project = os.getenv('NEPTUNE_PROJECT')
    
    if not api_key or not project:
        print("Warning: Neptune API key or project not found in .env file")
        return None
    
    return {
        'api_key': api_key,
        'project': project
    }

def init_neptune_run(args, tags=None):
    """
    Initialize Neptune run
    
    Args:
        args: Command line arguments
        tags: List of tags to add to the run
        
    Returns:
        neptune.Run: Neptune run object or None if Neptune is not configured
    """
    config = load_neptune_config()
    if not config:
        return None
    
    if not tags:
        tags = []
    
    # Add model type and endpoint as tags
    if hasattr(args, 'model_type'):
        tags.append(args.model_type)
    if hasattr(args, 'endpoint'):
        tags.append(args.endpoint)
    
    # Initialize Neptune run
    run = neptune.init_run(
        project=config['project'],
        api_token=config['api_key'],
        tags=tags
    )
    
    # Log parameters
    params = vars(args)
    run["parameters"] = params
    
    return run

def log_figure(run, fig, name):
    """
    Log a matplotlib figure to Neptune
    
    Args:
        run: Neptune run object
        fig: Matplotlib figure
        name: Name of the figure
    """
    if run is None:
        return
    
    # Save figure to BytesIO
    buffer = BytesIO()
    fig.savefig(buffer, format='png')
    buffer.seek(0)
    
    # Log figure to Neptune
    run[f"visualizations/{name}"].upload(buffer)

def log_model(run, model, name="model"):
    """
    Log a model to Neptune
    
    Args:
        run: Neptune run object
        model: PyTorch model
        name: Name of the model
    """
    if run is None:
        return
    
    # Save model to file
    model_path = f"{name}.pt"
    torch.save(model.state_dict(), model_path)
    
    # Log model to Neptune
    run[f"models/{name}"].upload(model_path)
    
    # Remove temporary file
    os.remove(model_path)

def log_confusion_matrix(run, cm, name="confusion_matrix"):
    """
    Log confusion matrix to Neptune
    
    Args:
        run: Neptune run object
        cm: Confusion matrix (numpy array)
        name: Name for the confusion matrix
    """
    if run is None:
        return
    
    # Convert to list if numpy array
    if isinstance(cm, np.ndarray):
        cm = cm.tolist()
        
    # Log confusion matrix
    run[f"evaluation/{name}"] = cm