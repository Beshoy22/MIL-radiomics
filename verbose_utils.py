import sys
import os
import time
from functools import wraps
from contextlib import contextmanager

# Custom logging utilities for verbose mode
class VerboseLogger:
    """
    Utility class for verbose logging with formatting options
    """
    def __init__(self, verbose=False, indent_level=0):
        self.verbose = verbose
        self.indent_level = indent_level
        self._start_time = time.time()
        
    def log(self, message, indent=0, timestamp=False):
        """Log a message if verbose mode is enabled"""
        if not self.verbose:
            return
            
        # Create indentation
        indent_str = "  " * (self.indent_level + indent)
        
        # Add timestamp if requested
        if timestamp:
            elapsed = time.time() - self._start_time
            time_str = f"[{elapsed:.2f}s] "
        else:
            time_str = ""
            
        # Print the message
        print(f"{indent_str}{time_str}{message}", flush=True)
    
    def header(self, title, width=80, char='='):
        """Print a formatted header"""
        if not self.verbose:
            return
            
        print("\n" + char * width)
        print(f"{title.center(width)}")
        print(char * width + "\n", flush=True)
    
    def subheader(self, title, width=80, char='-'):
        """Print a formatted subheader"""
        if not self.verbose:
            return
            
        print("\n" + char * width)
        print(f"{title}")
        print(char * width + "\n", flush=True)
    
    def tensor_info(self, name, tensor):
        """Log information about a tensor"""
        if not self.verbose:
            return
            
        if hasattr(tensor, 'shape'):
            shape_str = f"shape={tensor.shape}"
            if hasattr(tensor, 'dtype'):
                shape_str += f", dtype={tensor.dtype}"
            if hasattr(tensor, 'device'):
                shape_str += f", device={tensor.device}"
            self.log(f"{name}: {shape_str}")
        else:
            self.log(f"{name}: {type(tensor)}")
    
    def model_info(self, model):
        """Log information about a model"""
        if not self.verbose:
            return
            
        self.log(f"Model: {type(model).__name__}")
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.log(f"Total parameters: {total_params:,}")
        self.log(f"Trainable parameters: {trainable_params:,}")
        
        # Log model summary if verbose is enabled
        self.log("Model architecture:")
        for i, (name, module) in enumerate(model.named_children()):
            self.log(f"Layer {i}: {name} - {module.__class__.__name__}", indent=1)

    @contextmanager
    def section(self, name, indent=0):
        """Context manager for logging a section with timing"""
        if not self.verbose:
            yield
            return
            
        section_start = time.time()
        self.log(f"Starting: {name}...", indent=indent)
        
        yield
        
        section_end = time.time()
        elapsed = section_end - section_start
        self.log(f"Completed: {name} in {elapsed:.4f}s", indent=indent)

# Create global logger instance
logger = VerboseLogger(verbose=False)

def set_verbose_mode(verbose):
    """Set verbose mode for the logger"""
    logger.verbose = verbose
    return logger

# Utility function to fix tqdm in grid search
def configure_tqdm_for_grid_search(is_grid_search=False):
    """
    Configure tqdm for grid search to avoid nested progress bars
    This function modifies tqdm behavior based on whether we're in grid search mode
    """
    if is_grid_search:
        # When in grid search, suppress nested tqdm progress bars
        from functools import partial
        from tqdm import tqdm
        
        # Create a dummy tqdm that doesn't actually display progress
        def dummy_tqdm(*args, **kwargs):
            if 'disable' not in kwargs:
                kwargs['disable'] = True
            return tqdm(*args, **kwargs)
        
        # Replace tqdm in specified modules
        import model_train
        model_train.tqdm = dummy_tqdm
        
        import dataloader
        if hasattr(dataloader, 'tqdm'):
            dataloader.tqdm = dummy_tqdm
            
        import metrics_with_ci
        if hasattr(metrics_with_ci, 'tqdm'):
            metrics_with_ci.tqdm = dummy_tqdm