import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix
)


def train_model(model, train_loader, val_loader, criterion, optimizer, 
                scheduler=None, num_epochs=100, early_stopping_patience=10,
                device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Train the model.
    
    Args:
        model (nn.Module): Model to train
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler (optional)
        num_epochs (int): Maximum number of epochs
        early_stopping_patience (int): Patience for early stopping
        device (str): Device to use for training
        
    Returns:
        model: Trained model
        dict: Training history
    """
    model = model.to(device)
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'val_f1_macro': [],   # Added F1 Macro tracking
        'val_f1_weighted': [] # Added F1 Weighted tracking
    }
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        # Use tqdm for progress bar
        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        
        for features, labels in train_loader_tqdm:
            features, labels = features.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Update statistics
            train_loss += loss.item() * features.size(0)
            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)
            
            # Update progress bar
            train_loader_tqdm.set_postfix(
                loss=f"{loss.item():.4f}", 
                acc=f"{100.0 * (predicted == labels).sum().item() / labels.size(0):.2f}%"
            )
        
        # Calculate epoch statistics
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = 100 * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_val_labels = []
        all_val_preds = []
        
        # Use tqdm for progress bar
        val_loader_tqdm = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
        
        with torch.no_grad():
            for features, labels in val_loader_tqdm:
                features, labels = features.to(device), labels.to(device)
                
                outputs = model(features)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * features.size(0)
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)
                
                # Collect labels and predictions for F1 calculation
                all_val_labels.extend(labels.cpu().numpy())
                all_val_preds.extend(predicted.cpu().numpy())
                
                # Update progress bar
                val_loader_tqdm.set_postfix(
                    loss=f"{loss.item():.4f}", 
                    acc=f"{100.0 * (predicted == labels).sum().item() / labels.size(0):.2f}%"
                )
        
        # Calculate validation metrics
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = 100 * val_correct / val_total
        
        # Calculate F1 scores
        val_f1_macro = f1_score(all_val_labels, all_val_preds, average='macro', zero_division=0)
        val_f1_weighted = f1_score(all_val_labels, all_val_preds, average='weighted', zero_division=0)
        
        # Update learning rate
        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        # Store statistics
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['val_f1_macro'].append(val_f1_macro)
        history['val_f1_weighted'].append(val_f1_weighted)
        
        # Calculate time elapsed
        epoch_time = time.time() - epoch_start_time
        total_time = time.time() - start_time
        
        # Print progress with F1 scores
        print(f'Epoch {epoch+1}/{num_epochs} | '
              f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | '
              f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | '
              f'F1 Macro: {val_f1_macro:.4f} | F1 Weighted: {val_f1_weighted:.4f} | '
              f'Time: {epoch_time:.1f}s | Total: {total_time/60:.1f}m')
        
        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= early_stopping_patience:
            print(f'Early stopping at epoch {epoch+1}')
            break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    total_time = time.time() - start_time
    print(f'Training completed in {total_time/60:.2f} minutes')
    
    return model, history


def evaluate_model(model, test_loader, criterion=None, 
                  train_loader=None, val_loader=None,
                  device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Evaluate the model on test data, and optionally on training and validation data.
    
    Args:
        model (nn.Module): Trained model
        test_loader (DataLoader): Test data loader
        criterion: Loss function (optional)
        train_loader (DataLoader, optional): Training data loader for reporting train metrics
        val_loader (DataLoader, optional): Validation data loader for reporting validation metrics
        device (str): Device to use for evaluation
        
    Returns:
        dict: Evaluation metrics
    """
    model = model.to(device)
    model.eval()
    
    # Function to evaluate on a specific dataloader
    def evaluate_dataloader(dataloader, name):
        loss_val = 0.0
        all_labels = []
        all_preds = []
        all_probs = []
        
        # Use tqdm for progress tracking
        dataloader_tqdm = tqdm(dataloader, desc=f"Evaluating {name}")
        
        with torch.no_grad():
            for features, labels in dataloader_tqdm:
                features, labels = features.to(device), labels.to(device)
                
                outputs = model(features)
                
                if criterion is not None:
                    loss = criterion(outputs, labels)
                    loss_val += loss.item() * features.size(0)
                    dataloader_tqdm.set_postfix(loss=f"{loss.item():.4f}")
                
                # Get predictions and probabilities
                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)
                
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of positive class
        
        # Calculate metrics
        if criterion is not None:
            loss_val = loss_val / len(dataloader.dataset)
        
        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)
        
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        f1_weighted = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        
        try:
            auc = roc_auc_score(all_labels, all_probs)
        except:
            auc = 0.0  # In case of only one class
        
        cm = confusion_matrix(all_labels, all_preds)
        
        return {
            'loss': loss_val if criterion is not None else None,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'auc': auc,
            'confusion_matrix': cm,
            'all_labels': all_labels,
            'all_preds': all_preds,
            'all_probs': all_probs
        }
    
    # Collect metrics for each dataset
    metrics = {}
    
    # Evaluate on test set
    test_metrics = evaluate_dataloader(test_loader, "test")
    metrics['test'] = test_metrics
    
    # Evaluate on training set if provided
    if train_loader:
        train_metrics = evaluate_dataloader(train_loader, "train")
        metrics['train'] = train_metrics
    
    # Evaluate on validation set if provided
    if val_loader:
        val_metrics = evaluate_dataloader(val_loader, "validation")
        metrics['val'] = val_metrics
    
    # Print comprehensive metrics for all datasets
    print("\n===== COMPREHENSIVE MODEL EVALUATION =====")
    
    datasets = [('Train', 'train'), ('Validation', 'val'), ('Test', 'test')]
    for name, key in datasets:
        if key in metrics:
            print(f"\n{name} metrics:")
            if metrics[key]['loss'] is not None:
                print(f"  Loss: {metrics[key]['loss']:.4f}")
            print(f"  Accuracy: {metrics[key]['accuracy']:.4f}")
            print(f"  Precision: {metrics[key]['precision']:.4f}")
            print(f"  Recall: {metrics[key]['recall']:.4f}")
            print(f"  F1 Score: {metrics[key]['f1']:.4f}")
            print(f"  F1 Macro: {metrics[key]['f1_macro']:.4f}")
            print(f"  F1 Weighted: {metrics[key]['f1_weighted']:.4f}")
            print(f"  AUC: {metrics[key]['auc']:.4f}")
            print(f"  Confusion Matrix:\n{metrics[key]['confusion_matrix']}")
    
    # For backward compatibility, return test metrics at the top level
    result = {
        'test_loss': test_metrics['loss'],
        'accuracy': test_metrics['accuracy'],
        'precision': test_metrics['precision'],
        'recall': test_metrics['recall'],
        'f1': test_metrics['f1'],
        'f1_macro': test_metrics['f1_macro'],
        'f1_weighted': test_metrics['f1_weighted'],
        'auc': test_metrics['auc'],
        'confusion_matrix': test_metrics['confusion_matrix'],
        'all_labels': test_metrics['all_labels'],
        'all_preds': test_metrics['all_preds'],
        'all_probs': test_metrics['all_probs'],
        'all_datasets': metrics  # Include metrics for all datasets
    }
    
    return result


def predict(model, dataloader, return_attention=False, 
           device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Make predictions using a trained model.
    
    Args:
        model (nn.Module): Trained model
        dataloader (DataLoader): DataLoader with data
        return_attention (bool): Whether to return attention weights
        device (str): Device to use for inference
        
    Returns:
        tuple: (labels, predictions, probabilities, attention_weights) if return_attention=True
               (labels, predictions, probabilities) otherwise
    """
    model = model.to(device)
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []
    all_attns = []
    
    # Use tqdm for progress tracking
    dataloader_tqdm = tqdm(dataloader, desc="Predicting")
    
    with torch.no_grad():
        for features, labels in dataloader_tqdm:
            features, labels = features.to(device), labels.to(device)
            
            # Forward pass with or without attention weights
            if return_attention and hasattr(model, 'forward') and 'return_attn' in model.forward.__code__.co_varnames:
                outputs, attn_weights = model(features, return_attn=True)
                all_attns.extend(attn_weights.cpu().numpy())
            else:
                outputs = model(features)
            
            # Get predictions and probabilities
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of positive class
    
    if return_attention:
        return np.array(all_labels), np.array(all_preds), np.array(all_probs), all_attns
    else:
        return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def setup_training(model, learning_rate=1e-4, weight_decay=1e-4, 
                  class_weights=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Set up loss function, optimizer, and scheduler for training.
    
    Args:
        model (nn.Module): Model to train
        learning_rate (float): Learning rate
        weight_decay (float): Weight decay for regularization
        class_weights (torch.Tensor): Class weights for loss function
        device (str): Device to use
        
    Returns:
        tuple: (criterion, optimizer, scheduler)
    """
    # Move class weights to device if provided
    if class_weights is not None:
        class_weights = class_weights.to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Create scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )
    
    return criterion, optimizer, scheduler