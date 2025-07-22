import os
import numpy as np
from dataclasses import dataclass
from sklearn.model_selection import KFold
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler, WeightedRandomSampler
from tqdm import tqdm

from dataset import create_data_splits
from metrics_validation import calculate_metrics, validate

@dataclass
class TrainingConfig:
    """Configuration class for training hyperparameters"""
    batch_size: int = 16
    num_epochs: int = 100
    base_lr: float = 1e-4
    weight_decay: float = 0.01
    label_smoothing: float = 0.1
    
    warmup_epochs: int = 5
    min_lr: float = 1e-6
    
    patience: int = 15
    num_folds: int = 5
    test_size: float = 0.2
    seed: int = 42
    
    # Augmentation and regularization
    mixup_alpha: float = 0.2
    grad_clip: float = 1.0
    aug_strength: float = 0.5  # Controls strength of augmentations
    
    optimizer: str = 'adamw'  # ['adamw', 'adam', 'sgd']
    scheduler: str = 'cosine'  # ['cosine', 'step', 'plateau']
    backbone_lr_factor: float = 0.1  # Backbone learning rate = base_lr * backbone_lr_factor


def mixup_data(x, y, alpha=0.2):
    """Performs mixup on the input batch and returns mixed inputs and targets"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup loss function"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train_epoch(model, train_loader, criterion, optimizer, device, config):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    predictions = []
    true_labels = []
    
    pbar = tqdm(train_loader, desc='Training')
    for _, _, inputs, labels in pbar:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        if config.mixup_alpha > 0:
            inputs, labels_a, labels_b, lam = mixup_data(inputs, labels, config.mixup_alpha)
            
        optimizer.zero_grad()
        outputs = model(inputs)
        
        if config.mixup_alpha > 0:
            loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
        else:
            loss = criterion(outputs, labels)
            
        loss.backward()
        
        if config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            
        optimizer.step()
        
        running_loss += loss.item()
        
        if config.mixup_alpha == 0:
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
            
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    epoch_loss = running_loss / len(train_loader)
    if config.mixup_alpha == 0:
        metrics = calculate_metrics(true_labels, predictions)
    else:
        metrics = {}
    
    return epoch_loss, metrics


def train_model(model, dataset, config, device, model_save_path):
    """Complete training pipeline with k-fold cross-validation"""
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    train_val_indices, test_indices = create_data_splits(dataset, config)
    
    test_indices_path = f"{model_save_path}_test_indices.pt"
    torch.save(test_indices, test_indices_path)
    
    labels = [dataset[i][3].item() if torch.is_tensor(dataset[i][3]) 
             else dataset[i][3] for i in train_val_indices]
    class_counts = Counter(labels)
    total_samples = len(labels)
    class_weights = {class_idx: total_samples / count 
                    for class_idx, count in class_counts.items()}

    print("Class weights:", class_weights)
    
    kfold = KFold(n_splits=config.num_folds, shuffle=True, random_state=config.seed)
    fold_results = []
    best_overall_metrics = {'val_acc': 0.0}
    best_overall_model = None
    best_overall_fold = 0
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(train_val_indices)):
        print(f'\nFold {fold+1}/{config.num_folds}')
        print('-' * 40)
        
        train_indices = [train_val_indices[i] for i in train_idx]
        val_indices = [train_val_indices[i] for i in val_idx]
        
        train_labels = [dataset[i][3].item() if torch.is_tensor(dataset[i][3]) 
                       else dataset[i][3] for i in train_indices]
        sample_weights = [class_weights[label] for label in train_labels]
        
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_indices),
            replacement=True
        )
        
        train_loader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            sampler=train_sampler,
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            sampler=SubsetRandomSampler(val_indices),
            num_workers=4,
            pin_memory=True
        )
        
        model = model.to(device)
        class_weights_tensor = torch.FloatTensor([class_weights[i] for i in range(model.config.num_classes)]).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=config.label_smoothing, reduction="mean")
        
        params = [
            {'params': model.features.parameters(), 'lr': config.base_lr},
            {'params': model.classifier.parameters(), 'lr': config.base_lr * 10}
        ]
        optimizer = optim.AdamW(params, weight_decay=config.weight_decay)
        
        def lr_lambda(epoch):
            if epoch < config.warmup_epochs:
                return epoch / config.warmup_epochs
            return 0.5 * (1 + np.cos((epoch - config.warmup_epochs) * np.pi / 
                                   (config.num_epochs - config.warmup_epochs)))
            
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        best_val_metrics = {'val_acc': 0.0}
        best_model_state = None
        plateau_counter = 0
        
        for epoch in range(config.num_epochs):
            train_loss, train_metrics = train_epoch(
                model, train_loader, criterion, optimizer, device, config
            )
            
            val_loss, val_metrics = validate(
                model, val_loader, criterion, device, config
            )
            
            scheduler.step()
            
            print(f'Epoch {epoch+1}/{config.num_epochs}:')
            print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
            print('Validation Metrics:', {k: f'{v:.4f}' for k, v in val_metrics.items()})
            
            if val_metrics['accuracy'] > best_val_metrics['val_acc']:
                best_val_metrics = val_metrics
                best_val_metrics['val_acc'] = val_metrics['accuracy']
                best_model_state = model.state_dict().copy()
                plateau_counter = 0
                
                fold_save_path = f"{model_save_path}_fold{fold+1}.pth"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'metrics': val_metrics,
                    'class_weights': class_weights,
                    'fold': fold + 1,
                    'train_indices': train_indices,
                    'val_indices': val_indices
                }, fold_save_path)
                
                if val_metrics['accuracy'] > best_overall_metrics['val_acc']:
                    best_overall_metrics = val_metrics
                    best_overall_model = model.state_dict().copy()
                    best_overall_fold = fold + 1
            else:
                plateau_counter += 1
            
            if plateau_counter >= config.patience:
                print(f'Early stopping triggered after epoch {epoch+1}')
                break
        
        fold_results.append({
            'fold': fold + 1,
            'metrics': best_val_metrics,
            'model_path': f"{model_save_path}_fold{fold+1}.pth"
        })
    
    best_model_path = f"{model_save_path}_best.pth"
    torch.save({
        'model_state_dict': best_overall_model,
        'metrics': best_overall_metrics,
        'fold': best_overall_fold,
        'class_weights': class_weights,
        'test_indices': test_indices
    }, best_model_path)
    
    print("Training Complete!")
    print("Fold Results:")
    for result in fold_results:
        print(f"Fold {result['fold']}:")
        print("Metrics:", {k: f"{v:.4f}" for k, v in result['metrics'].items()})
        print(f"Model saved at: {result['model_path']}")
    
    print(f'Best Overall Model:')
    print(f'Fold: {best_overall_fold}')
    print("Metrics:", {k: f"{v:.4f}" for k, v in best_overall_metrics.items()})
    print(f'Saved at: {best_model_path}')
    
    return {
        'fold_results': fold_results,
        'best_model_path': best_model_path,
        'test_indices_path': test_indices_path,
        'best_fold': best_overall_fold,
        'best_metrics': best_overall_metrics
    }