import torch
import numpy as np
from collections import defaultdict

def validate(model, val_loader, criterion, device, config):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    predictions = []
    true_labels = []
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    
    with torch.no_grad():
        for _, _, inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
            
            for label, pred in zip(labels, preds):
                if label == pred:
                    class_correct[label.item()] += 1
                class_total[label.item()] += 1
    
    metrics = calculate_metrics(true_labels, predictions)
    
    for class_idx in class_total.keys():
        metrics[f'class_{class_idx}_acc'] = class_correct[class_idx] / class_total[class_idx]
    
    val_loss = running_loss / len(val_loader)
    return val_loss, metrics

def calculate_metrics(true_labels, predictions):
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
    
    metrics = {
        'accuracy': accuracy_score(true_labels, predictions),
    }
    
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average=None)
    
    for i, (p, r, f) in enumerate(zip(precision, recall, f1)):
        metrics[f'class_{i}_precision'] = p
        metrics[f'class_{i}_recall'] = r
        metrics[f'class_{i}_f1'] = f
    
    metrics['macro_precision'] = np.mean(precision)
    metrics['macro_recall'] = np.mean(recall)
    metrics['macro_f1'] = np.mean(f1)
    
    return metrics
