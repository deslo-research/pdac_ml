import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
import os
from tqdm import tqdm
import pandas as pd
from collections import Counter
from typing import Dict, List, Tuple, Optional

from model import CustomModel, ModelConfig

class ModelEvaluator:
    def __init__(self, 
            model_path: str,
            model_config: ModelConfig,
            dataset,
            test_indices,
            device: torch.device,
            class_names: Optional[List[str]] = None,
            batch_size: int = 32,
            num_workers: int = 4,
            results_dir: str = 'results'):
        """Initialize the model evaluator"""
        self.model_path = model_path
        self.model_config = model_config
        self.dataset = dataset
        self.test_indices = test_indices
        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        
        self._setup_model()
        test_batch = next(iter(self._create_dataloader()))
        with torch.no_grad():
            test_output = self.model(test_batch[2].to(self.device))
        self.num_classes = test_output.shape[1]
        
        if class_names is None:
            if self.num_classes == 2:
                self.class_names = ['Negative', 'Positive']
            else:
                self.class_names = [f'Class_{i}' for i in range(self.num_classes)]
        else:
            self.class_names = class_names
        
        self._print_test_set_info()
        
    def _print_test_set_info(self):
        test_labels = [self.dataset[i][3].item() if torch.is_tensor(self.dataset[i][3]) 
                      else self.dataset[i][3] for i in self.test_indices]
        class_dist = Counter(test_labels)
        
        print("Test Set Information:")
        print(f"Total test samples: {len(self.test_indices)}")
        print("Class distribution:")
        for class_idx, count in class_dist.items():
            print(f"  {self.class_names[class_idx]}: {count} samples")
    
    def _setup_model(self):
        checkpoint = torch.load(self.model_path)
        self.model = CustomModel(self.model_config).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
    def _create_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            sampler=torch.utils.data.SubsetRandomSampler(self.test_indices),
            num_workers=self.num_workers,
            pin_memory=True
        )
        
    def _create_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            sampler=torch.utils.data.SubsetRandomSampler(self.test_indices),
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def _evaluate(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        test_loader = self._create_dataloader()
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for _, _, inputs, labels in tqdm(test_loader, desc="Evaluating"):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(inputs)
                probabilities = F.softmax(outputs, dim=1)
                
                if self.num_classes == 2:
                    preds = (probabilities[:, 1] > 0.5).long()
                else:
                    preds = outputs.argmax(1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probabilities.cpu().numpy())
        
        return (np.array(all_preds), 
                np.array(all_labels), 
                np.array(all_probs))

    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict:
        metrics = {}
        
        conf_matrix = confusion_matrix(y_true, y_pred)
        
        metrics['overall'] = {
            'accuracy': float((y_true == y_pred).mean()),
            'confusion_matrix': conf_matrix.tolist(),
            'classification_report': classification_report(y_true, y_pred, 
                target_names=self.class_names, 
                output_dict=True
            )
        }
        
        if self.num_classes == 2:
            metrics['overall']['roc_auc'] = float(roc_auc_score(y_true, y_prob[:, 1]))
        else:
            metrics['overall']['roc_auc_macro'] = float(roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro'))
        
        for class_name in metrics['overall']['classification_report']:
            if isinstance(metrics['overall']['classification_report'][class_name], dict):
                metrics['overall']['classification_report'][class_name] = {
                    k: float(v) for k, v in metrics['overall']['classification_report'][class_name].items()
                }
            else:
                metrics['overall']['classification_report'][class_name] = float(
                    metrics['overall']['classification_report'][class_name]
                )
        
        metrics['per_class'] = {}
        for i, class_name in enumerate(self.class_names):
            binary_labels = (y_true == i).astype(int)
            binary_preds = (y_pred == i).astype(int)
            binary_probs = y_prob[:, i]
            
            tn, fp, fn, tp = confusion_matrix(binary_labels, binary_preds).ravel()
            
            metrics['per_class'][class_name] = {
                'sensitivity': float(tp / (tp + fn) if (tp + fn) > 0 else 0),
                'specificity': float(tn / (tn + fp) if (tn + fp) > 0 else 0),
                'precision': float(tp / (tp + fp) if (tp + fp) > 0 else 0),
                'npv': float(tn / (tn + fn) if (tn + fn) > 0 else 0),
                'f1_score': float(metrics['overall']['classification_report'][class_name]['f1-score']),
                'roc_auc': float(roc_auc_score(binary_labels, binary_probs)),
                'samples': int(binary_labels.sum()),
                'confusion_matrix': {
                    'tn': int(tn), 'fp': int(fp),
                    'fn': int(fn), 'tp': int(tp)
                }
            }
        
        return metrics
    
    def _plot_confusion_matrix(self, conf_matrix: np.ndarray, save_path: str):
        plt.figure(figsize=(12, 10))
        sns.set_theme(font_scale=1.2)
        
        conf_matrix_percent = conf_matrix / conf_matrix.sum(axis=1)[:, np.newaxis] * 100
        
        annot = np.asarray([f'{count}\n({percent:.1f}%)'
                           for count, percent in zip(conf_matrix.flatten(), 
                                                   conf_matrix_percent.flatten())])
        annot = annot.reshape(conf_matrix.shape)
        
        ax = sns.heatmap(conf_matrix_percent, annot=annot, fmt='', cmap='Blues',
                        xticklabels=self.class_names, yticklabels=self.class_names)
        
        plt.title('Confusion Matrix\nCount and (Percentage)')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        
        plt.xticks(rotation=45)
        plt.yticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_roc_curves(self, y_true: np.ndarray, y_prob: np.ndarray, save_path: str):
        plt.figure(figsize=(12, 8))
        
        for i, class_name in enumerate(self.class_names):
            binary_labels = (y_true == i).astype(int)
            binary_probs = y_prob[:, i]
            
            fpr, tpr, _ = roc_curve(binary_labels, binary_probs)
            roc_auc = roc_auc_score(binary_labels, binary_probs)
            
            plt.plot(fpr, tpr, label=f'{class_name} (AUC = {roc_auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curves')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_precision_recall_curves(self, y_true: np.ndarray, y_prob: np.ndarray, save_path: str):
        plt.figure(figsize=(12, 8))
        
        for i, class_name in enumerate(self.class_names):
            binary_labels = (y_true == i).astype(int)
            binary_probs = y_prob[:, i]
            
            precision, recall, _ = precision_recall_curve(binary_labels, binary_probs)
            
            plt.plot(recall, precision, label=f'{class_name}')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_per_class_metrics(self, metrics: Dict, save_path: str):
        metrics_to_plot = ['sensitivity', 'specificity', 'precision', 'f1_score', 'roc_auc']
        
        data = []
        for class_name in self.class_names:
            for metric in metrics_to_plot:
                data.append({
                    'Class': class_name,
                    'Metric': metric.replace('_', ' ').title(),
                    'Value': metrics['per_class'][class_name][metric]
                })
        
        plt.figure(figsize=(15, 8))
        df = pd.DataFrame(data)
        
        ax = sns.barplot(x='Class', y='Value', hue='Metric', data=df)
        
        plt.title('Performance Metrics by Class')
        plt.xlabel('Class')
        plt.ylabel('Score')
        
        plt.xticks(rotation=45)
        
        for container in ax.containers:
            ax.bar_label(container, fmt='%.3f', padding=3)
        
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def evaluate(self) -> Dict:
        y_pred, y_true, y_prob = self._evaluate()
        
        metrics = self._calculate_metrics(y_true, y_pred, y_prob)
        
        self._plot_confusion_matrix(
            np.array(metrics['overall']['confusion_matrix']),
            os.path.join(self.results_dir, 'confusion_matrix.png')
        )
        
        self._plot_roc_curves(
            y_true, y_prob,
            os.path.join(self.results_dir, 'roc_curves.png')
        )
        
        self._plot_precision_recall_curves(
            y_true, y_prob,
            os.path.join(self.results_dir, 'precision_recall_curves.png')
        )
        
        self._plot_per_class_metrics(
            metrics,
            os.path.join(self.results_dir, 'per_class_metrics.png')
        )
        
        results = {
            'metrics': metrics,
            'predictions': y_pred.tolist(),
            'true_labels': y_true.tolist(),
            'probabilities': y_prob.tolist()
        }
        
        with open(os.path.join(self.results_dir, 'evaluation_results.json'), 'w') as f:
            json.dump(results, f, indent=4)
        
        self._print_summary(metrics)
        return results
    
    def _print_summary(self, metrics: Dict):
        """Print comprehensive evaluation summary"""
        print("="*50)
        print("EVALUATION SUMMARY")
        print("="*50)
        
        print("\nOVERALL METRICS:")
        print(f"Accuracy: {metrics['overall']['accuracy']:.4f}")
        if 'roc_auc' in metrics['overall']:
            print(f"ROC AUC: {metrics['overall']['roc_auc']:.4f}")
        else:
            print(f"ROC AUC (macro): {metrics['overall']['roc_auc_macro']:.4f}")
        
        print("\nPER-CLASS METRICS:")
        for class_name in self.class_names:
            class_metrics = metrics['per_class'][class_name]
            print(f"\n{class_name.upper()}:")
            print(f"  Samples: {class_metrics['samples']}")
            print(f"  Sensitivity: {class_metrics['sensitivity']:.4f}")
            print(f"  Specificity: {class_metrics['specificity']:.4f}")
            print(f"  Precision: {class_metrics['precision']:.4f}")
            print(f"  F1-Score: {class_metrics['f1_score']:.4f}")
            print(f"  ROC AUC: {class_metrics['roc_auc']:.4f}")
            
            cm = class_metrics['confusion_matrix']
            print(f"  True Positives: {cm['tp']}")
            print(f"  False Positives: {cm['fp']}")
            print(f"  False Negatives: {cm['fn']}")
            print(f"  True Negatives: {cm['tn']}")

def run_evaluation(
        model_path: str,
        model_config: ModelConfig,
        dataset,
        indices_path: str,  # Path to test indices
        device: torch.device,
        results_dir: str = 'results') -> Dict:
    test_indices = torch.load(indices_path)
    
    evaluator = ModelEvaluator(
        model_path=model_path,
        model_config=model_config,
        dataset=dataset,
        test_indices=test_indices,
        device=device,
        results_dir=results_dir,
        # class_names=["Fibrosis", "Cancer", "Normal"]
        class_names=["Cancer", "Normal\Fibrosis"]

    )
    
    return evaluator.evaluate()

def evaluate_with_indices(model_path, model_config, dataset, device):
    """Evaluate model using saved test indices"""
    indices_path = model_path.replace('best.pth', 'test_indices.pt')
    
    results = run_evaluation(
        model_path=model_path,
        model_config=model_config,
        dataset=dataset,
        indices_path=indices_path,
        device=device,
        results_dir='test_results'
    )
    
    return results