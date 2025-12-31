"""
Evaluation Metrics for NeuroFusionXAI

Provides comprehensive metrics for classification performance evaluation.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score as sklearn_f1,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    average_precision_score,
)


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute accuracy."""
    return accuracy_score(y_true, y_pred)


def sensitivity(y_true: np.ndarray, y_pred: np.ndarray, average: str = 'macro') -> float:
    """Compute sensitivity (recall)."""
    return recall_score(y_true, y_pred, average=average, zero_division=0)


def specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute specificity for binary classification."""
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        return tn / (tn + fp + 1e-8)
    else:
        # Multi-class: compute per-class and average
        specs = []
        for i in range(cm.shape[0]):
            tn = cm.sum() - cm[i, :].sum() - cm[:, i].sum() + cm[i, i]
            fp = cm[:, i].sum() - cm[i, i]
            specs.append(tn / (tn + fp + 1e-8))
        return np.mean(specs)


def f1_score(y_true: np.ndarray, y_pred: np.ndarray, average: str = 'macro') -> float:
    """Compute F1 score."""
    return sklearn_f1(y_true, y_pred, average=average, zero_division=0)


def auc_roc(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    multi_class: str = 'ovr',
) -> float:
    """Compute AUC-ROC score."""
    try:
        if len(y_prob.shape) == 1 or y_prob.shape[1] == 2:
            if len(y_prob.shape) == 2:
                y_prob = y_prob[:, 1]
            return roc_auc_score(y_true, y_prob)
        else:
            return roc_auc_score(y_true, y_prob, multi_class=multi_class)
    except ValueError:
        return 0.0


def auc_pr(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute AUC-PR (Average Precision) score."""
    try:
        if len(y_prob.shape) == 2:
            y_prob = y_prob[:, 1] if y_prob.shape[1] == 2 else y_prob.max(axis=1)
        return average_precision_score(y_true, y_prob)
    except ValueError:
        return 0.0


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Compute comprehensive classification metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_prob: Prediction probabilities (optional)
        class_names: Names of classes (optional)
    
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy(y_true, y_pred),
        'sensitivity': sensitivity(y_true, y_pred),
        'specificity': specificity(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
    }
    
    if y_prob is not None:
        metrics['auc_roc'] = auc_roc(y_true, y_prob)
        metrics['auc_pr'] = auc_pr(y_true, y_prob)
    
    # Confusion matrix
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()
    
    # Per-class metrics
    if class_names is not None:
        report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
        for class_name in class_names:
            if class_name in report:
                metrics[f'{class_name}_precision'] = report[class_name]['precision']
                metrics[f'{class_name}_recall'] = report[class_name]['recall']
                metrics[f'{class_name}_f1'] = report[class_name]['f1-score']
    
    return metrics


def compute_privacy_metrics(
    attack_results: Dict[str, np.ndarray],
) -> Dict[str, float]:
    """
    Compute privacy-related metrics.
    
    Args:
        attack_results: Results from privacy attacks
    
    Returns:
        Dictionary of privacy metrics
    """
    metrics = {}
    
    for attack_name, results in attack_results.items():
        if 'success_rate' in results:
            metrics[f'{attack_name}_success_rate'] = results['success_rate']
        if 'auc' in results:
            metrics[f'{attack_name}_auc'] = results['auc']
    
    return metrics


class MetricTracker:
    """
    Track metrics during training.
    
    Args:
        metrics: List of metric names to track
    """
    
    def __init__(self, metrics: List[str]):
        self.metrics = metrics
        self.history = {m: [] for m in metrics}
        self.current = {m: 0.0 for m in metrics}
        self.count = 0
    
    def update(self, values: Dict[str, float], n: int = 1):
        """Update metrics with new values."""
        for metric, value in values.items():
            if metric in self.current:
                self.current[metric] += value * n
        self.count += n
    
    def compute(self) -> Dict[str, float]:
        """Compute average metrics."""
        return {m: self.current[m] / max(1, self.count) for m in self.metrics}
    
    def reset(self):
        """Reset metrics for new epoch."""
        for metric in self.current:
            self.history[metric].append(self.current[metric] / max(1, self.count))
            self.current[metric] = 0.0
        self.count = 0
    
    def get_best(self, metric: str, mode: str = 'max') -> Tuple[float, int]:
        """Get best value and epoch for a metric."""
        values = self.history[metric]
        if not values:
            return 0.0, 0
        if mode == 'max':
            best_idx = np.argmax(values)
        else:
            best_idx = np.argmin(values)
        return values[best_idx], best_idx
