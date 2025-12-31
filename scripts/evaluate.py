"""
Evaluation Script for NeuroFusionXAI

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/best_model.pt --dataset ADNI
"""

import argparse
import sys
import yaml
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from models import create_model, load_model
from data import create_data_loaders
from utils import compute_metrics, plot_confusion_matrix


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def evaluate(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    class_names: list = None,
) -> dict:
    """Evaluate model on test set."""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            if isinstance(inputs, dict):
                inputs = {k: v.to(device) for k, v in inputs.items()}
            else:
                inputs = {'smri': inputs.to(device)}
            labels = labels.to(device)
            
            outputs = model(inputs)
            logits = outputs['logits']
            probs = outputs['probabilities']
            
            all_preds.extend(logits.argmax(dim=-1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    metrics = compute_metrics(
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs),
        class_names=class_names,
    )
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate NeuroFusionXAI')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--dataset', type=str, default='ADNI')
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='results')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = load_model(args.checkpoint, config, device)
    print("Model loaded successfully")
    
    # Class names
    if args.dataset == 'ADNI':
        class_names = ['CN', 'MCI', 'AD', 'Advanced AD']
    else:
        class_names = ['HC', 'PD', 'Prodromal', 'Advanced PD']
    
    # Create test loader
    if args.data_dir and Path(args.data_dir).exists():
        _, _, test_loader = create_data_loaders(config, args.dataset)
    else:
        print("Using dummy data for demonstration")
        from torch.utils.data import TensorDataset, DataLoader
        dummy_inputs = {
            'smri': torch.randn(50, 1, 96, 112, 96),
            'fmri': torch.randn(50, 1, 96, 112, 96),
            'pet': torch.randn(50, 1, 96, 112, 96),
        }
        dummy_labels = torch.randint(0, 4, (50,))
        
        class DummyDataset(torch.utils.data.Dataset):
            def __init__(self, data, labels):
                self.data = data
                self.labels = labels
            def __len__(self):
                return len(self.labels)
            def __getitem__(self, idx):
                return {k: v[idx] for k, v in self.data.items()}, self.labels[idx]
        
        test_loader = DataLoader(DummyDataset(dummy_inputs, dummy_labels), batch_size=8)
    
    # Evaluate
    metrics = evaluate(model, test_loader, device, class_names)
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Accuracy:    {metrics['accuracy']:.4f}")
    print(f"F1 Score:    {metrics['f1_score']:.4f}")
    print(f"Sensitivity: {metrics['sensitivity']:.4f}")
    print(f"Specificity: {metrics['specificity']:.4f}")
    if 'auc_roc' in metrics:
        print(f"AUC-ROC:     {metrics['auc_roc']:.4f}")
    print("="*50)
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot confusion matrix
    cm = np.array(metrics['confusion_matrix'])
    fig = plot_confusion_matrix(cm, class_names, save_path=output_dir / 'confusion_matrix.png')
    print(f"\nConfusion matrix saved to {output_dir / 'confusion_matrix.png'}")
    
    # Save metrics
    with open(output_dir / 'metrics.txt', 'w') as f:
        for key, value in metrics.items():
            if key != 'confusion_matrix':
                f.write(f"{key}: {value}\n")
    print(f"Metrics saved to {output_dir / 'metrics.txt'}")


if __name__ == '__main__':
    main()
