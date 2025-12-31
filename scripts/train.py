"""
Training Script for NeuroFusionXAI

Usage:
    python scripts/train.py --config configs/config.yaml --dataset ADNI
    python scripts/train.py --config configs/config.yaml --federated --num_sites 5
"""

import argparse
import os
import sys
import yaml
import random
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from models import NeuroFusionXAI, create_model
from privacy import DifferentialPrivacy, DPOptimizer, FederatedLearning, FederatedClient
from data import ADNIDataset, PPMIDataset, create_data_loaders
from utils import compute_metrics, MetricTracker


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer,
    criterion: nn.Module,
    device: torch.device,
    config: dict,
) -> dict:
    """Train for one epoch."""
    model.train()
    tracker = MetricTracker(['loss', 'accuracy'])
    
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        # Move to device
        if isinstance(inputs, dict):
            inputs = {k: v.to(device) for k, v in inputs.items()}
        else:
            inputs = {'smri': inputs.to(device)}
        labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        logits = outputs['logits']
        
        # Compute loss
        loss = criterion(logits, labels)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        if config['training'].get('grad_clip', 0) > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 
                config['training']['grad_clip']
            )
        
        optimizer.step()
        
        # Update metrics
        pred = logits.argmax(dim=-1)
        acc = (pred == labels).float().mean().item()
        tracker.update({'loss': loss.item(), 'accuracy': acc}, n=labels.size(0))
    
    return tracker.compute()


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    """Validate the model."""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    total_loss = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            if isinstance(inputs, dict):
                inputs = {k: v.to(device) for k, v in inputs.items()}
            else:
                inputs = {'smri': inputs.to(device)}
            labels = labels.to(device)
            
            outputs = model(inputs)
            logits = outputs['logits']
            probs = outputs['probabilities']
            
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)
            
            all_preds.extend(logits.argmax(dim=-1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    metrics = compute_metrics(
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs),
    )
    metrics['loss'] = total_loss / len(all_labels)
    
    return metrics


def train_standard(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: dict,
    device: torch.device,
):
    """Standard (non-federated) training."""
    # Setup
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config['training']['epochs'],
        eta_min=config['training']['scheduler'].get('min_lr', 1e-6),
    )
    
    # Apply differential privacy if enabled
    if config['privacy']['differential_privacy'].get('enabled', True):
        dp = DifferentialPrivacy(
            epsilon=config['privacy']['differential_privacy']['epsilon'],
            delta=config['privacy']['differential_privacy']['delta'],
            max_grad_norm=config['privacy']['differential_privacy']['max_grad_norm'],
            noise_multiplier=config['privacy']['differential_privacy']['noise_multiplier'],
        )
        sample_rate = config['training']['batch_size'] / len(train_loader.dataset)
        optimizer = DPOptimizer(optimizer, dp, sample_rate)
    
    # Training loop
    best_acc = 0
    checkpoint_dir = Path(config['checkpointing']['checkpoint_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(config['training']['epochs']):
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device, config)
        val_metrics = validate(model, val_loader, criterion, device)
        
        if hasattr(optimizer, 'optimizer'):
            scheduler.step()
        else:
            scheduler.step()
        
        print(f"Epoch {epoch+1}/{config['training']['epochs']}")
        print(f"  Train Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
        print(f"  Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}")
        
        # Save best model
        if val_metrics['accuracy'] > best_acc:
            best_acc = val_metrics['accuracy']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict() if not hasattr(optimizer, 'optimizer') else optimizer.optimizer.state_dict(),
                'val_metrics': val_metrics,
            }, checkpoint_dir / 'best_model.pt')
            print(f"  New best model saved! Accuracy: {best_acc:.4f}")
        
        # Periodic checkpoint
        if (epoch + 1) % config['checkpointing'].get('save_every', 10) == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
            }, checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pt')
    
    return best_acc


def train_federated(
    model: nn.Module,
    client_loaders: list,
    val_loader: DataLoader,
    config: dict,
    device: torch.device,
):
    """Federated training."""
    fed_config = config['federated']
    
    # Initialize federated learning
    fl = FederatedLearning(
        model=model,
        config=config,
        use_secure_aggregation=config['privacy']['secure_aggregation'].get('enabled', True),
        use_domain_aware=fed_config.get('domain_aware', {}).get('enabled', True),
    )
    
    # Initialize clients
    clients = fl.initialize_clients(client_loaders)
    
    # Evaluation function
    criterion = nn.CrossEntropyLoss()
    def eval_fn(model):
        return validate(model, val_loader, criterion, device)
    
    # Train
    history = fl.train(clients, eval_fn)
    
    # Save final model
    checkpoint_dir = Path(config['checkpointing']['checkpoint_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    torch.save({
        'model_state_dict': fl.get_global_model().state_dict(),
        'history': history,
    }, checkpoint_dir / 'federated_final_model.pt')
    
    return history


def main():
    parser = argparse.ArgumentParser(description='Train NeuroFusionXAI')
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--dataset', type=str, default='ADNI', choices=['ADNI', 'PPMI'])
    parser.add_argument('--federated', action='store_true')
    parser.add_argument('--num_sites', type=int, default=5)
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Set seed
    set_seed(args.seed)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Update config with command line args
    if args.data_dir:
        config['data']['root_dir'] = args.data_dir
    
    # Create model
    model = create_model(config, with_privacy=True)
    model = model.to(device)
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create data loaders
    if args.data_dir and Path(args.data_dir).exists():
        train_loader, val_loader, test_loader = create_data_loaders(config, args.dataset)
    else:
        print("Warning: Data directory not found. Using dummy data for demonstration.")
        # Create dummy data for demonstration
        from torch.utils.data import TensorDataset
        dummy_data = {
            'smri': torch.randn(100, 1, 96, 112, 96),
            'fmri': torch.randn(100, 1, 96, 112, 96),
            'pet': torch.randn(100, 1, 96, 112, 96),
        }
        dummy_labels = torch.randint(0, 4, (100,))
        
        class DummyDataset(torch.utils.data.Dataset):
            def __init__(self, data, labels):
                self.data = data
                self.labels = labels
            def __len__(self):
                return len(self.labels)
            def __getitem__(self, idx):
                return {k: v[idx] for k, v in self.data.items()}, self.labels[idx]
        
        dataset = DummyDataset(dummy_data, dummy_labels)
        train_size = int(0.7 * len(dataset))
        val_size = int(0.15 * len(dataset))
        test_size = len(dataset) - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size, test_size]
        )
        
        train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'])
        test_loader = DataLoader(test_dataset, batch_size=config['training']['batch_size'])
    
    # Train
    if args.federated:
        print(f"Starting federated training with {args.num_sites} sites...")
        # Split data across sites
        from torch.utils.data import random_split
        site_size = len(train_loader.dataset) // args.num_sites
        site_sizes = [site_size] * (args.num_sites - 1) + [len(train_loader.dataset) - site_size * (args.num_sites - 1)]
        site_datasets = random_split(train_loader.dataset, site_sizes)
        
        client_loaders = [
            DataLoader(ds, batch_size=config['training']['batch_size'], shuffle=True)
            for ds in site_datasets
        ]
        
        history = train_federated(model, client_loaders, val_loader, config, device)
    else:
        print("Starting standard training...")
        best_acc = train_standard(model, train_loader, val_loader, config, device)
        print(f"Training complete. Best validation accuracy: {best_acc:.4f}")
    
    # Final evaluation on test set
    criterion = nn.CrossEntropyLoss()
    test_metrics = validate(model, test_loader, criterion, device)
    print(f"\nTest Results:")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  F1 Score: {test_metrics['f1_score']:.4f}")
    print(f"  Sensitivity: {test_metrics['sensitivity']:.4f}")
    print(f"  Specificity: {test_metrics['specificity']:.4f}")


if __name__ == '__main__':
    main()
