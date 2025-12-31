import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, auc, roc_curve
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
torch.manual_seed(42)

class MedicalImagingDataset(Dataset):
    def __init__(self, X, y, transform=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.transform = transform
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        if self.transform:
            x = self.transform(x)
        return x, y

class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim):
        super(AttentionLayer, self).__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.scale = np.sqrt(hidden_dim)
    
    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attention_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights

class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads=4):
        super(TransformerBlock, self).__init__()
        self.attention = AttentionLayer(hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(0.1)
        )
    
    def forward(self, x):
        attn_output, attn_weights = self.attention(x)
        x = self.norm1(x + attn_output)
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        return x, attn_weights

class NeuroFusionXAI(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_classes=2, num_transformer_blocks=3):
        super(NeuroFusionXAI, self).__init__()
        
        self.feature_extraction = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(hidden_dim) for _ in range(num_transformer_blocks)
        ])
        
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim // 2, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )
        
        self.attention_weights_storage = []
        self.feature_importance_scores = None
    
    def forward(self, x, return_attention=False):
        features = self.feature_extraction(x)
        
        attention_weights_list = []
        for transformer_block in self.transformer_blocks:
            features, attn_weights = transformer_block(features)
            attention_weights_list.append(attn_weights)
        
        if return_attention:
            self.attention_weights_storage = attention_weights_list
        
        fused_features = self.fusion_layer(features)
        logits = self.classifier(fused_features)
        
        return logits, features if not return_attention else (logits, features, attention_weights_list)

class ExplainabilityModule:
    def __init__(self, model, device):
        self.model = model
        self.device = device
    
    def compute_gradients(self, x, y):
        x.requires_grad = True
        logits, _ = self.model(x)
        loss = nn.CrossEntropyLoss()(logits, y)
        loss.backward()
        gradients = x.grad.abs().mean(dim=0).detach().cpu().numpy()
        return gradients
    
    def compute_feature_importance(self, X):
        importance_scores = np.zeros(X.shape[1])
        self.model.eval()
        
        with torch.no_grad():
            for i in range(X.shape[1]):
                X_perturbed = X.copy()
                X_perturbed[:, i] = np.random.permutation(X_perturbed[:, i])
                
                X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
                X_perturbed_tensor = torch.tensor(X_perturbed, dtype=torch.float32).to(self.device)
                
                logits_original, _ = self.model(X_tensor)
                logits_perturbed, _ = self.model(X_perturbed_tensor)
                
                importance_scores[i] = torch.mean((logits_original - logits_perturbed) ** 2).cpu().numpy()
        
        return importance_scores / importance_scores.sum()
    
    def compute_saliency_maps(self, X):
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        X_tensor.requires_grad = True
        
        logits, _ = self.model(X_tensor)
        gradients_list = []
        
        for class_idx in range(logits.shape[1]):
            if X_tensor.grad is not None:
                X_tensor.grad.zero_()
            
            logits[0, class_idx].backward(retain_graph=True)
            gradient = X_tensor.grad.abs().detach().cpu().numpy()
            gradients_list.append(gradient)
        
        return np.array(gradients_list)

def generate_synthetic_medical_dataset(n_samples=1000, n_features=128):
    np.random.seed(42)
    
    X_healthy = np.random.normal(loc=0, scale=1, size=(n_samples//2, n_features))
    y_healthy = np.zeros(n_samples//2)
    
    X_disease = np.random.normal(loc=0.5, scale=1.2, size=(n_samples//2, n_features))
    y_disease = np.ones(n_samples//2)
    
    X = np.vstack([X_healthy, X_disease])
    y = np.hstack([y_healthy, y_disease])
    
    shuffle_idx = np.random.permutation(len(X))
    X = X[shuffle_idx]
    y = y[shuffle_idx]
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    return X, y.astype(int), scaler

def split_data(X, y, train_size=0.7, val_size=0.15):
    n = len(X)
    train_idx = int(n * train_size)
    val_idx = int(n * (train_size + val_size))
    
    X_train, y_train = X[:train_idx], y[:train_idx]
    X_val, y_val = X[train_idx:val_idx], y[train_idx:val_idx]
    X_test, y_test = X[val_idx:], y[val_idx:]
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def train_epoch(model, train_loader, optimizer, criterion, device, regularization=1e-5):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        logits, features = model(X_batch)
        loss = criterion(logits, y_batch)
        
        l2_reg = sum(p.pow(2.0).sum() for p in model.parameters())
        loss = loss + regularization * l2_reg
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(logits, 1)
        correct += (predicted == y_batch).sum().item()
        total += y_batch.size(0)
    
    return total_loss / len(train_loader), correct / total

def validate_epoch(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_predictions = []
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            logits, _ = model(X_batch)
            loss = criterion(logits, y_batch)
            total_loss += loss.item()
            
            _, predicted = torch.max(logits, 1)
            probs = torch.softmax(logits, dim=1)[:, 1]
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(y_batch.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)
    
    return total_loss / len(val_loader), correct / total, all_predictions, all_targets, all_probs

def evaluate_model(model, X_test, y_test, criterion, device):
    model.eval()
    
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)
    
    with torch.no_grad():
        logits, _ = model(X_test_tensor)
        test_loss = criterion(logits, y_test_tensor).item()
        
        probs = torch.softmax(logits, dim=1)
        predictions = torch.argmax(logits, dim=1).cpu().numpy()
        probs_np = probs.cpu().numpy()
    
    accuracy = np.mean(predictions == y_test)
    
    conf_matrix = confusion_matrix(y_test, predictions)
    class_report = classification_report(y_test, predictions, output_dict=True)
    
    roc_auc = roc_auc_score(y_test, probs_np[:, 1])
    fpr, tpr, _ = roc_curve(y_test, probs_np[:, 1])
    
    sensitivity = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[1, 0]) if (conf_matrix[1, 1] + conf_matrix[1, 0]) > 0 else 0
    specificity = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1]) if (conf_matrix[0, 0] + conf_matrix[0, 1]) > 0 else 0
    
    return {
        'loss': test_loss,
        'accuracy': accuracy,
        'auc': roc_auc,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'conf_matrix': conf_matrix,
        'class_report': class_report,
        'fpr': fpr,
        'tpr': tpr,
        'predictions': predictions,
        'probabilities': probs_np
    }

def plot_training_history(train_losses, val_losses, train_accs, val_accs):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
    
    ax1.plot(train_losses, label='Train Loss', linewidth=2)
    ax1.plot(val_losses, label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(train_accs, label='Train Accuracy', linewidth=2)
    ax2.plot(val_accs, label='Val Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/training_history.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix(conf_matrix, title='Confusion Matrix'):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax, cbar_kws={'label': 'Count'})
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_roc_curve(fpr, tpr, auc_score):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.4f})', linewidth=2)
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/roc_curve.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_feature_importance(importance_scores, top_k=20):
    top_indices = np.argsort(importance_scores)[-top_k:][::-1]
    top_scores = importance_scores[top_indices]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(range(len(top_scores)), top_scores, color='steelblue')
    ax.set_yticks(range(len(top_scores)))
    ax.set_yticklabels([f'Feature {i}' for i in top_indices])
    ax.set_xlabel('Importance Score')
    ax.set_title(f'Top {top_k} Feature Importance Scores')
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_comprehensive_report(train_metrics, val_metrics, test_metrics, explainability_metrics):
    report = []
    report.append("="*80)
    report.append("NEUROFUSIONXAI - COMPREHENSIVE EVALUATION REPORT")
    report.append("="*80)
    report.append("")
    
    report.append("TRAINING METRICS")
    report.append("-"*80)
    report.append(f"Final Train Loss: {train_metrics['losses'][-1]:.6f}")
    report.append(f"Final Train Accuracy: {train_metrics['accuracies'][-1]:.4f}")
    report.append("")
    
    report.append("VALIDATION METRICS")
    report.append("-"*80)
    report.append(f"Final Val Loss: {val_metrics['losses'][-1]:.6f}")
    report.append(f"Final Val Accuracy: {val_metrics['accuracies'][-1]:.4f}")
    report.append("")
    
    report.append("TEST METRICS")
    report.append("-"*80)
    report.append(f"Test Loss: {test_metrics['loss']:.6f}")
    report.append(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    report.append(f"AUC-ROC Score: {test_metrics['auc']:.4f}")
    report.append(f"Sensitivity: {test_metrics['sensitivity']:.4f}")
    report.append(f"Specificity: {test_metrics['specificity']:.4f}")
    report.append("")
    
    report.append("CONFUSION MATRIX")
    report.append("-"*80)
    report.append(str(test_metrics['conf_matrix']))
    report.append("")
    
    report.append("CLASSIFICATION REPORT")
    report.append("-"*80)
    for key, val in test_metrics['class_report'].items():
        if isinstance(val, dict):
            report.append(f"{key}:")
            for metric, score in val.items():
                report.append(f"  {metric}: {score:.4f}")
        else:
            report.append(f"{key}: {val:.4f}")
    report.append("")
    
    report.append("EXPLAINABILITY ANALYSIS")
    report.append("-"*80)
    report.append(f"Mean Feature Importance: {explainability_metrics['mean_importance']:.6f}")
    report.append(f"Std Feature Importance: {explainability_metrics['std_importance']:.6f}")
    report.append(f"Max Feature Importance: {explainability_metrics['max_importance']:.6f}")
    report.append("")
    
    report.append("="*80)
    
    return "\n".join(report)

def main():
    print("Generating synthetic medical imaging dataset...")
    X, y, scaler = generate_synthetic_medical_dataset(n_samples=2000, n_features=128)
    
    print("Splitting data into train, validation, and test sets...")
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_data(X, y)
    
    print(f"Train set size: {X_train.shape}")
    print(f"Validation set size: {X_val.shape}")
    print(f"Test set size: {X_test.shape}")
    print("")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    train_dataset = MedicalImagingDataset(X_train, y_train)
    val_dataset = MedicalImagingDataset(X_val, y_val)
    test_dataset = MedicalImagingDataset(X_test, y_test)
    
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print("Initializing NeuroFusionXAI model...")
    model = NeuroFusionXAI(input_dim=128, hidden_dim=128, num_classes=2, num_transformer_blocks=3).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    criterion = nn.CrossEntropyLoss()
    
    print("Starting model training...")
    num_epochs = 100
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_acc = 0
    patience = 15
    patience_counter = 0
    
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device, regularization=1e-5)
        val_loss, val_acc, _, _, _ = validate_epoch(model, val_loader, criterion, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        scheduler.step()
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), '/mnt/user-data/outputs/best_model.pth')
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    print("Loading best model...")
    model.load_state_dict(torch.load('/mnt/user-data/outputs/best_model.pth'))
    
    print("Evaluating model on test set...")
    test_results = evaluate_model(model, X_test, y_test, criterion, device)
    
    print("\nTest Set Results:")
    print(f"Accuracy: {test_results['accuracy']:.4f}")
    print(f"AUC-ROC: {test_results['auc']:.4f}")
    print(f"Sensitivity: {test_results['sensitivity']:.4f}")
    print(f"Specificity: {test_results['specificity']:.4f}")
    
    print("\nComputing explainability metrics...")
    explainability = ExplainabilityModule(model, device)
    feature_importance = explainability.compute_feature_importance(X_test)
    
    explainability_metrics = {
        'mean_importance': feature_importance.mean(),
        'std_importance': feature_importance.std(),
        'max_importance': feature_importance.max(),
        'feature_importance': feature_importance
    }
    
    print("\nGenerating visualizations...")
    plot_training_history(train_losses, val_losses, train_accs, val_accs)
    plot_confusion_matrix(test_results['conf_matrix'])
    plot_roc_curve(test_results['fpr'], test_results['tpr'], test_results['auc'])
    plot_feature_importance(feature_importance, top_k=20)
    
    train_metrics = {'losses': train_losses, 'accuracies': train_accs}
    val_metrics = {'losses': val_losses, 'accuracies': val_accs}
    
    report = generate_comprehensive_report(train_metrics, val_metrics, test_results, explainability_metrics)
    print("\n" + report)
    
    with open('/mnt/user-data/outputs/evaluation_report.txt', 'w') as f:
        f.write(report)
    
    results_df = pd.DataFrame({
        'Metric': ['Accuracy', 'AUC-ROC', 'Sensitivity', 'Specificity', 'Test Loss'],
        'Score': [test_results['accuracy'], test_results['auc'], test_results['sensitivity'], test_results['specificity'], test_results['loss']]
    })
    
    results_df.to_csv('/mnt/user-data/outputs/test_results.csv', index=False)
    
    feature_importance_df = pd.DataFrame({
        'Feature': [f'Feature_{i}' for i in range(len(feature_importance))],
        'Importance': feature_importance
    }).sort_values('Importance', ascending=False)
    
    feature_importance_df.to_csv('/mnt/user-data/outputs/feature_importance.csv', index=False)
    
    print("\nAll results saved to /mnt/user-data/outputs/")
    print("Files generated:")
    print("  - best_model.pth")
    print("  - training_history.png")
    print("  - confusion_matrix.png")
    print("  - roc_curve.png")
    print("  - feature_importance.png")
    print("  - evaluation_report.txt")
    print("  - test_results.csv")
    print("  - feature_importance.csv")

if __name__ == "__main__":
    main()
