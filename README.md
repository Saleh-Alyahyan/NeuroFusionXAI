# NeuroFusionXAI

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

**A Privacy-Preserving Cross-Modality Explainable Fusion Framework for Early Neurodegenerative Disease Detection**

## 📋 Overview

NeuroFusionXAI is a novel privacy-preserving cross-modality explainable fusion framework designed for early neurodegenerative disease detection. The framework integrates multimodal neuroimaging data (structural MRI, functional MRI, and PET scans) through a sophisticated fusion architecture that employs federated learning principles to ensure data privacy.

### Key Features

- **Cross-Modality Fusion**: Vision Transformers and Graph Neural Networks for enhanced feature extraction from multimodal neuroimaging data
- **Privacy Preservation**: Homomorphic encryption (CKKS scheme) and differential privacy mechanisms
- **Explainable AI**: Integrated LIME, SHAP, and Grad-CAM for clinically interpretable insights
- **Federated Learning**: Domain-shift-aware federated aggregation for multi-institutional collaboration

### Performance

| Disease | Accuracy | Sensitivity | Specificity | F1-Score |
|---------|----------|-------------|-------------|----------|
| Alzheimer's Disease | 94.7% | 93.2% | 95.8% | 94.5% |
| Parkinson's Disease | 92.3% | 91.1% | 93.4% | 91.8% |
| MCI Detection | 91.3% | 89.7% | 92.1% | 90.4% |

## 🗂️ Repository Structure

```
NeuroFusionXAI/
├── README.md
├── requirements.txt
├── setup.py
├── LICENSE
├── configs/
│   └── config.yaml                    # Main configuration file
├── src/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── vision_transformer.py      # 3D Vision Transformer for neuroimaging
│   │   ├── graph_neural_network.py    # GNN for brain connectivity
│   │   ├── cross_attention_fusion.py  # Cross-modality fusion module
│   │   └── neurofusionxai.py          # Main NeuroFusionXAI model
│   ├── privacy/
│   │   ├── __init__.py
│   │   ├── differential_privacy.py    # DP mechanisms
│   │   ├── homomorphic_encryption.py  # CKKS encryption
│   │   └── federated_learning.py      # Federated learning with secure aggregation
│   ├── explainability/
│   │   ├── __init__.py
│   │   ├── lime_explainer.py          # LIME explanations
│   │   ├── shap_explainer.py          # SHAP values
│   │   └── gradcam.py                 # Grad-CAM visualizations
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py                 # Dataset classes
│   │   └── preprocessing.py           # Data preprocessing utilities
│   └── utils/
│       ├── __init__.py
│       ├── metrics.py                 # Evaluation metrics
│       └── visualization.py           # Visualization utilities
├── scripts/
│   ├── train.py                       # Training script
│   ├── evaluate.py                    # Evaluation script
│   └── inference.py                   # Inference script
├── tests/
│   └── test_models.py                 # Unit tests
└── docs/
    └── METHODOLOGY.md                 # Detailed methodology
```

## 📦 Installation

### Prerequisites

- Python 3.8+
- CUDA 11.0+ (for GPU support)
- 16GB+ GPU memory recommended

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/NeuroFusionXAI.git
cd NeuroFusionXAI

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

## 📊 Datasets

### ADNI (Alzheimer's Disease Neuroimaging Initiative)
- **Access**: https://adni.loni.usc.edu/data-samples/adni-data/
- **Subjects**: 2,847 participants
- **Modalities**: sMRI, fMRI, PET
- **Registration Required**: Yes (free for researchers)

### PPMI (Parkinson's Progression Markers Initiative)
- **Access**: https://www.ppmi-info.org/access-data-specimens/download-data/
- **Alternative**: https://www.kaggle.com/datasets/vikasukani/parkinsons-disease-data-set
- **Subjects**: 1,423 participants
- **Modalities**: sMRI, fMRI, DaTscan

### Data Preparation

After downloading the datasets, organize them as follows:

```
data/
├── ADNI/
│   ├── sMRI/
│   ├── fMRI/
│   └── PET/
├── PPMI/
│   ├── sMRI/
│   ├── fMRI/
│   └── DaTscan/
└── labels/
    ├── adni_labels.csv
    └── ppmi_labels.csv
```

## 🚀 Quick Start

### Training

```bash
# Single-site training
python scripts/train.py --config configs/config.yaml --dataset ADNI

# Federated learning across multiple sites
python scripts/train.py --config configs/config.yaml --federated --num_sites 5
```

### Evaluation

```bash
python scripts/evaluate.py --checkpoint checkpoints/best_model.pt --dataset ADNI
```

### Inference with Explainability

```bash
python scripts/inference.py --checkpoint checkpoints/best_model.pt \
    --input_smri patient_smri.nii.gz \
    --input_fmri patient_fmri.nii.gz \
    --input_pet patient_pet.nii.gz \
    --explain
```

## 🔧 Configuration

Key configuration parameters in `configs/config.yaml`:

```yaml
model:
  vit_patch_size: [16, 16, 16]
  vit_embed_dim: 768
  vit_num_heads: 12
  vit_num_layers: 12
  fusion_layers: 8
  fusion_heads: 16

privacy:
  epsilon: 0.5
  delta: 1e-5
  noise_multiplier: 1.1
  max_grad_norm: 1.0

training:
  batch_size: 16
  learning_rate: 1e-4
  epochs: 120
  optimizer: adamw
  weight_decay: 0.01
```

## 📈 Results Reproduction

To reproduce the results from the paper:

```bash
# Run full experiment pipeline
python scripts/train.py --config configs/config.yaml \
    --dataset ADNI \
    --federated \
    --num_sites 5 \
    --privacy_budget 0.5 \
    --cross_validation 5
```

## 🔬 Model Architecture

### Vision Transformer (ViT)
- **Patch Size**: 16×16×16 volumetric patches
- **Embedding Dimension**: 768
- **Transformer Blocks**: 12
- **Attention Heads**: 12

### Cross-Attention Fusion
- **Fusion Layers**: 8
- **Attention Heads**: 16
- **Hidden Dimension**: 1024

### Graph Neural Network
- **Architecture**: 3-layer Graph Attention Network (GAT)
- **Hidden Dimensions**: 256 → 128 → 64
- **Brain Regions**: Based on AAL atlas (116 regions)

### Privacy Mechanisms
- **Differential Privacy**: (ε=0.5, δ=10⁻⁵)
- **Encryption**: CKKS homomorphic encryption
- **Federated Aggregation**: Domain-shift-aware FedAvg

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@article{neurofusionxai2025,
  title={NeuroFusionXAI: A Privacy-Preserving Cross-Modality Explainable Fusion Framework for Early Neurodegenerative Disease Detection},
  author={[Authors]},
  journal={[Journal]},
  year={2025}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please read our contributing guidelines and submit pull requests.

## 📧 Contact

For questions or issues, please open a GitHub issue or contact [author email].

## 🙏 Acknowledgments

- ADNI and PPMI consortiums for providing the neuroimaging datasets
- The open-source community for the foundational libraries

---

**Disclaimer**: This framework is for research purposes only. Clinical use requires appropriate regulatory approval.
