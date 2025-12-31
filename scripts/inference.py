"""
Inference Script for NeuroFusionXAI

Usage:
    python scripts/inference.py --checkpoint checkpoints/best_model.pt \
        --input_smri patient_smri.nii.gz \
        --input_fmri patient_fmri.nii.gz \
        --input_pet patient_pet.nii.gz \
        --explain
"""

import argparse
import sys
import yaml
from pathlib import Path

import numpy as np
import torch
import nibabel as nib

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from models import load_model
from data.preprocessing import preprocess_volume
from explainability import LIMEExplainer, SHAPExplainer, GradCAM3D
from utils import plot_gradcam, plot_shap_values


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_inputs(args, config) -> dict:
    """Load and preprocess input volumes."""
    inputs = {}
    target_shape = tuple(config['data']['input_shape'][1:])
    
    if args.input_smri and Path(args.input_smri).exists():
        volume = preprocess_volume(args.input_smri, target_shape)
        inputs['smri'] = torch.from_numpy(volume).float().unsqueeze(0).unsqueeze(0)
    
    if args.input_fmri and Path(args.input_fmri).exists():
        volume = preprocess_volume(args.input_fmri, target_shape)
        inputs['fmri'] = torch.from_numpy(volume).float().unsqueeze(0).unsqueeze(0)
    
    if args.input_pet and Path(args.input_pet).exists():
        volume = preprocess_volume(args.input_pet, target_shape)
        inputs['pet'] = torch.from_numpy(volume).float().unsqueeze(0).unsqueeze(0)
    
    return inputs


def run_inference(model, inputs, device, class_names):
    """Run inference and return predictions."""
    model.eval()
    inputs_device = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(inputs_device)
        probs = outputs['probabilities'][0].cpu().numpy()
        pred_class = probs.argmax()
    
    return {
        'predicted_class': class_names[pred_class],
        'confidence': probs[pred_class],
        'probabilities': {name: float(probs[i]) for i, name in enumerate(class_names)},
    }


def generate_explanations(model, inputs, device, output_dir):
    """Generate explainability outputs."""
    inputs_device = {k: v.to(device) for k, v in inputs.items()}
    
    # Grad-CAM
    print("Generating Grad-CAM explanations...")
    target_layers = ['vit_encoders.smri.blocks.11']  # Last transformer block
    gradcam = GradCAM3D(model, target_layers)
    heatmaps = gradcam.generate(inputs_device)
    
    # Save Grad-CAM visualization
    if 'smri' in inputs:
        volume = inputs['smri'].squeeze().numpy()
        for layer_name, heatmap in heatmaps.items():
            upsampled = gradcam.upsample_to_input(heatmap, volume.shape)
            fig = plot_gradcam(volume, upsampled, save_path=output_dir / f'gradcam_{layer_name.replace(".", "_")}.png')
            print(f"  Saved Grad-CAM to {output_dir / f'gradcam_{layer_name.replace('.', '_')}.png'}")
    
    # LIME
    print("Generating LIME explanations...")
    lime_explainer = LIMEExplainer(model, num_samples=500)
    lime_results = lime_explainer.explain_multimodal(inputs_device)
    
    for modality, result in lime_results.items():
        importance_map = lime_explainer.get_explanation_map(result)
        np.save(output_dir / f'lime_{modality}.npy', importance_map)
        print(f"  Saved LIME map for {modality}")
    
    # SHAP
    print("Generating SHAP explanations...")
    shap_explainer = SHAPExplainer(model, max_evals=200)
    shap_results = shap_explainer.explain(inputs_device)
    
    fig = plot_shap_values(shap_results['shap_values'], save_path=output_dir / 'shap_values.png')
    print(f"  Saved SHAP values to {output_dir / 'shap_values.png'}")
    
    return {
        'gradcam': heatmaps,
        'lime': lime_results,
        'shap': shap_results,
    }


def main():
    parser = argparse.ArgumentParser(description='Run inference with NeuroFusionXAI')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--input_smri', type=str, default=None)
    parser.add_argument('--input_fmri', type=str, default=None)
    parser.add_argument('--input_pet', type=str, default=None)
    parser.add_argument('--explain', action='store_true', help='Generate explanations')
    parser.add_argument('--output_dir', type=str, default='inference_results')
    parser.add_argument('--dataset', type=str, default='ADNI')
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
    
    # Load inputs
    if args.input_smri or args.input_fmri or args.input_pet:
        inputs = load_inputs(args, config)
    else:
        print("No input files provided. Using random data for demonstration.")
        target_shape = tuple(config['data']['input_shape'][1:])
        inputs = {
            'smri': torch.randn(1, 1, *target_shape),
            'fmri': torch.randn(1, 1, *target_shape),
            'pet': torch.randn(1, 1, *target_shape),
        }
    
    print(f"Loaded modalities: {list(inputs.keys())}")
    
    # Run inference
    results = run_inference(model, inputs, device, class_names)
    
    print("\n" + "="*50)
    print("INFERENCE RESULTS")
    print("="*50)
    print(f"Predicted Class: {results['predicted_class']}")
    print(f"Confidence: {results['confidence']:.4f}")
    print("\nClass Probabilities:")
    for class_name, prob in results['probabilities'].items():
        print(f"  {class_name}: {prob:.4f}")
    print("="*50)
    
    # Generate explanations
    if args.explain:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print("\nGenerating explanations...")
        explanations = generate_explanations(model, inputs, device, output_dir)
        print(f"\nExplanations saved to {output_dir}")


if __name__ == '__main__':
    main()
