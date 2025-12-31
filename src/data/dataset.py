"""
Dataset Classes for NeuroFusionXAI

Provides PyTorch Dataset implementations for ADNI, PPMI, and custom
multimodal neuroimaging datasets.
"""

import os
from typing import Optional, Dict, List, Tuple, Callable, Any
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import pandas as pd


class NeuroDataset(Dataset):
    """
    Base dataset class for neuroimaging data.
    
    Args:
        root_dir: Root directory containing data
        modalities: List of modalities to load
        transform: Optional transforms to apply
        target_shape: Target shape for volumes
    """
    
    def __init__(
        self,
        root_dir: str,
        modalities: List[str] = ['smri', 'fmri', 'pet'],
        transform: Optional[Callable] = None,
        target_shape: Tuple[int, int, int] = (96, 112, 96),
    ):
        self.root_dir = Path(root_dir)
        self.modalities = modalities
        self.transform = transform
        self.target_shape = target_shape
        self.samples = []
        self.labels = []
        
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], int]:
        sample_info = self.samples[idx]
        label = self.labels[idx]
        
        # Load each modality
        data = {}
        for modality in self.modalities:
            if modality in sample_info:
                volume = self._load_volume(sample_info[modality])
                volume = self._preprocess(volume)
                data[modality] = torch.from_numpy(volume).float().unsqueeze(0)
        
        if self.transform:
            data = self.transform(data)
        
        return data, label
    
    def _load_volume(self, path: str) -> np.ndarray:
        """Load NIfTI volume."""
        img = nib.load(path)
        return img.get_fdata().astype(np.float32)
    
    def _preprocess(self, volume: np.ndarray) -> np.ndarray:
        """Basic preprocessing: resize and normalize."""
        from scipy.ndimage import zoom
        
        # Resize to target shape
        factors = [t / s for t, s in zip(self.target_shape, volume.shape)]
        volume = zoom(volume, factors, order=1)
        
        # Normalize
        volume = (volume - volume.mean()) / (volume.std() + 1e-8)
        
        return volume


class ADNIDataset(NeuroDataset):
    """
    ADNI (Alzheimer's Disease Neuroimaging Initiative) Dataset.
    
    Expected directory structure:
    root_dir/
        sMRI/
            subject_001.nii.gz
            ...
        fMRI/
            subject_001.nii.gz
            ...
        PET/
            subject_001.nii.gz
            ...
        labels.csv
    
    Args:
        root_dir: Root directory containing ADNI data
        split: Data split ('train', 'val', 'test')
        modalities: List of modalities to load
        transform: Optional transforms
    """
    
    LABEL_MAP = {
        'CN': 0,   # Cognitively Normal
        'MCI': 1,  # Mild Cognitive Impairment
        'AD': 2,   # Alzheimer's Disease
        'EMCI': 1, # Early MCI
        'LMCI': 1, # Late MCI
    }
    
    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        modalities: List[str] = ['smri', 'fmri', 'pet'],
        transform: Optional[Callable] = None,
    ):
        super().__init__(root_dir, modalities, transform)
        self.split = split
        self._load_dataset()
    
    def _load_dataset(self):
        """Load ADNI dataset metadata and file paths."""
        labels_path = self.root_dir / 'labels.csv'
        
        if labels_path.exists():
            df = pd.read_csv(labels_path)
            
            # Filter by split if specified
            if 'split' in df.columns:
                df = df[df['split'] == self.split]
            
            for _, row in df.iterrows():
                sample = {}
                subject_id = row['subject_id']
                
                # Check each modality
                for modality in self.modalities:
                    modality_dir = self.root_dir / modality.upper()
                    if modality == 'smri':
                        modality_dir = self.root_dir / 'sMRI'
                    elif modality == 'fmri':
                        modality_dir = self.root_dir / 'fMRI'
                    
                    # Find matching file
                    for ext in ['.nii.gz', '.nii']:
                        path = modality_dir / f"{subject_id}{ext}"
                        if path.exists():
                            sample[modality] = str(path)
                            break
                
                if sample:  # At least one modality found
                    self.samples.append(sample)
                    label = self.LABEL_MAP.get(row.get('diagnosis', 'CN'), 0)
                    self.labels.append(label)
        else:
            # Auto-discover files if no labels.csv
            self._auto_discover()
    
    def _auto_discover(self):
        """Auto-discover files without labels file."""
        smri_dir = self.root_dir / 'sMRI'
        if smri_dir.exists():
            for f in smri_dir.glob('*.nii*'):
                subject_id = f.stem.replace('.nii', '')
                sample = {'smri': str(f)}
                
                # Look for other modalities
                for modality in ['fmri', 'pet']:
                    mod_dir = self.root_dir / modality.upper()
                    if modality == 'fmri':
                        mod_dir = self.root_dir / 'fMRI'
                    
                    for ext in ['.nii.gz', '.nii']:
                        path = mod_dir / f"{subject_id}{ext}"
                        if path.exists():
                            sample[modality] = str(path)
                            break
                
                self.samples.append(sample)
                self.labels.append(0)  # Default label


class PPMIDataset(NeuroDataset):
    """
    PPMI (Parkinson's Progression Markers Initiative) Dataset.
    
    Args:
        root_dir: Root directory containing PPMI data
        split: Data split
        modalities: List of modalities
        transform: Optional transforms
    """
    
    LABEL_MAP = {
        'HC': 0,      # Healthy Control
        'PD': 1,      # Parkinson's Disease
        'PROD': 2,    # Prodromal
        'SWEDD': 1,   # Scans Without Evidence of Dopamine Deficit
    }
    
    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        modalities: List[str] = ['smri', 'fmri', 'datscan'],
        transform: Optional[Callable] = None,
    ):
        super().__init__(root_dir, modalities, transform)
        self.split = split
        self._load_dataset()
    
    def _load_dataset(self):
        """Load PPMI dataset."""
        labels_path = self.root_dir / 'labels.csv'
        
        if labels_path.exists():
            df = pd.read_csv(labels_path)
            
            if 'split' in df.columns:
                df = df[df['split'] == self.split]
            
            for _, row in df.iterrows():
                sample = {}
                subject_id = row['subject_id']
                
                for modality in self.modalities:
                    modality_dir = self.root_dir / modality
                    
                    for ext in ['.nii.gz', '.nii']:
                        path = modality_dir / f"{subject_id}{ext}"
                        if path.exists():
                            sample[modality] = str(path)
                            break
                
                if sample:
                    self.samples.append(sample)
                    label = self.LABEL_MAP.get(row.get('diagnosis', 'HC'), 0)
                    self.labels.append(label)


class MultiModalDataset(Dataset):
    """
    Generic multimodal neuroimaging dataset.
    
    Args:
        data_dict: Dictionary mapping subject_id to modality paths
        labels: Dictionary mapping subject_id to label
        modalities: List of modalities
        transform: Optional transforms
        target_shape: Target volume shape
    """
    
    def __init__(
        self,
        data_dict: Dict[str, Dict[str, str]],
        labels: Dict[str, int],
        modalities: List[str] = ['smri', 'fmri', 'pet'],
        transform: Optional[Callable] = None,
        target_shape: Tuple[int, int, int] = (96, 112, 96),
    ):
        self.data_dict = data_dict
        self.labels_dict = labels
        self.modalities = modalities
        self.transform = transform
        self.target_shape = target_shape
        self.subject_ids = list(data_dict.keys())
    
    def __len__(self) -> int:
        return len(self.subject_ids)
    
    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], int]:
        subject_id = self.subject_ids[idx]
        paths = self.data_dict[subject_id]
        label = self.labels_dict[subject_id]
        
        data = {}
        for modality in self.modalities:
            if modality in paths:
                volume = self._load_and_preprocess(paths[modality])
                data[modality] = torch.from_numpy(volume).float().unsqueeze(0)
        
        if self.transform:
            data = self.transform(data)
        
        return data, label
    
    def _load_and_preprocess(self, path: str) -> np.ndarray:
        """Load and preprocess a volume."""
        from scipy.ndimage import zoom
        
        img = nib.load(path)
        volume = img.get_fdata().astype(np.float32)
        
        # Resize
        factors = [t / s for t, s in zip(self.target_shape, volume.shape)]
        volume = zoom(volume, factors, order=1)
        
        # Normalize
        volume = (volume - volume.mean()) / (volume.std() + 1e-8)
        
        return volume


def create_data_loaders(
    config: Dict[str, Any],
    dataset_name: str = 'ADNI',
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test data loaders.
    
    Args:
        config: Configuration dictionary
        dataset_name: Name of dataset ('ADNI' or 'PPMI')
    
    Returns:
        Train, validation, and test DataLoaders
    """
    data_config = config['data']
    root_dir = data_config.get('root_dir', f'./data/{dataset_name}')
    modalities = data_config.get('modalities', ['smri', 'fmri', 'pet'])
    batch_size = config['training'].get('batch_size', 16)
    num_workers = config['hardware'].get('num_workers', 4)
    
    if dataset_name.upper() == 'ADNI':
        DatasetClass = ADNIDataset
    elif dataset_name.upper() == 'PPMI':
        DatasetClass = PPMIDataset
    else:
        DatasetClass = NeuroDataset
    
    train_dataset = DatasetClass(root_dir, split='train', modalities=modalities)
    val_dataset = DatasetClass(root_dir, split='val', modalities=modalities)
    test_dataset = DatasetClass(root_dir, split='test', modalities=modalities)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader, test_loader
