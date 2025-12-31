"""
Homomorphic Encryption Implementation for NeuroFusionXAI

This module implements CKKS-based homomorphic encryption for
privacy-preserving inference as described in Section III.C.1.

Implements Eq. 2: c^mod = Encrypt(x^mod, pk, Δ)

Reference: "CKKS: Approximate Homomorphic Encryption" (Cheon et al., 2017)
"""

from typing import Optional, Tuple, List, Dict, Any
import math

import torch
import torch.nn as nn
import numpy as np

# Optional TenSEAL import for actual HE operations
try:
    import tenseal as ts
    TENSEAL_AVAILABLE = True
except ImportError:
    TENSEAL_AVAILABLE = False
    print("Warning: TenSEAL not available. Using simulated encryption.")


class CKKSEncoder:
    """
    CKKS encoding for converting between plaintexts and ciphertexts.
    
    The CKKS scheme encodes real-valued data into polynomial form
    suitable for homomorphic operations.
    
    Args:
        poly_modulus_degree: Polynomial modulus degree (power of 2)
        coeff_mod_bit_sizes: Coefficient modulus bit sizes
        scale: Encoding scale (Δ in the paper)
    """
    
    def __init__(
        self,
        poly_modulus_degree: int = 8192,
        coeff_mod_bit_sizes: List[int] = [60, 40, 40, 60],
        scale: float = 2**40,
    ):
        self.poly_modulus_degree = poly_modulus_degree
        self.coeff_mod_bit_sizes = coeff_mod_bit_sizes
        self.scale = scale
        
        if TENSEAL_AVAILABLE:
            self.context = ts.context(
                ts.SCHEME_TYPE.CKKS,
                poly_modulus_degree=poly_modulus_degree,
                coeff_mod_bit_sizes=coeff_mod_bit_sizes,
            )
            self.context.global_scale = scale
            self.context.generate_galois_keys()
            self.context.generate_relin_keys()
        else:
            self.context = None
            
    def encode(self, data: np.ndarray) -> Any:
        """
        Encode plaintext data.
        
        Args:
            data: Numpy array to encode
            
        Returns:
            Encoded (and optionally encrypted) data
        """
        if TENSEAL_AVAILABLE and self.context is not None:
            return ts.ckks_vector(self.context, data.flatten().tolist())
        return data
    
    def decode(self, encoded: Any) -> np.ndarray:
        """
        Decode encoded data back to plaintext.
        
        Args:
            encoded: Encoded data
            
        Returns:
            Decoded numpy array
        """
        if TENSEAL_AVAILABLE and hasattr(encoded, 'decrypt'):
            return np.array(encoded.decrypt())
        return np.array(encoded)


class HomomorphicEncryption:
    """
    Homomorphic Encryption module using CKKS scheme.
    
    Provides encryption, decryption, and homomorphic operations
    for privacy-preserving neural network inference.
    
    Implements Eq. 2: c^mod = Encrypt(x^mod, pk, Δ)
    
    Args:
        config: HE configuration dictionary
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        
        self.poly_modulus_degree = config.get('poly_modulus_degree', 8192)
        self.coeff_mod_bit_sizes = config.get(
            'coeff_mod_bit_sizes', [60, 40, 40, 60]
        )
        self.scale = 2 ** config.get('scale', 40)
        
        # Initialize encoder
        self.encoder = CKKSEncoder(
            poly_modulus_degree=self.poly_modulus_degree,
            coeff_mod_bit_sizes=self.coeff_mod_bit_sizes,
            scale=self.scale,
        )
        
        # Store context for TenSEAL operations
        self.context = self.encoder.context
        
    def encrypt(self, tensor: torch.Tensor) -> Any:
        """
        Encrypt a PyTorch tensor.
        
        Implements Eq. 2: c^mod = Encrypt(x^mod, pk, Δ)
        
        Args:
            tensor: Input tensor to encrypt
            
        Returns:
            Encrypted tensor (or simulated encryption)
        """
        # Convert to numpy
        data = tensor.detach().cpu().numpy()
        original_shape = data.shape
        
        if TENSEAL_AVAILABLE and self.context is not None:
            # Flatten and encrypt
            flat_data = data.flatten().tolist()
            encrypted = ts.ckks_vector(self.context, flat_data)
            return EncryptedTensor(encrypted, original_shape)
        else:
            # Simulated encryption (for testing without TenSEAL)
            return EncryptedTensor(data, original_shape, simulated=True)
    
    def decrypt(self, encrypted: 'EncryptedTensor') -> torch.Tensor:
        """
        Decrypt an encrypted tensor.
        
        Args:
            encrypted: Encrypted tensor object
            
        Returns:
            Decrypted PyTorch tensor
        """
        if encrypted.simulated:
            return torch.from_numpy(encrypted.data).float()
        
        if TENSEAL_AVAILABLE:
            decrypted = np.array(encrypted.data.decrypt())
            decrypted = decrypted.reshape(encrypted.shape)
            return torch.from_numpy(decrypted).float()
        
        raise RuntimeError("Cannot decrypt without TenSEAL")
    
    def add(self, enc1: 'EncryptedTensor', enc2: 'EncryptedTensor') -> 'EncryptedTensor':
        """
        Homomorphic addition of two encrypted tensors.
        
        Args:
            enc1: First encrypted tensor
            enc2: Second encrypted tensor
            
        Returns:
            Encrypted sum
        """
        if enc1.simulated and enc2.simulated:
            result = enc1.data + enc2.data
            return EncryptedTensor(result, enc1.shape, simulated=True)
        
        if TENSEAL_AVAILABLE:
            result = enc1.data + enc2.data
            return EncryptedTensor(result, enc1.shape)
        
        raise RuntimeError("Cannot perform HE addition")
    
    def multiply_plain(
        self, encrypted: 'EncryptedTensor', plain: np.ndarray
    ) -> 'EncryptedTensor':
        """
        Multiply encrypted tensor by plaintext.
        
        Args:
            encrypted: Encrypted tensor
            plain: Plaintext array
            
        Returns:
            Encrypted product
        """
        if encrypted.simulated:
            result = encrypted.data * plain
            return EncryptedTensor(result, encrypted.shape, simulated=True)
        
        if TENSEAL_AVAILABLE:
            result = encrypted.data * plain.flatten().tolist()
            return EncryptedTensor(result, encrypted.shape)
        
        raise RuntimeError("Cannot perform HE multiplication")
    
    def get_context_bytes(self) -> bytes:
        """Serialize encryption context for transmission."""
        if TENSEAL_AVAILABLE and self.context is not None:
            return self.context.serialize()
        return b""
    
    @classmethod
    def from_context_bytes(cls, context_bytes: bytes) -> 'HomomorphicEncryption':
        """Create HE instance from serialized context."""
        instance = cls()
        if TENSEAL_AVAILABLE and context_bytes:
            instance.context = ts.context_from(context_bytes)
            instance.encoder.context = instance.context
        return instance


class EncryptedTensor:
    """
    Wrapper for encrypted tensor data.
    
    Stores encrypted data along with original shape for reconstruction.
    
    Args:
        data: Encrypted data (TenSEAL vector or numpy array)
        shape: Original tensor shape
        simulated: Whether this is simulated encryption
    """
    
    def __init__(
        self,
        data: Any,
        shape: Tuple[int, ...],
        simulated: bool = False,
    ):
        self.data = data
        self.shape = shape
        self.simulated = simulated
        
    def __repr__(self) -> str:
        mode = "simulated" if self.simulated else "encrypted"
        return f"EncryptedTensor(shape={self.shape}, mode={mode})"


class EncryptedLinear:
    """
    Encrypted linear layer for HE inference.
    
    Performs linear transformation on encrypted inputs.
    
    Args:
        weight: Weight matrix (plaintext)
        bias: Bias vector (plaintext, optional)
        he_module: HomomorphicEncryption instance
    """
    
    def __init__(
        self,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        he_module: HomomorphicEncryption,
    ):
        self.weight = weight.detach().cpu().numpy()
        self.bias = bias.detach().cpu().numpy() if bias is not None else None
        self.he = he_module
        
    def forward(self, encrypted_input: EncryptedTensor) -> EncryptedTensor:
        """
        Apply linear transformation to encrypted input.
        
        Args:
            encrypted_input: Encrypted input tensor
            
        Returns:
            Encrypted output
        """
        # Matrix-vector multiplication in encrypted domain
        # This is a simplified implementation
        if encrypted_input.simulated:
            input_flat = encrypted_input.data.flatten()
            output = input_flat @ self.weight.T
            if self.bias is not None:
                output = output + self.bias
            return EncryptedTensor(output, (output.shape[0],), simulated=True)
        
        # For actual HE, this would use the TenSEAL operations
        raise NotImplementedError("Full HE linear layer requires TenSEAL")


class HEInferenceEngine:
    """
    Inference engine for running models on encrypted data.
    
    Converts a standard PyTorch model for HE-compatible inference.
    
    Args:
        model: PyTorch model
        he_config: Homomorphic encryption configuration
    """
    
    def __init__(
        self,
        model: nn.Module,
        he_config: Optional[Dict[str, Any]] = None,
    ):
        self.model = model
        self.he = HomomorphicEncryption(he_config)
        
        # Track computation overhead
        self.encryption_time = 0.0
        self.computation_time = 0.0
        self.decryption_time = 0.0
        
    def encrypt_input(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, EncryptedTensor]:
        """Encrypt all input modalities."""
        import time
        start = time.time()
        
        encrypted_inputs = {}
        for modality, tensor in inputs.items():
            encrypted_inputs[modality] = self.he.encrypt(tensor)
            
        self.encryption_time = time.time() - start
        return encrypted_inputs
    
    def decrypt_output(self, encrypted_output: EncryptedTensor) -> torch.Tensor:
        """Decrypt model output."""
        import time
        start = time.time()
        
        result = self.he.decrypt(encrypted_output)
        
        self.decryption_time = time.time() - start
        return result
    
    def get_timing_stats(self) -> Dict[str, float]:
        """Get timing statistics for HE operations."""
        return {
            'encryption_time': self.encryption_time,
            'computation_time': self.computation_time,
            'decryption_time': self.decryption_time,
            'total_time': self.encryption_time + self.computation_time + self.decryption_time,
        }
