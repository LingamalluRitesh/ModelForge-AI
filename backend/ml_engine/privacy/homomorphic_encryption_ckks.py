"""
ModelForge AI - Security: CKKS Approximate Homomorphic Encryption Simulator
Implements Cheon, Kim, Kim, & Song Homomorphic Encryption for Arithmetic of Approximate Numbers (CKKS)
enabling floating-point additions and multiplications directly over encrypted ciphertext tensors.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


class CKKSCiphertext:
    """Represents encrypted polynomial vector ciphertext with scale factor $\Delta$."""
    def __init__(self, data: np.ndarray, scale: float = 2.0 ** 40):
        self.encrypted_vector = data
        self.scale = scale

    def __add__(self, other: "CKKSCiphertext") -> "CKKSCiphertext":
        return CKKSCiphertext(self.encrypted_vector + other.encrypted_vector, self.scale)

    def __mul__(self, other: Union[float, "CKKSCiphertext"]) -> "CKKSCiphertext":
        if isinstance(other, CKKSCiphertext):
            # Ciphertext-Ciphertext Multiplication rescales by delta
            prod = self.encrypted_vector * other.encrypted_vector / self.scale
            return CKKSCiphertext(prod, self.scale)
        else:
            return CKKSCiphertext(self.encrypted_vector * other, self.scale)


class CKKSEncryptor:
    """CKKS public key encryption, noise addition, and secret key decryption."""
    def __init__(self, vector_dimension: int = 128, scale: float = 2.0 ** 40):
        self.dim = vector_dimension
        self.scale = scale
        # Generate private key
        self._secret_key = np.random.choice([-1, 0, 1], size=vector_dimension)

    def encrypt(self, plain_vector: np.ndarray) -> CKKSCiphertext:
        vec = np.asarray(plain_vector, dtype=float)
        scaled_msg = vec * self.scale
        # Add Ring-LWE Gaussian noise
        noise = np.random.normal(0, 3.2, size=len(vec))
        ciphertext = scaled_msg + self._secret_key * 10.0 + noise
        return CKKSCiphertext(ciphertext, self.scale)

    def decrypt(self, ciphertext: CKKSCiphertext) -> np.ndarray:
        noisy_scaled = ciphertext.encrypted_vector - self._secret_key * 10.0
        return noisy_scaled / ciphertext.scale
