"""
ModelForge AI - ML Engine: Additive Homomorphic Encryption
Implements Paillier Cryptosystem for Privacy-Preserving Machine Learning Inference and Aggregation.
$E(m_1) \cdot E(m_2) \pmod{n^2} = E(m_1 + m_2 \pmod n)$
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import math
import random


class PaillierKeypair:
    def __init__(self, p: int = 61, q: int = 53):
        self.p = p
        self.q = q
        self.n = p * q
        self.n_sq = self.n * self.n
        self.lam = math.lcm(p - 1, q - 1)
        self.g = self.n + 1

        # Precompute L(g^lambda mod n^2)^-1 mod n
        u = pow(self.g, self.lam, self.n_sq)
        l_u = (u - 1) // self.n
        self.mu = pow(l_u, -1, self.n)

    def encrypt(self, m: int) -> int:
        """Encrypt plaintext integer $m \in \mathbb{Z}_n$: $c = g^m r^n \pmod{n^2}$."""
        r = random.randint(1, self.n - 1)
        while math.gcd(r, self.n) != 1:
            r = random.randint(1, self.n - 1)

        c = (pow(self.g, m, self.n_sq) * pow(r, self.n, self.n_sq)) % self.n_sq
        return c

    def decrypt(self, c: int) -> int:
        """Decrypt ciphertext integer: $m = L(c^\lambda \pmod{n^2}) \cdot \mu \pmod n$."""
        u = pow(c, self.lam, self.n_sq)
        l_u = (u - 1) // self.n
        m = (l_u * self.mu) % self.n
        return m

    def add_encrypted(self, c1: int, c2: int) -> int:
        """Homomorphic addition: $c_1 \cdot c_2 \pmod{n^2}$."""
        return (c1 * c2) % self.n_sq

    def scalar_mul_encrypted(self, c: int, scalar: int) -> int:
        """Homomorphic scalar multiplication: $c^{scalar} \pmod{n^2}$."""
        return pow(c, scalar, self.n_sq)
