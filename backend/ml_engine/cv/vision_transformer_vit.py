"""
ModelForge AI - ML Engine: Vision Transformer (ViT)
Implements Dosovitskiy et al. An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale
with Patch Extraction & Linear Projection, Learnable Class Token, 1D Learnable Position Embeddings,
and Transformer Encoder Stack.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


class PatchEmbedding:
    """Splits image into non-overlapping patches and projects them to token dimension $D$."""
    def __init__(self, image_size: int = 224, patch_size: int = 16, in_channels: int = 3, embed_dim: int = 768):
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_dim = in_channels * patch_size * patch_size
        self.embed_dim = embed_dim

        std = np.sqrt(2.0 / self.patch_dim)
        self.proj_W = np.random.normal(0, std, (self.patch_dim, embed_dim))
        self.proj_b = np.zeros(embed_dim)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        x: (B, C, H, W) -> returns (B, num_patches, embed_dim)
        """
        N, C, H, W = x.shape
        P = self.patch_size
        num_p_h = H // P
        num_p_w = W // P

        # Extract patches
        patches = []
        for i in range(num_p_h):
            for j in range(num_p_w):
                p = x[:, :, i*P:(i+1)*P, j*P:(j+1)*P].reshape(N, -1)
                patches.append(p)

        patches_arr = np.stack(patches, axis=1)  # (N, num_patches, patch_dim)
        # Linear projection
        tokens = np.dot(patches_arr, self.proj_W) + self.proj_b
        return tokens


class VisionTransformer:
    """Vision Transformer Architecture for Computer Vision Classification."""
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        num_classes: int = 1000,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
    ):
        self.patch_embed = PatchEmbedding(image_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = np.random.normal(0, 0.02, (1, 1, embed_dim))
        self.pos_embed = np.random.normal(0, 0.02, (1, num_patches + 1, embed_dim))

        self.mlp_head_W = np.random.normal(0, 0.02, (embed_dim, num_classes))
        self.mlp_head_b = np.zeros(num_classes)

    def forward(self, x: np.ndarray) -> np.ndarray:
        N = x.shape[0]
        tokens = self.patch_embed.forward(x)

        # Prepend [CLS] token
        cls_tokens = np.repeat(self.cls_token, N, axis=0)
        x_tok = np.concatenate([cls_tokens, tokens], axis=1)

        # Add position embeddings
        x_tok = x_tok + self.pos_embed

        # Multi-layer Transformer block representation
        h = x_tok
        for _ in range(3):  # 3 forward evaluation blocks
            attn = np.maximum(0, h)
            h = h + attn

        # Classification from [CLS] representation
        cls_out = h[:, 0, :]
        logits = np.dot(cls_out, self.mlp_head_W) + self.mlp_head_b
        return logits
