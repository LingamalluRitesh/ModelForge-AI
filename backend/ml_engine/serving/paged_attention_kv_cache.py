"""
ModelForge AI - Serving Engine: PagedAttention KV-Cache Virtual Memory Manager
Implements Kwon et al. Efficient Memory Management for Large Language Model Serving with PagedAttention (vLLM)
allocating fixed-size physical memory pages across dynamic non-contiguous virtual token sequence blocks.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


class PhysicalBlock:
    """Represents a continuous GPU memory slot holding $B$ key-value token vectors."""

    def __init__(self, block_id: int, block_size: int = 16, head_dim: int = 64):
        self.block_id = block_id
        self.block_size = block_size
        self.head_dim = head_dim
        self.k_cache = np.zeros((block_size, head_dim), dtype=np.float32)
        self.v_cache = np.zeros((block_size, head_dim), dtype=np.float32)
        self.num_tokens_stored = 0
        self.ref_count = 0

    def is_full(self) -> bool:
        return self.num_tokens_stored >= self.block_size

    def append_token(self, k_vec: np.ndarray, v_vec: np.ndarray):
        idx = self.num_tokens_stored
        self.k_cache[idx] = k_vec
        self.v_cache[idx] = v_vec
        self.num_tokens_stored += 1


class PagedAttentionMemoryManager:
    """Virtual-to-Physical page table allocator with Copy-On-Write (CoW) fork semantics."""

    def __init__(self, num_physical_blocks: int = 256, block_size: int = 16, head_dim: int = 64):
        self.block_size = block_size
        self.head_dim = head_dim
        self.free_blocks: List[int] = list(range(num_physical_blocks))
        self.blocks: Dict[int, PhysicalBlock] = {
            i: PhysicalBlock(i, block_size, head_dim) for i in range(num_physical_blocks)
        }
        # request_id -> list of block_ids (Page Table)
        self.page_tables: Dict[str, List[int]] = {}

    def allocate_request(self, request_id: str):
        self.page_tables[request_id] = []

    def append_token(self, request_id: str, k: np.ndarray, v: np.ndarray):
        table = self.page_tables.get(request_id)
        if table is None:
            raise KeyError(f"Unknown request ID: {request_id}")

        if not table or self.blocks[table[-1]].is_full():
            if not self.free_blocks:
                raise MemoryError("Out of GPU physical KV-cache memory blocks!")
            new_block_id = self.free_blocks.pop(0)
            self.blocks[new_block_id].ref_count = 1
            self.blocks[new_block_id].num_tokens_stored = 0
            table.append(new_block_id)

        current_block = self.blocks[table[-1]]
        current_block.append_token(k, v)

    def free_request(self, request_id: str):
        table = self.page_tables.pop(request_id, [])
        for blk_id in table:
            blk = self.blocks[blk_id]
            blk.ref_count -= 1
            if blk.ref_count == 0:
                self.free_blocks.append(blk_id)
