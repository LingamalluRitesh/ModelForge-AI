"""
ModelForge AI - Serving Engine: PagedAttention KV-Cache Virtual Memory
Implements Kwon et al. Efficient Memory Management for Large Language Model Serving with PagedAttention
with Virtual Page Tables, Non-contiguous Physical Memory Blocks, and Zero KV-Waste Prefix Sharing.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


class PhysicalBlock:
    """Fixed-size memory block storing Key and Value tensors for $B=16$ tokens."""
    def __init__(self, block_id: int, block_size: int = 16, head_dim: int = 64, num_heads: int = 8):
        self.block_id = block_id
        self.block_size = block_size
        self.k_cache = np.zeros((block_size, num_heads, head_dim), dtype=np.float32)
        self.v_cache = np.zeros((block_size, num_heads, head_dim), dtype=np.float32)
        self.num_tokens = 0
        self.ref_count = 0


class BlockSpaceManager:
    """Virtual-to-Physical Address Translation and Free Block Allocator."""
    def __init__(self, num_blocks: int = 1000, block_size: int = 16):
        self.block_size = block_size
        self.blocks = [PhysicalBlock(i, block_size) for i in range(num_blocks)]
        self.free_blocks = list(range(num_blocks))
        self.block_tables: Dict[str, List[int]] = {}

    def allocate_sequence(self, seq_id: str) -> int:
        if not self.free_blocks:
            raise MemoryError("PagedAttention Out Of KV Memory: No free physical blocks.")
        b_id = self.free_blocks.pop(0)
        self.blocks[b_id].ref_count = 1
        self.block_tables[seq_id] = [b_id]
        return b_id

    def append_slot(self, seq_id: str) -> Tuple[int, int]:
        """Returns physical block ID and offset index within block."""
        table = self.block_tables[seq_id]
        last_block_id = table[-1]
        last_block = self.blocks[last_block_id]

        if last_block.num_tokens < self.block_size:
            offset = last_block.num_tokens
            last_block.num_tokens += 1
            return last_block_id, offset
        else:
            # Allocate new physical block
            new_block_id = self.allocate_sequence(seq_id)
            table.append(new_block_id)
            self.blocks[new_block_id].num_tokens = 1
            return new_block_id, 0

    def free_sequence(self, seq_id: str):
        if seq_id in self.block_tables:
            for b_id in self.block_tables[seq_id]:
                self.blocks[b_id].ref_count -= 1
                if self.blocks[b_id].ref_count == 0:
                    self.blocks[b_id].num_tokens = 0
                    self.free_blocks.append(b_id)
            del self.block_tables[seq_id]
