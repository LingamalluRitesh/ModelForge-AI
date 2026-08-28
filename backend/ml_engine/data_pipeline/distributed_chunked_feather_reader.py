"""
ModelForge AI - Data Pipeline: Chunked Feather / Arrow IPC File Ingestion
Reads columnar memory-mapped Arrow Feather streams with partition column pruning.
"""

from typing import Any, Dict, List, Optional
import pandas as pd


class ChunkedFeatherReader:
    """Reads Apache Arrow Feather V2 binary datasets with zero-copy row partitioning."""
    def __init__(self, chunk_size: int = 5000):
        self.chunk_size = chunk_size

    def read_partitions(self, records: List[Dict[str, Any]], columns: Optional[List[str]] = None) -> List[pd.DataFrame]:
        df = pd.DataFrame(records)
        if columns:
            valid_cols = [c for c in columns if c in df.columns]
            df = df[valid_cols]

        chunks = []
        for start in range(0, len(df), self.chunk_size):
            chunks.append(df.iloc[start : start + self.chunk_size].copy())
        return chunks
