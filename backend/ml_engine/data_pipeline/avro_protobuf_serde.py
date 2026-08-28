"""
ModelForge AI - Data Pipeline: High-Performance Binary Serialization (Avro / Protobuf)
Implements schema-enforced binary encoders and zero-copy binary decoders for streaming telemetry.
"""

from typing import Any, Dict, List, Optional
import struct


class BinaryRecordEncoder:
    """Encodes structured schema dictionaries into compact binary byte buffers."""
    def __init__(self, schema_fields: List[Tuple[str, str]]):
        self.schema = schema_fields  # list of (field_name, type: 'int' | 'float' | 'str')

    def encode(self, record: Dict[str, Any]) -> bytes:
        buf = bytearray()
        for name, f_type in self.schema:
            val = record.get(name)
            if f_type == "int":
                buf.extend(struct.pack(">q", int(val or 0)))
            elif f_type == "float":
                buf.extend(struct.pack(">d", float(val or 0.0)))
            elif f_type == "str":
                str_bytes = str(val or "").encode("utf-8")
                buf.extend(struct.pack(">H", len(str_bytes)))
                buf.extend(str_bytes)
        return bytes(buf)

    def decode(self, binary_data: bytes) -> Dict[str, Any]:
        offset = 0
        rec = {}
        for name, f_type in self.schema:
            if f_type == "int":
                val = struct.unpack_from(">q", binary_data, offset)[0]
                offset += 8
            elif f_type == "float":
                val = struct.unpack_from(">d", binary_data, offset)[0]
                offset += 8
            elif f_type == "str":
                str_len = struct.unpack_from(">H", binary_data, offset)[0]
                offset += 2
                val = binary_data[offset : offset + str_len].decode("utf-8")
                offset += str_len
            rec[name] = val
        return rec
