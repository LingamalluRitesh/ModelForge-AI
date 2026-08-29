"""
Model Artifact Cryptographic Integrity & Deserialization Security Guard.
Validates SHA-384 weight checksums and scans binary model streams for unsafe deserialization opcodes.
"""

from typing import Dict, List, Any, Tuple, Optional
import hashlib
import hmac
import io
import pickletools


class ArtifactIntegrityService:
    """Protects model registries against supply-chain tampering and insecure pickle payloads."""

    DISALLOWED_MODULES = {"os", "subprocess", "posix", "nt", "builtins.eval", "builtins.exec", "socket"}

    def __init__(self, signing_key: str = "modelforge-enterprise-root-cert"):
        self.signing_key = signing_key.encode("utf-8")

    def compute_artifact_fingerprint(self, raw_bytes: bytes) -> Dict[str, str]:
        sha384_hash = hashlib.sha384(raw_bytes).hexdigest()
        signature = hmac.new(self.signing_key, raw_bytes, hashlib.sha384).hexdigest()
        return {
            "sha384_checksum": sha384_hash,
            "hmac_signature": signature,
            "byte_size": str(len(raw_bytes)),
        }

    def verify_artifact_signature(self, raw_bytes: bytes, signature: str) -> bool:
        expected = hmac.new(self.signing_key, raw_bytes, hashlib.sha384).hexdigest()
        return hmac.compare_digest(expected, signature)

    def scan_pickle_bytecode(self, raw_bytes: bytes) -> Dict[str, Any]:
        """Scans pickle bytecode instructions for arbitrary code execution vulnerability."""
        unsafe_opcodes = []
        is_safe = True

        try:
            for opcode, arg, pos in pickletools.genops(raw_bytes):
                if opcode.name in ("GLOBAL", "INST", "REDUCE"):
                    if arg:
                        for bad in self.DISALLOWED_MODULES:
                            if bad in arg:
                                unsafe_opcodes.append(f"DANGEROUS_OPCODE_{opcode.name}_{arg}")
                                is_safe = False
        except Exception as e:
            # If not valid pickle or parsing fails, flag for manual review
            unsafe_opcodes.append(f"UNPARSEABLE_BYTECODE_{str(e)}")
            is_safe = False

        return {
            "is_safe_for_loading": is_safe,
            "detected_threats": unsafe_opcodes,
            "total_bytes_scanned": len(raw_bytes),
        }
