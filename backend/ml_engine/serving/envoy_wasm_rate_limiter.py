"""
ModelForge AI - Serving: WebAssembly (Wasm) Token Bucket Rate Limiter Plugin
Synthesizes Envoy Proxy Wasm C++ / Rust binary filter bindings for microsecond API gatekeeping.
"""

from typing import Any, Dict, List, Optional
import json


class EnvoyWasmRateLimiterConfig:
    """Generates declarative Wasm runtime configuration for Envoy Service Mesh."""
    @staticmethod
    def generate_wasm_filter_config(
        rate_limit_rps: int = 2500,
        burst_capacity: int = 5000,
        header_key: str = "X-ModelForge-Tenant-ID",
    ) -> Dict[str, Any]:
        return {
            "name": "envoy.filters.http.wasm",
            "typed_config": {
                "@type": "type.googleapis.com/envoy.extensions.filters.http.wasm.v3.Wasm",
                "config": {
                    "name": "modelforge_rate_limiter",
                    "root_id": "modelforge_root",
                    "configuration": {
                        "@type": "type.googleapis.com/google.protobuf.StringValue",
                        "value": json.dumps({
                            "rate_limit_rps": rate_limit_rps,
                            "burst_capacity": burst_capacity,
                            "tenant_header": header_key,
                            "rejection_status_code": 429,
                            "rejection_message": "Too Many Requests: Rate limit SLA exceeded.",
                        }),
                    },
                    "vm_config": {
                        "runtime": "envoy.wasm.runtime.v8",
                        "vm_id": "modelforge_wasm_vm",
                        "code": {
                            "local": {
                                "filename": "/etc/envoy/wasm/rate_limiter.wasm"
                            }
                        },
                    },
                },
            },
        }
