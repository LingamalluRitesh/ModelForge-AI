"""
ModelForge AI - Cloud Native: Istio Service Mesh & Envoy Traffic Routing
Generates Istio VirtualServices, DestinationRules, and EnvoyFilter configurations
for dynamic weighted Canary routing, shadow mirroring, and mTLS fault injection.
"""

from typing import Any, Dict, List, Optional, Tuple, Union


class IstioTrafficRouter:
    """Istio Service Mesh routing manifest generator."""

    def __init__(self, service_name: str, namespace: str = "default", host_domain: str = "modelforge.local"):
        self.service_name = service_name
        self.namespace = namespace
        self.host = f"{service_name}.{host_domain}"

    def build_canary_virtual_service(
        self,
        primary_subset: str = "v1",
        canary_subset: str = "v2",
        canary_weight: float = 10.0,
    ) -> Dict[str, Any]:
        """Generate Istio VirtualService with weighted subset distribution."""
        prim_weight = max(0.0, min(100.0, 100.0 - canary_weight))

        return {
            "apiVersion": "networking.istio.io/v1alpha3",
            "kind": "VirtualService",
            "metadata": {
                "name": f"{self.service_name}-virtualservice",
                "namespace": self.namespace,
            },
            "spec": {
                "hosts": [self.host],
                "http": [
                    {
                        "route": [
                            {
                                "destination": {
                                    "host": f"{self.service_name}.{self.namespace}.svc.cluster.local",
                                    "subset": primary_subset,
                                },
                                "weight": int(prim_weight),
                            },
                            {
                                "destination": {
                                    "host": f"{self.service_name}.{self.namespace}.svc.cluster.local",
                                    "subset": canary_subset,
                                },
                                "weight": int(canary_weight),
                            },
                        ],
                        "timeout": "5s",
                        "retries": {
                            "attempts": 3,
                            "perTryTimeout": "1s",
                            "retryOn": "5xx,connect-failure,refused-stream",
                        },
                    }
                ],
            },
        }

    def build_shadow_traffic_virtual_service(
        self,
        primary_subset: str = "v1",
        shadow_subset: str = "v2-shadow",
    ) -> Dict[str, Any]:
        """Generate Istio mirror configuration for 100% async dark shadow traffic."""
        return {
            "apiVersion": "networking.istio.io/v1alpha3",
            "kind": "VirtualService",
            "metadata": {
                "name": f"{self.service_name}-shadow-vs",
                "namespace": self.namespace,
            },
            "spec": {
                "hosts": [self.host],
                "http": [
                    {
                        "route": [
                            {
                                "destination": {
                                    "host": f"{self.service_name}.{self.namespace}.svc.cluster.local",
                                    "subset": primary_subset,
                                },
                                "weight": 100,
                            }
                        ],
                        "mirror": {
                            "host": f"{self.service_name}.{self.namespace}.svc.cluster.local",
                            "subset": shadow_subset,
                        },
                        "mirrorPercentage": {"value": 100.0},
                    }
                ],
            },
        }

    def build_destination_rule(
        self,
        subsets: List[Dict[str, str]],
        enable_mtls: bool = True,
    ) -> Dict[str, Any]:
        """Generate DestinationRule with mTLS and connection pool circuit breaking."""
        return {
            "apiVersion": "networking.istio.io/v1alpha3",
            "kind": "DestinationRule",
            "metadata": {
                "name": f"{self.service_name}-destrule",
                "namespace": self.namespace,
            },
            "spec": {
                "host": f"{self.service_name}.{self.namespace}.svc.cluster.local",
                "trafficPolicy": {
                    "tls": {"mode": "ISTIO_MUTUAL" if enable_mtls else "DISABLE"},
                    "connectionPool": {
                        "tcp": {"maxConnections": 1024},
                        "http": {"http1MaxPendingRequests": 100, "maxRequestsPerConnection": 10},
                    },
                    "outlierDetection": {
                        "consecutive5xxErrors": 3,
                        "interval": "10s",
                        "baseEjectionTime": "30s",
                        "maxEjectionPercent": 50,
                    },
                },
                "subsets": [{"name": s["name"], "labels": {"version": s["version"]}} for s in subsets],
            },
        }
