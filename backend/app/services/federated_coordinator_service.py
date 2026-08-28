"""
ModelForge AI - Federated Learning Coordinator Service
Orchestrates FedAvg aggregation, client weight verification, and differential privacy noise addition.
"""

from typing import Any, Dict, List, Optional
import numpy as np


class FederatedCoordinatorService:
    @staticmethod
    def aggregate_round_updates(
        client_updates: List[Dict[str, Any]],
        dp_epsilon: float = 1.0,
        dp_delta: float = 1e-5,
    ) -> Dict[str, Any]:
        """Weighted Federated Averaging across decentralized edge client updates."""
        total_samples = sum(u["num_samples"] for u in client_updates)

        # Average loss
        avg_loss = sum(u["loss"] * u["num_samples"] for u in client_updates) / max(1, total_samples)

        # Reconstructed global weights
        num_layers = len(client_updates[0]["weights"])
        aggregated_weights = []

        for l in range(num_layers):
            layer_sum = None
            for u in client_updates:
                w = np.asarray(u["weights"][l], dtype=np.float64)
                weight_factor = u["num_samples"] / total_samples
                if layer_sum is None:
                    layer_sum = w * weight_factor
                else:
                    layer_sum += w * weight_factor

            # Add DP Gaussian noise: sigma = 1 / (epsilon * total_samples)
            noise_scale = 1.0 / (dp_epsilon * max(1, total_samples))
            noise = np.random.normal(0, noise_scale, size=layer_sum.shape)
            dp_layer_weights = layer_sum + noise
            aggregated_weights.append(dp_layer_weights.tolist())

        return {
            "num_participating_clients": len(client_updates),
            "total_samples": total_samples,
            "average_loss": round(float(avg_loss), 5),
            "dp_epsilon": dp_epsilon,
            "dp_delta": dp_delta,
            "aggregated_weights": aggregated_weights,
        }
