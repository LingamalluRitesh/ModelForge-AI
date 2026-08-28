"""
ModelForge AI - Federated Learning & Decentralized Coordination Service
Coordinates federated training rounds, client node registration, and FedAvg parameter aggregation.
"""

from typing import Any, Dict, List, Optional
import time
from sqlalchemy.ext.asyncio import AsyncSession
from ml_engine.algorithms.federated_learning import FederatedAveragingAggregator, DifferentialPrivacyEngine


class FederatedLearningService:
    def __init__(self, session: AsyncSession):
        self.session = session
        self.aggregator = FederatedAveragingAggregator()
        self.dp_engine = DifferentialPrivacyEngine(epsilon=1.0, delta=1e-5)

    async def aggregate_federated_round(
        self,
        round_number: int,
        client_updates: List[Dict[str, Any]],
        apply_differential_privacy: bool = True,
    ) -> Dict[str, Any]:
        """Aggregate decentralized weights submitted by edge client workers."""
        if not client_updates:
            return {"error": "No client updates provided for aggregation"}

        aggregated_params = self.aggregator.aggregate_weights(client_updates)

        if apply_differential_privacy:
            # Inject DP noise into aggregated parameters
            for k in aggregated_params.keys():
                arr = aggregated_params[k]
                noise = self.dp_engine.add_gaussian_noise(arr, batch_size=len(client_updates), dataset_size=10000)
                aggregated_params[k] = noise.tolist() if hasattr(noise, "tolist") else noise
        else:
            aggregated_params = {k: v.tolist() if hasattr(v, "tolist") else v for k, v in aggregated_params.items()}

        return {
            "federated_round": round_number,
            "participating_clients_count": len(client_updates),
            "status": "COMPLETED",
            "dp_applied": apply_differential_privacy,
            "global_parameters": aggregated_params,
            "aggregated_at": time.time(),
        }
