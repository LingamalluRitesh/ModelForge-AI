'use client';

import React from 'react';
import { FederatedLearningStudio } from '@/components/studios/FederatedLearningStudio';

export default function FederatedLearningPage({ params }: { params: { id: string } }) {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Federated Learning & Differential Privacy Studio</h1>
        <p className="text-sm text-muted-foreground">
          Federated Averaging (FedAvg), edge round synchronization, and $(\epsilon, \delta)$-Differential Privacy budget trackers.
        </p>
      </div>

      <FederatedLearningStudio />
    </div>
  );
}
