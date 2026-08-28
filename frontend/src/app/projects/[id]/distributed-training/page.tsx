'use client';

import React from 'react';
import { DistributedTrainingStudio } from '@/components/studios/DistributedTrainingStudio';

export default function DistributedTrainingPage({ params }: { params: { id: string } }) {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Distributed Multi-Node Training Studio</h1>
        <p className="text-sm text-muted-foreground">
          Ray Train, PyTorch DDP, DeepSpeed ZeRO-3, and Ring-AllReduce GPU cluster synchronization.
        </p>
      </div>

      <DistributedTrainingStudio />
    </div>
  );
}
