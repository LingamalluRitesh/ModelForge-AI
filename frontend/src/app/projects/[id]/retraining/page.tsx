'use client';

import React from 'react';
import { ContinuousRetrainingStudio } from '@/components/studios/ContinuousRetrainingStudio';

export default function RetrainingPage({ params }: { params: { id: string } }) {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Continuous Automated Retraining Studio</h1>
        <p className="text-sm text-muted-foreground">
          Drift-triggered retraining triggers, automated quality gate evaluations, and Canary promotions.
        </p>
      </div>

      <ContinuousRetrainingStudio />
    </div>
  );
}
