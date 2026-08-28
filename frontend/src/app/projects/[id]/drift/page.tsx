'use client';

import React from 'react';
import { DatasetDriftStudio } from '@/components/studios/DatasetDriftStudio';

export default function DriftPage({ params }: { params: { id: string } }) {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Statistical Drift & Data Quality Studio</h1>
        <p className="text-sm text-muted-foreground">
          Continuous Kolmogorov-Smirnov, Population Stability Index, Wasserstein Distance, and MMD tests.
        </p>
      </div>

      <DatasetDriftStudio />
    </div>
  );
}
