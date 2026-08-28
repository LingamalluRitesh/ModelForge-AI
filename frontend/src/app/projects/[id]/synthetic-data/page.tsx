'use client';

import React from 'react';
import { SyntheticDataStudio } from '@/components/studios/SyntheticDataStudio';

export default function SyntheticDataPage({ params }: { params: { id: string } }) {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Synthetic Data Generation & Privacy Studio</h1>
        <p className="text-sm text-muted-foreground">
          Conditional Tabular GAN (CTGAN), Gaussian Copula joint distribution generators, and Differential Privacy budgets.
        </p>
      </div>

      <SyntheticDataStudio />
    </div>
  );
}
