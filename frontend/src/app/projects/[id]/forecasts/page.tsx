'use client';

import React from 'react';
import { ForecastReconciliationStudio } from '@/components/studios/ForecastReconciliationStudio';

export default function ForecastsPage({ params }: { params: { id: string } }) {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Hierarchical Time Series Forecasting Studio</h1>
        <p className="text-sm text-muted-foreground">
          MinT Optimal Reconciliation, Decomposable Fourier Seasonality, ARIMA, and DeepAR Probabilistic Quantile Forecasts.
        </p>
      </div>

      <ForecastReconciliationStudio />
    </div>
  );
}
