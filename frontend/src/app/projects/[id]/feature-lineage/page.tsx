'use client';

import React from 'react';
import { FeatureCorrelationHeatmap } from '@/components/visualizations/FeatureCorrelationHeatmap';

export default function FeatureLineagePage({ params }: { params: { id: string } }) {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Feature Store Provenance & Lineage</h1>
        <p className="text-sm text-muted-foreground">
          Bi-temporal transformations, upstream data source DAGs, and collinearity heatmaps.
        </p>
      </div>

      <FeatureCorrelationHeatmap />
    </div>
  );
}
