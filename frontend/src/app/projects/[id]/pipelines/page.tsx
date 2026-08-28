'use client';

import React from 'react';
import { PipelineOrchestrationStudio } from '@/components/studios/PipelineOrchestrationStudio';
import { DataSankeyDiagram } from '@/components/visualizations/DataSankeyDiagram';

export default function PipelinesPage({ params }: { params: { id: string } }) {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Enterprise Pipeline DAG Orchestration</h1>
        <p className="text-sm text-muted-foreground">
          Autonomous execution flow from data validation, feature store joins, HPO blenders to Istio Canary rollouts.
        </p>
      </div>

      <PipelineOrchestrationStudio />
      <DataSankeyDiagram />
    </div>
  );
}
