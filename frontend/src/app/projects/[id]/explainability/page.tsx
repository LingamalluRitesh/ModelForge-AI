'use client';

import React from 'react';
import { ExplainabilityStudio } from '@/components/studios/ExplainabilityStudio';
import { NeuralArchitectureVisualizer } from '@/components/visualizations/NeuralArchitectureVisualizer';

export default function ExplainabilityPage({ params }: { params: { id: string } }) {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Model Explainability & Attribution Studio</h1>
        <p className="text-sm text-muted-foreground">
          TreeSHAP, KernelSHAP, Integrated Gradients, Anchor Rules, and Neural Computational Graph Visualizer.
        </p>
      </div>

      <ExplainabilityStudio />
      <NeuralArchitectureVisualizer />
    </div>
  );
}
