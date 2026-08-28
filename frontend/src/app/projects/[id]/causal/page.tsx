'use client';

import React from 'react';
import { CausalInferenceStudio } from '@/components/studios/CausalInferenceStudio';
import { CausalDAGCanvas } from '@/components/visualizations/CausalDAGCanvas';
import { UpliftQiniCurve } from '@/components/visualizations/UpliftQiniCurve';

export default function CausalInferencePage({ params }: { params: { id: string } }) {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Causal Inference & Policy Optimization Studio</h1>
        <p className="text-sm text-muted-foreground">
          Double Machine Learning (DML), Propensity Score Matching, Meta-Learners, and Structural Causal DAGs.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <CausalDAGCanvas />
        <UpliftQiniCurve />
      </div>

      <CausalInferenceStudio />
    </div>
  );
}
