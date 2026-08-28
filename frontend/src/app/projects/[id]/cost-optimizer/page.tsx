'use client';

import React from 'react';
import { CloudCostOptimizerStudio } from '@/components/studios/CloudCostOptimizerStudio';

export default function CostOptimizerPage({ params }: { params: { id: string } }) {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Cloud Infrastructure Cost Optimizer</h1>
        <p className="text-sm text-muted-foreground">
          Real-time compute breakdown, ONNX INT8 quantization savings, and spot instance scheduler.
        </p>
      </div>

      <CloudCostOptimizerStudio />
    </div>
  );
}
