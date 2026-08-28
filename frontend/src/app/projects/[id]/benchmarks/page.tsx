'use client';

import React from 'react';
import { BenchmarksStudio } from '@/components/studios/BenchmarksStudio';

export default function BenchmarksPage({ params }: { params: { id: string } }) {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Model Arena & Benchmark Leaderboard</h1>
        <p className="text-sm text-muted-foreground">
          Automated cross-validation tournament rankings, ROC comparisons, and latency profiling.
        </p>
      </div>

      <BenchmarksStudio />
    </div>
  );
}
