'use client';

import React, { useState } from 'react';
import { Card, CardHeader, CardTitle, CardContent, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { UpliftQiniCurve } from '@/components/visualizations/UpliftQiniCurve';
import { CausalDAGCanvas } from '@/components/visualizations/CausalDAGCanvas';
import { GitBranch, Sparkles, TrendingUp } from 'lucide-react';

export function CausalInferenceStudio() {
  const [method, setMethod] = useState<'DML' | 'PSM' | 'X_LEARNER'>('DML');

  return (
    <div className="space-y-6">
      {/* Top Method Switcher */}
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div className="flex gap-2">
          <Button
            size="sm"
            variant={method === 'DML' ? 'default' : 'outline'}
            className="text-xs"
            onClick={() => setMethod('DML')}
          >
            <Sparkles className="mr-1.5 h-3.5 w-3.5" />
            Double Machine Learning (DML)
          </Button>
          <Button
            size="sm"
            variant={method === 'PSM' ? 'default' : 'outline'}
            className="text-xs"
            onClick={() => setMethod('PSM')}
          >
            <GitBranch className="mr-1.5 h-3.5 w-3.5" />
            Propensity Score Matching (PSM)
          </Button>
          <Button
            size="sm"
            variant={method === 'X_LEARNER' ? 'default' : 'outline'}
            className="text-xs"
            onClick={() => setMethod('X_LEARNER')}
          >
            <TrendingUp className="mr-1.5 h-3.5 w-3.5" />
            X-Learner Uplift (CATE)
          </Button>
        </div>

        <Badge variant="outline" className="border-primary/40 text-primary font-mono text-xs">
          Pearl Backdoor Criterion Satisfied
        </Badge>
      </div>

      {/* Uplift Qini Curve */}
      <UpliftQiniCurve />

      {/* Interactive Causal DAG Canvas */}
      <CausalDAGCanvas />
    </div>
  );
}
