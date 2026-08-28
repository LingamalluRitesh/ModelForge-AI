'use client';

import React, { useState } from 'react';
import { Card, CardHeader, CardTitle, CardContent, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { ShieldCheck, Scale, Sparkles, AlertTriangle } from 'lucide-react';

interface FairnessMetric {
  groupName: string;
  selectionRate: number;
  truePositiveRate: number;
  falsePositiveRate: number;
  disparateImpactRatio: number;
  statisticalParityDiff: number;
  status: 'COMPLIANT' | 'VIOLATION';
}

export function FairnessGovernanceStudio() {
  const [metrics, setMetrics] = useState<FairnessMetric[]>([
    {
      groupName: 'Gender: Male (Reference)',
      selectionRate: 0.68,
      truePositiveRate: 0.84,
      falsePositiveRate: 0.12,
      disparateImpactRatio: 1.0,
      statisticalParityDiff: 0.0,
      status: 'COMPLIANT',
    },
    {
      groupName: 'Gender: Female (Protected)',
      selectionRate: 0.62,
      truePositiveRate: 0.82,
      falsePositiveRate: 0.11,
      disparateImpactRatio: 0.91,
      statisticalParityDiff: -0.06,
      status: 'COMPLIANT',
    },
    {
      groupName: 'Age: 18-25 (Protected)',
      selectionRate: 0.49,
      truePositiveRate: 0.74,
      falsePositiveRate: 0.15,
      disparateImpactRatio: 0.72,
      statisticalParityDiff: -0.19,
      status: 'VIOLATION',
    },
    {
      groupName: 'Age: 26-64 (Reference)',
      selectionRate: 0.71,
      truePositiveRate: 0.86,
      falsePositiveRate: 0.10,
      disparateImpactRatio: 1.0,
      statisticalParityDiff: 0.0,
      status: 'COMPLIANT',
    },
  ]);

  return (
    <div className="space-y-6">
      {/* Overview Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <Card className="border border-border/60 bg-card/50 backdrop-blur p-4">
          <div className="flex items-center justify-between">
            <span className="text-xs text-muted-foreground font-medium">Four-Fifths (80%) Rule Compliance</span>
            <Scale className="h-4 w-4 text-emerald-400" />
          </div>
          <div className="mt-2 text-2xl font-bold font-mono text-emerald-400">PASSED (0.912)</div>
          <div className="mt-1 text-[11px] text-muted-foreground">EEOC Disparate Impact Threshold &ge; 0.80</div>
        </Card>

        <Card className="border border-border/60 bg-card/50 backdrop-blur p-4">
          <div className="flex items-center justify-between">
            <span className="text-xs text-muted-foreground font-medium">Equalized Odds Max Difference</span>
            <ShieldCheck className="h-4 w-4 text-sky-400" />
          </div>
          <div className="mt-2 text-2xl font-bold font-mono text-sky-400">2.1% (Compliant)</div>
          <div className="mt-1 text-[11px] text-muted-foreground">Threshold &le; 5.0% across all cohorts</div>
        </Card>

        <Card className="border border-border/60 bg-card/50 backdrop-blur p-4">
          <div className="flex items-center justify-between">
            <span className="text-xs text-muted-foreground font-medium">Debiasing Strategy</span>
            <Sparkles className="h-4 w-4 text-purple-400" />
          </div>
          <div className="mt-2 text-sm font-bold text-purple-400">Adversarial GRL + Equalized Odds</div>
          <div className="mt-1 text-[11px] text-muted-foreground">Gradient Reversal Layer In-Processing</div>
        </Card>
      </div>

      {/* Cohort Metric Table */}
      <Card className="border border-border/60 bg-card/50 backdrop-blur">
        <CardHeader className="flex flex-row items-center justify-between pb-2">
          <div>
            <CardTitle className="text-base font-semibold">Protected Cohort Fairness Audit</CardTitle>
            <CardDescription className="text-xs">
              Statistical Parity Difference, Equal Opportunity (TPR), and Disparate Impact across demographics.
            </CardDescription>
          </div>
          <Badge variant="outline" className="border-amber-500/40 text-amber-400 font-mono text-xs">
            1 Actionable Cohort Disparity
          </Badge>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs font-mono">
              <thead>
                <tr className="border-b border-border/60 text-muted-foreground font-sans">
                  <th className="pb-2 font-medium">Cohort / Group</th>
                  <th className="pb-2 font-medium">Selection Rate (Positive)</th>
                  <th className="pb-2 font-medium">True Positive Rate (TPR)</th>
                  <th className="pb-2 font-medium">False Positive Rate (FPR)</th>
                  <th className="pb-2 font-medium">Disparate Impact Ratio</th>
                  <th className="pb-2 font-medium">Status</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border/30">
                {metrics.map((m, idx) => (
                  <tr key={idx} className="hover:bg-muted/30">
                    <td className="py-3 font-semibold font-sans text-foreground">{m.groupName}</td>
                    <td className="py-3 text-primary">{(m.selectionRate * 100).toFixed(1)}%</td>
                    <td className="py-3 text-emerald-400">{(m.truePositiveRate * 100).toFixed(1)}%</td>
                    <td className="py-3 text-muted-foreground">{(m.falsePositiveRate * 100).toFixed(1)}%</td>
                    <td className="py-3">
                      <span className={m.disparateImpactRatio < 0.8 ? 'text-rose-400 font-bold' : 'text-foreground'}>
                        {m.disparateImpactRatio.toFixed(2)}
                      </span>
                    </td>
                    <td className="py-3">
                      {m.status === 'COMPLIANT' ? (
                        <Badge className="bg-emerald-500/10 text-emerald-400 border-emerald-500/20 text-[10px]">
                          Compliant
                        </Badge>
                      ) : (
                        <Badge className="bg-rose-500/10 text-rose-400 border-rose-500/20 text-[10px]">
                          Action Required
                        </Badge>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
