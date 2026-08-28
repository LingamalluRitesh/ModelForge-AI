'use client';

import React from 'react';
import { Card, CardHeader, CardTitle, CardContent, CardDescription } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Activity, AlertTriangle, CheckCircle2, TrendingUp } from 'lucide-react';
import { FeatureCorrelationHeatmap } from '@/components/visualizations/FeatureCorrelationHeatmap';

interface DriftFeatureRow {
  featureName: string;
  type: 'NUMERICAL' | 'CATEGORICAL';
  testName: 'PSI' | 'KS_TEST' | 'WASSERSTEIN' | 'CHI_SQUARE';
  statistic: number;
  pValue?: number;
  driftDetected: boolean;
}

export function DatasetDriftStudio() {
  const rows: DriftFeatureRow[] = [
    { featureName: 'transaction_amount', type: 'NUMERICAL', testName: 'KS_TEST', statistic: 0.184, pValue: 0.001, driftDetected: true },
    { featureName: 'device_trust_score', type: 'NUMERICAL', testName: 'PSI', statistic: 0.245, driftDetected: true },
    { featureName: 'ip_country_code', type: 'CATEGORICAL', testName: 'CHI_SQUARE', statistic: 14.8, pValue: 0.042, driftDetected: true },
    { featureName: 'merchant_category', type: 'CATEGORICAL', testName: 'PSI', statistic: 0.042, driftDetected: false },
    { featureName: 'card_age_months', type: 'NUMERICAL', testName: 'KS_TEST', statistic: 0.021, pValue: 0.482, driftDetected: false },
    { featureName: 'login_frequency_7d', type: 'NUMERICAL', testName: 'WASSERSTEIN', statistic: 0.015, driftDetected: false },
  ];

  return (
    <div className="space-y-6">
      {/* Top Drift Status */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <Card className="border border-border/60 bg-card/50 backdrop-blur p-4">
          <div className="flex items-center justify-between">
            <span className="text-xs text-muted-foreground font-medium">Multivariate MMD Drift Score</span>
            <Activity className="h-4 w-4 text-amber-400" />
          </div>
          <div className="mt-2 text-2xl font-bold font-mono text-amber-400">0.084 (Elevated)</div>
          <div className="mt-1 text-[11px] text-muted-foreground">Maximum Mean Discrepancy with RBF Kernel</div>
        </Card>

        <Card className="border border-border/60 bg-card/50 backdrop-blur p-4">
          <div className="flex items-center justify-between">
            <span className="text-xs text-muted-foreground font-medium">Adversarial Domain AUC</span>
            <TrendingUp className="h-4 w-4 text-rose-400" />
          </div>
          <div className="mt-2 text-2xl font-bold font-mono text-rose-400">0.762 (Drift Breach)</div>
          <div className="mt-1 text-[11px] text-muted-foreground">Classifier discriminates training vs live inference</div>
        </Card>

        <Card className="border border-border/60 bg-card/50 backdrop-blur p-4">
          <div className="flex items-center justify-between">
            <span className="text-xs text-muted-foreground font-medium">Drifted Feature Columns</span>
            <AlertTriangle className="h-4 w-4 text-amber-400" />
          </div>
          <div className="mt-2 text-2xl font-bold font-mono">3 / 48 Features</div>
          <div className="mt-1 text-[11px] text-muted-foreground">Auto-retraining trigger condition met</div>
        </Card>
      </div>

      {/* Feature Correlation Matrix */}
      <FeatureCorrelationHeatmap />

      {/* Drift Matrix Table */}
      <Card className="border border-border/60 bg-card/50 backdrop-blur">
        <CardHeader className="flex flex-row items-center justify-between pb-2">
          <div>
            <CardTitle className="text-base font-semibold">Column-Level Statistical Drift Diagnostics</CardTitle>
            <CardDescription className="text-xs">
              Continuous hypothesis testing between Baseline Gold Ingestion and Last 24-Hour Production Stream.
            </CardDescription>
          </div>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs font-mono">
              <thead>
                <tr className="border-b border-border/60 text-muted-foreground font-sans">
                  <th className="pb-2 font-medium">Feature Name</th>
                  <th className="pb-2 font-medium">Data Type</th>
                  <th className="pb-2 font-medium">Statistical Test</th>
                  <th className="pb-2 font-medium">Test Statistic</th>
                  <th className="pb-2 font-medium">P-Value</th>
                  <th className="pb-2 font-medium">Drift Decision</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border/30">
                {rows.map((row, idx) => (
                  <tr key={idx} className="hover:bg-muted/30">
                    <td className="py-2.5 font-sans font-semibold text-foreground">{row.featureName}</td>
                    <td className="py-2.5 text-muted-foreground">{row.type}</td>
                    <td className="py-2.5 text-primary">{row.testName}</td>
                    <td className="py-2.5 font-bold">{row.statistic.toFixed(3)}</td>
                    <td className="py-2.5 text-muted-foreground">{row.pValue !== undefined ? row.pValue.toFixed(3) : 'N/A'}</td>
                    <td className="py-2.5">
                      {row.driftDetected ? (
                        <Badge className="bg-rose-500/10 text-rose-400 border-rose-500/20 text-[10px]">
                          Drift Detected
                        </Badge>
                      ) : (
                        <Badge className="bg-emerald-500/10 text-emerald-400 border-emerald-500/20 text-[10px]">
                          Stable
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
