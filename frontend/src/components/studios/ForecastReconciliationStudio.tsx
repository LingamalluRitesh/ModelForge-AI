'use client';

import React from 'react';
import { Card, CardHeader, CardTitle, CardContent, CardDescription } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { GitFork, TrendingUp, CheckCircle2 } from 'lucide-react';

export function ForecastReconciliationStudio() {
  const hierarchyLevels = [
    { level: 'Total Enterprise Revenue (Level 0)', baseForecast: '$12,450,000', reconciledForecast: '$12,380,000', adjustment: '-$70,000', coherent: true },
    { level: 'Region: North America (Level 1)', baseForecast: '$7,200,000', reconciledForecast: '$7,150,000', adjustment: '-$50,000', coherent: true },
    { level: 'Region: EMEA (Level 1)', baseForecast: '$3,450,000', reconciledForecast: '$3,430,000', adjustment: '-$20,000', coherent: true },
    { level: 'Region: APAC (Level 1)', baseForecast: '$1,800,000', reconciledForecast: '$1,800,000', adjustment: '$0', coherent: true },
    { level: 'Product: Cloud Subscriptions (Level 2)', baseForecast: '$4,500,000', reconciledForecast: '$4,480,000', adjustment: '-$20,000', coherent: true },
    { level: 'Product: Enterprise Professional Services (Level 2)', baseForecast: '$2,700,000', reconciledForecast: '$2,670,000', adjustment: '-$30,000', coherent: true },
  ];

  return (
    <div className="space-y-6">
      <Card className="border border-border/60 bg-card/50 backdrop-blur">
        <CardHeader className="flex flex-row items-center justify-between pb-2">
          <div>
            <CardTitle className="text-base font-semibold flex items-center gap-2">
              <GitFork className="h-4 w-4 text-primary" />
              Optimal Hierarchical Forecast Reconciliation (MinT)
            </CardTitle>
            <CardDescription className="text-xs">
              Ensures bottom-up, regional, and total enterprise time series forecasts sum coherently across all hierarchy nodes without discrepancies.
            </CardDescription>
          </div>
          <Badge variant="outline" className="border-emerald-500/40 text-emerald-400 font-mono text-xs">
            Summing Constraint Satisfied • Zero Discrepancy
          </Badge>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs font-mono">
              <thead>
                <tr className="border-b border-border/60 text-muted-foreground font-sans">
                  <th className="pb-2 font-medium">Hierarchy Node / Sub-aggregate</th>
                  <th className="pb-2 font-medium">Base Independent Forecast</th>
                  <th className="pb-2 font-medium">MinT Reconciled Forecast</th>
                  <th className="pb-2 font-medium">Coherence Adjustment</th>
                  <th className="pb-2 font-medium">Status</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border/30">
                {hierarchyLevels.map((node, idx) => (
                  <tr key={idx} className="hover:bg-muted/30">
                    <td className="py-2.5 font-sans font-semibold text-foreground">{node.level}</td>
                    <td className="py-2.5 text-muted-foreground">{node.baseForecast}</td>
                    <td className="py-2.5 text-primary font-bold">{node.reconciledForecast}</td>
                    <td className="py-2.5 text-emerald-400">{node.adjustment}</td>
                    <td className="py-2.5">
                      <Badge className="bg-emerald-500/10 text-emerald-400 border-emerald-500/20 text-[10px]">
                        Coherent
                      </Badge>
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
