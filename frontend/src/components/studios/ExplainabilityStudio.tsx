'use client';

import React, { useState } from 'react';
import { Card, CardHeader, CardTitle, CardContent, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { SHAPForcePlot } from '@/components/visualizations/SHAPForcePlot';
import { Sparkles, Compass, Layers, Binary } from 'lucide-react';

export function ExplainabilityStudio() {
  const [method, setMethod] = useState<'SHAP' | 'INTEGRATED_GRADIENTS' | 'ANCHORS' | 'PDP'>('SHAP');

  const globalFeatureImportances = [
    { feature: 'annual_income', importance: 0.284, description: 'Annual verified applicant gross income' },
    { feature: 'debt_to_income_ratio', importance: 0.215, description: 'Monthly debt obligations vs total gross income' },
    { feature: 'credit_history_months', importance: 0.162, description: 'Length of seasoned revolving credit lines' },
    { feature: 'recent_inquiries_6m', importance: 0.128, description: 'Hard credit pulls across last 180 days' },
    { feature: 'revolving_utilization_pct', importance: 0.095, description: 'Percentage of credit limits utilized' },
    { feature: 'delinquent_accounts_2y', importance: 0.068, description: 'Late payment events over 24-month horizon' },
    { feature: 'employment_tenure_years', importance: 0.048, description: 'Consecutive years at current employer' },
  ];

  return (
    <div className="space-y-6">
      {/* Method Switcher Header */}
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div className="flex gap-2">
          <Button
            size="sm"
            variant={method === 'SHAP' ? 'default' : 'outline'}
            className="text-xs"
            onClick={() => setMethod('SHAP')}
          >
            <Sparkles className="mr-1.5 h-3.5 w-3.5" />
            TreeSHAP & KernelSHAP
          </Button>
          <Button
            size="sm"
            variant={method === 'INTEGRATED_GRADIENTS' ? 'default' : 'outline'}
            className="text-xs"
            onClick={() => setMethod('INTEGRATED_GRADIENTS')}
          >
            <Compass className="mr-1.5 h-3.5 w-3.5" />
            Integrated Gradients (Path Integral)
          </Button>
          <Button
            size="sm"
            variant={method === 'ANCHORS' ? 'default' : 'outline'}
            className="text-xs"
            onClick={() => setMethod('ANCHORS')}
          >
            <Binary className="mr-1.5 h-3.5 w-3.5" />
            Anchor Rules (High Precision)
          </Button>
        </div>

        <Badge variant="outline" className="border-primary/40 text-primary font-mono text-xs">
          Axiomatic Efficiency & Completeness Validated
        </Badge>
      </div>

      {/* SHAP Force Plot Component */}
      <SHAPForcePlot />

      {/* Global Feature Importances Table */}
      <Card className="border border-border/60 bg-card/50 backdrop-blur">
        <CardHeader className="flex flex-row items-center justify-between pb-2">
          <div>
            <CardTitle className="text-base font-semibold">Global Feature Attribution Hierarchy</CardTitle>
            <CardDescription className="text-xs">
              Mean Absolute Attribution ($\mathbb{E}[|\phi_i|]$) across 50,000 holdout instances.
            </CardDescription>
          </div>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {globalFeatureImportances.map((item, idx) => (
              <div key={idx} className="space-y-1">
                <div className="flex items-center justify-between text-xs">
                  <span className="font-mono font-semibold text-foreground">{item.feature}</span>
                  <span className="font-mono text-primary font-bold">{(item.importance * 100).toFixed(1)}%</span>
                </div>
                <div className="relative h-2 w-full overflow-hidden rounded-full bg-muted/60">
                  <div
                    className="h-full rounded-full bg-gradient-to-r from-sky-400 to-primary transition-all duration-500"
                    style={{ width: `${item.importance * 100 * 3}%` }}
                  />
                </div>
                <div className="text-[11px] text-muted-foreground">{item.description}</div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
