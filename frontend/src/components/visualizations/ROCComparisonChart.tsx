'use client';

import React from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, Legend, ReferenceLine } from 'recharts';

interface ROCPoint {
  fpr: number;
  tpr_xgboost: number;
  tpr_lightgbm: number;
  tpr_neuralnet: number;
}

interface ROCComparisonProps {
  data?: ROCPoint[];
}

export function ROCComparisonChart({
  data = [
    { fpr: 0.0, tpr_xgboost: 0.0, tpr_lightgbm: 0.0, tpr_neuralnet: 0.0 },
    { fpr: 0.05, tpr_xgboost: 0.62, tpr_lightgbm: 0.58, tpr_neuralnet: 0.51 },
    { fpr: 0.1, tpr_xgboost: 0.78, tpr_lightgbm: 0.75, tpr_neuralnet: 0.69 },
    { fpr: 0.2, tpr_xgboost: 0.89, tpr_lightgbm: 0.86, tpr_neuralnet: 0.81 },
    { fpr: 0.3, tpr_xgboost: 0.94, tpr_lightgbm: 0.91, tpr_neuralnet: 0.88 },
    { fpr: 0.4, tpr_xgboost: 0.96, tpr_lightgbm: 0.95, tpr_neuralnet: 0.92 },
    { fpr: 0.6, tpr_xgboost: 0.98, tpr_lightgbm: 0.97, tpr_neuralnet: 0.96 },
    { fpr: 0.8, tpr_xgboost: 0.99, tpr_lightgbm: 0.99, tpr_neuralnet: 0.98 },
    { fpr: 1.0, tpr_xgboost: 1.0, tpr_lightgbm: 1.0, tpr_neuralnet: 1.0 },
  ],
}: ROCComparisonProps) {
  return (
    <Card className="border border-border/60 bg-card/50 backdrop-blur">
      <CardHeader className="flex flex-row items-center justify-between pb-2">
        <div>
          <CardTitle className="text-base font-semibold">ROC Multi-Model Benchmark</CardTitle>
          <p className="text-xs text-muted-foreground">True Positive Rate vs False Positive Rate Comparison</p>
        </div>
        <div className="flex gap-2">
          <span className="rounded bg-sky-500/10 px-2 py-0.5 text-xs text-sky-400 font-mono">XGB (AUC 0.942)</span>
          <span className="rounded bg-emerald-500/10 px-2 py-0.5 text-xs text-emerald-400 font-mono">LGBM (AUC 0.918)</span>
          <span className="rounded bg-purple-500/10 px-2 py-0.5 text-xs text-purple-400 font-mono">TabNet (AUC 0.895)</span>
        </div>
      </CardHeader>
      <CardContent>
        <div className="h-[300px] w-full">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.4} />
              <XAxis dataKey="fpr" type="number" domain={[0, 1]} tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} stroke="#64748b" />
              <YAxis type="number" domain={[0, 1]} tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} stroke="#64748b" />
              <Tooltip
                content={({ active, payload }) => {
                  if (active && payload && payload.length) {
                    const item = payload[0].payload as ROCPoint;
                    return (
                      <div className="rounded-lg border bg-popover/90 p-2 text-xs shadow-xl backdrop-blur">
                        <div className="font-semibold text-foreground">FPR: {(item.fpr * 100).toFixed(1)}%</div>
                        <div className="text-sky-400 font-mono">XGBoost TPR: {(item.tpr_xgboost * 100).toFixed(1)}%</div>
                        <div className="text-emerald-400 font-mono">LightGBM TPR: {(item.tpr_lightgbm * 100).toFixed(1)}%</div>
                        <div className="text-purple-400 font-mono">TabNet TPR: {(item.tpr_neuralnet * 100).toFixed(1)}%</div>
                      </div>
                    );
                  }
                  return null;
                }}
              />
              <Legend verticalAlign="top" height={36} />
              <ReferenceLine stroke="#64748b" strokeDasharray="4 4" segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} ifOverflow="extendDomain" />
              <Line type="monotone" dataKey="tpr_xgboost" name="XGBoost (AUC 0.942)" stroke="#38bdf8" strokeWidth={2.5} dot={{ r: 3 }} />
              <Line type="monotone" dataKey="tpr_lightgbm" name="LightGBM (AUC 0.918)" stroke="#34d399" strokeWidth={2} dot={{ r: 3 }} />
              <Line type="monotone" dataKey="tpr_neuralnet" name="TabNet Neural (AUC 0.895)" stroke="#c084fc" strokeWidth={2} dot={{ r: 3 }} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  );
}
