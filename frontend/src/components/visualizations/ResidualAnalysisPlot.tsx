'use client';

import React from 'react';
import { Card, CardHeader, CardTitle, CardContent, CardDescription } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { ResponsiveContainer, ScatterChart, Scatter, XAxis, YAxis, Tooltip, CartesianGrid, ReferenceLine } from 'recharts';

export function ResidualAnalysisPlot() {
  const residualData = [
    { predicted: 12.4, residual: -0.21 },
    { predicted: 18.2, residual: 0.45 },
    { predicted: 24.1, residual: -0.12 },
    { predicted: 31.0, residual: 0.18 },
    { predicted: 38.5, residual: -0.34 },
    { predicted: 45.2, residual: 0.08 },
    { predicted: 52.8, residual: -0.15 },
    { predicted: 61.1, residual: 0.22 },
    { predicted: 69.4, residual: -0.05 },
    { predicted: 78.0, residual: 0.14 },
  ];

  return (
    <Card className="border border-border/60 bg-card/50 backdrop-blur">
      <CardHeader className="flex flex-row items-center justify-between pb-2">
        <div>
          <CardTitle className="text-base font-semibold">Residual vs Predicted Value Plot</CardTitle>
          <CardDescription className="text-xs">
            Checks homoscedasticity and zero-mean normality of model regression errors.
          </CardDescription>
        </div>
        <Badge variant="outline" className="border-emerald-500/40 text-emerald-400 font-mono text-xs">
          Homoscedasticity: PASSED
        </Badge>
      </CardHeader>
      <CardContent>
        <div className="h-[240px] w-full">
          <ResponsiveContainer width="100%" height="100%">
            <ScatterChart margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.4} />
              <XAxis dataKey="predicted" label={{ value: 'Predicted ($\hat{y}$)', position: 'insideBottom', offset: -5 }} stroke="#64748b" />
              <YAxis dataKey="residual" label={{ value: 'Residual ($y - \hat{y}$)', angle: -90, position: 'insideLeft' }} stroke="#64748b" domain={[-1, 1]} />
              <ReferenceLine y={0} stroke="#94a3b8" strokeDasharray="4 4" />
              <Tooltip
                content={({ active, payload }) => {
                  if (active && payload && payload.length) {
                    const d = payload[0].payload;
                    return (
                      <div className="rounded-lg border bg-popover/90 p-2 text-xs shadow-xl backdrop-blur font-mono">
                        <div>Predicted: {d.predicted}</div>
                        <div className="text-primary font-bold">Residual: {d.residual}</div>
                      </div>
                    );
                  }
                  return null;
                }}
              />
              <Scatter data={residualData} fill="#38bdf8" />
            </ScatterChart>
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  );
}
