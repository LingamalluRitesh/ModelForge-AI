'use client';

import React from 'react';
import { Card, CardHeader, CardTitle, CardContent, CardDescription } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { ResponsiveContainer, AreaChart, Area, XAxis, YAxis, Tooltip, CartesianGrid } from 'recharts';

export function PrecisionRecallCurvePlot() {
  const prData = [
    { recall: 0.0, precision: 1.0 },
    { recall: 0.1, precision: 0.98 },
    { recall: 0.2, precision: 0.96 },
    { recall: 0.3, precision: 0.95 },
    { recall: 0.4, precision: 0.94 },
    { recall: 0.5, precision: 0.92 },
    { recall: 0.6, precision: 0.89 },
    { recall: 0.7, precision: 0.85 },
    { recall: 0.8, precision: 0.81 },
    { recall: 0.9, precision: 0.72 },
    { recall: 1.0, precision: 0.48 },
  ];

  return (
    <Card className="border border-border/60 bg-card/50 backdrop-blur">
      <CardHeader className="flex flex-row items-center justify-between pb-2">
        <div>
          <CardTitle className="text-base font-semibold">Precision-Recall Curve (PR-AUC)</CardTitle>
          <CardDescription className="text-xs">
            Precision vs Recall trade-off across varying decision threshold cuts.
          </CardDescription>
        </div>
        <Badge variant="outline" className="border-emerald-500/40 text-emerald-400 font-mono text-xs">
          Average Precision (AP): 0.894
        </Badge>
      </CardHeader>
      <CardContent>
        <div className="h-[260px] w-full">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={prData} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
              <defs>
                <linearGradient id="prGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#10b981" stopOpacity={0.4} />
                  <stop offset="95%" stopColor="#10b981" stopOpacity={0.0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.4} />
              <XAxis dataKey="recall" label={{ value: 'Recall', position: 'insideBottom', offset: -5 }} stroke="#64748b" />
              <YAxis label={{ value: 'Precision', angle: -90, position: 'insideLeft' }} stroke="#64748b" domain={[0, 1]} />
              <Tooltip
                content={({ active, payload }) => {
                  if (active && payload && payload.length) {
                    const d = payload[0].payload;
                    return (
                      <div className="rounded-lg border bg-popover/90 p-2 text-xs shadow-xl backdrop-blur font-mono">
                        <div>Recall: {d.recall.toFixed(2)}</div>
                        <div className="text-emerald-400 font-bold">Precision: {d.precision.toFixed(2)}</div>
                      </div>
                    );
                  }
                  return null;
                }}
              />
              <Area type="monotone" dataKey="precision" stroke="#10b981" strokeWidth={2} fillOpacity={1} fill="url(#prGrad)" />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  );
}
