'use client';

import React from 'react';
import { Card, CardHeader, CardTitle, CardContent, CardDescription } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid } from 'recharts';

export function SurvivalKaplanMeierPlot() {
  const kmData = [
    { time: 0, cohortA: 1.0, cohortB: 1.0 },
    { time: 30, cohortA: 0.96, cohortB: 0.91 },
    { time: 60, cohortA: 0.92, cohortB: 0.84 },
    { time: 90, cohortA: 0.88, cohortB: 0.76 },
    { time: 120, cohortA: 0.85, cohortB: 0.69 },
    { time: 150, cohortA: 0.82, cohortB: 0.61 },
    { time: 180, cohortA: 0.79, cohortB: 0.54 },
    { time: 210, cohortA: 0.77, cohortB: 0.48 },
    { time: 240, cohortA: 0.75, cohortB: 0.42 },
  ];

  return (
    <Card className="border border-border/60 bg-card/50 backdrop-blur">
      <CardHeader className="flex flex-row items-center justify-between pb-2">
        <div>
          <CardTitle className="text-base font-semibold">Kaplan-Meier Survival Probability Curves</CardTitle>
          <CardDescription className="text-xs">
            Retention and churn survival probabilities across customer risk segments.
          </CardDescription>
        </div>
        <Badge variant="outline" className="border-primary/40 text-primary font-mono text-xs">
          Log-Rank Test: p &lt; 0.001
        </Badge>
      </CardHeader>
      <CardContent>
        <div className="h-[240px] w-full">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={kmData} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.4} />
              <XAxis dataKey="time" label={{ value: 'Days Elapsed', position: 'insideBottom', offset: -5 }} stroke="#64748b" />
              <YAxis label={{ value: 'Survival $S(t)$', angle: -90, position: 'insideLeft' }} stroke="#64748b" domain={[0, 1]} />
              <Tooltip
                content={({ active, payload }) => {
                  if (active && payload && payload.length) {
                    const d = payload[0].payload;
                    return (
                      <div className="rounded-lg border bg-popover/90 p-2 text-xs shadow-xl backdrop-blur font-mono">
                        <div>Time: {d.time} Days</div>
                        <div className="text-emerald-400">Low Risk Cohort: {(d.cohortA * 100).toFixed(1)}%</div>
                        <div className="text-rose-400">High Risk Cohort: {(d.cohortB * 100).toFixed(1)}%</div>
                      </div>
                    );
                  }
                  return null;
                }}
              />
              <Line type="stepAfter" dataKey="cohortA" stroke="#10b981" strokeWidth={2} dot={false} />
              <Line type="stepAfter" dataKey="cohortB" stroke="#f43f5e" strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  );
}
