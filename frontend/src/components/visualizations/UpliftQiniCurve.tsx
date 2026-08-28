'use client';

import React from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, Legend, ReferenceLine } from 'recharts';

interface QiniPoint {
  population_percent: number;
  qini_model: number;
  qini_random: number;
  qini_optimal: number;
}

interface UpliftQiniProps {
  data?: QiniPoint[];
  qiniScore?: number;
}

export function UpliftQiniCurve({
  data = [
    { population_percent: 0, qini_model: 0, qini_random: 0, qini_optimal: 0 },
    { population_percent: 10, qini_model: 85, qini_random: 20, qini_optimal: 120 },
    { population_percent: 20, qini_model: 150, qini_random: 40, qini_optimal: 180 },
    { population_percent: 30, qini_model: 195, qini_random: 60, qini_optimal: 200 },
    { population_percent: 40, qini_model: 215, qini_random: 80, qini_optimal: 200 },
    { population_percent: 50, qini_model: 220, qini_random: 100, qini_optimal: 200 },
    { population_percent: 70, qini_model: 210, qini_random: 140, qini_optimal: 200 },
    { population_percent: 100, qini_model: 200, qini_random: 200, qini_optimal: 200 },
  ],
  qiniScore = 0.732,
}: UpliftQiniProps) {
  return (
    <Card className="border border-border/60 bg-card/50 backdrop-blur">
      <CardHeader className="flex flex-row items-center justify-between pb-2">
        <div>
          <CardTitle className="text-base font-semibold">Causal Uplift & Qini Curve</CardTitle>
          <p className="text-xs text-muted-foreground">Cumulative Incremental Treatment Effect by Targeting Population</p>
        </div>
        <div className="rounded border bg-background/80 px-2 py-1 text-xs">
          <span className="text-muted-foreground">Normalized Qini (AUUC): </span>
          <span className="font-mono font-bold text-emerald-400">+{qiniScore.toFixed(3)}</span>
        </div>
      </CardHeader>
      <CardContent>
        <div className="h-[300px] w-full">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.4} />
              <XAxis dataKey="population_percent" tickFormatter={(v) => `${v}%`} stroke="#64748b" />
              <YAxis stroke="#64748b" />
              <Tooltip
                content={({ active, payload }) => {
                  if (active && payload && payload.length) {
                    const item = payload[0].payload as QiniPoint;
                    return (
                      <div className="rounded-lg border bg-popover/90 p-2 text-xs shadow-xl backdrop-blur">
                        <div className="font-semibold text-foreground">Top {item.population_percent}% Targeted</div>
                        <div className="text-primary font-mono">X-Learner Uplift: +{item.qini_model} units</div>
                        <div className="text-muted-foreground font-mono">Random Selection: +{item.qini_random} units</div>
                      </div>
                    );
                  }
                  return null;
                }}
              />
              <Legend verticalAlign="top" height={36} />
              <Line type="monotone" dataKey="qini_model" name="Causal Meta-Learner" stroke="#38bdf8" strokeWidth={2.5} dot={{ r: 4 }} />
              <Line type="monotone" dataKey="qini_optimal" name="Theoretical Maximum" stroke="#34d399" strokeWidth={1.5} strokeDasharray="4 4" dot={false} />
              <Line type="monotone" dataKey="qini_random" name="Random Targeting" stroke="#64748b" strokeWidth={1.5} strokeDasharray="3 3" dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  );
}
