'use client';

import React from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, Legend, ReferenceLine } from 'recharts';

interface CalibrationBin {
  bin_center: number;
  observed_fraction_positives: number;
  mean_predicted_probability: number;
  sample_count: number;
}

interface CalibrationCurveProps {
  data?: CalibrationBin[];
  eceScore?: number;
  brierScore?: number;
  modelName?: string;
}

export function CalibrationCurvePlot({
  data = [
    { bin_center: 0.05, mean_predicted_probability: 0.04, observed_fraction_positives: 0.05, sample_count: 120 },
    { bin_center: 0.15, mean_predicted_probability: 0.14, observed_fraction_positives: 0.16, sample_count: 240 },
    { bin_center: 0.25, mean_predicted_probability: 0.26, observed_fraction_positives: 0.24, sample_count: 180 },
    { bin_center: 0.35, mean_predicted_probability: 0.34, observed_fraction_positives: 0.33, sample_count: 310 },
    { bin_center: 0.45, mean_predicted_probability: 0.46, observed_fraction_positives: 0.48, sample_count: 290 },
    { bin_center: 0.55, mean_predicted_probability: 0.54, observed_fraction_positives: 0.53, sample_count: 350 },
    { bin_center: 0.65, mean_predicted_probability: 0.66, observed_fraction_positives: 0.64, sample_count: 280 },
    { bin_center: 0.75, mean_predicted_probability: 0.74, observed_fraction_positives: 0.76, sample_count: 210 },
    { bin_center: 0.85, mean_predicted_probability: 0.86, observed_fraction_positives: 0.84, sample_count: 190 },
    { bin_center: 0.95, mean_predicted_probability: 0.94, observed_fraction_positives: 0.96, sample_count: 140 },
  ],
  eceScore = 0.018,
  brierScore = 0.084,
  modelName = 'Production XGBoost Classifier',
}: CalibrationCurveProps) {
  return (
    <Card className="border border-border/60 bg-card/50 backdrop-blur">
      <CardHeader className="flex flex-row items-center justify-between pb-2">
        <div>
          <CardTitle className="text-base font-semibold">Reliability Calibration Diagram</CardTitle>
          <p className="text-xs text-muted-foreground">
            {modelName} — Empirical Probability Calibration vs Perfectly Calibrated Diagonal
          </p>
        </div>
        <div className="flex items-center gap-3">
          <div className="rounded border bg-background/80 px-2 py-1 text-xs">
            <span className="text-muted-foreground">ECE: </span>
            <span className="font-mono font-bold text-emerald-400">{(eceScore * 100).toFixed(2)}%</span>
          </div>
          <div className="rounded border bg-background/80 px-2 py-1 text-xs">
            <span className="text-muted-foreground">Brier Score: </span>
            <span className="font-mono font-bold text-primary">{brierScore.toFixed(3)}</span>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <div className="h-[300px] w-full">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.4} />
              <XAxis
                dataKey="mean_predicted_probability"
                type="number"
                domain={[0, 1]}
                tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
                stroke="#64748b"
                label={{ value: 'Mean Predicted Probability', position: 'insideBottom', offset: -5, fill: '#94a3b8' }}
              />
              <YAxis
                type="number"
                domain={[0, 1]}
                tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
                stroke="#64748b"
                label={{ value: 'Fraction of Positives', angle: -90, position: 'insideLeft', fill: '#94a3b8' }}
              />
              <Tooltip
                content={({ active, payload }) => {
                  if (active && payload && payload.length) {
                    const item = payload[0].payload as CalibrationBin;
                    return (
                      <div className="rounded-lg border bg-popover/90 p-2 text-xs shadow-xl backdrop-blur">
                        <div className="font-semibold text-foreground">Probability Bin ~{(item.bin_center * 100).toFixed(0)}%</div>
                        <div className="mt-1 text-muted-foreground">Sample Size: {item.sample_count}</div>
                        <div className="text-primary font-mono">Predicted: {(item.mean_predicted_probability * 100).toFixed(1)}%</div>
                        <div className="text-emerald-400 font-mono">Observed: {(item.observed_fraction_positives * 100).toFixed(1)}%</div>
                      </div>
                    );
                  }
                  return null;
                }}
              />
              <Legend verticalAlign="top" height={36} />
              <ReferenceLine stroke="#64748b" strokeDasharray="4 4" segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} ifOverflow="extendDomain" />
              <Line
                type="monotone"
                dataKey="observed_fraction_positives"
                name="Observed Empirical Calibration"
                stroke="#38bdf8"
                strokeWidth={2.5}
                dot={{ r: 4, fill: '#38bdf8' }}
                activeDot={{ r: 6 }}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  );
}
