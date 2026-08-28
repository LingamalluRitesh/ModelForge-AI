'use client';

import React from 'react';
import { Card, CardHeader, CardTitle, CardContent, CardDescription } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';

export function FeatureCorrelationHeatmap() {
  const featureNames = ['CreditScore', 'AnnualIncome', 'DebtRatio', 'Age', 'LoanAmount'];
  const corrMatrix = [
    [1.00, 0.42, -0.68, 0.31, -0.15],
    [0.42, 1.00, -0.35, 0.28, 0.65],
    [-0.68, -0.35, 1.00, -0.12, 0.48],
    [0.31, 0.28, -0.12, 1.00, 0.08],
    [-0.15, 0.65, 0.48, 0.08, 1.00],
  ];

  return (
    <Card className="border border-border/60 bg-card/50 backdrop-blur">
      <CardHeader className="flex flex-row items-center justify-between pb-2">
        <div>
          <CardTitle className="text-base font-semibold">Feature Pearson Correlation Matrix</CardTitle>
          <CardDescription className="text-xs">
            Collinearity diagnostic identifying multi-collinear and redundant input signals.
          </CardDescription>
        </div>
        <Badge variant="outline" className="border-emerald-500/40 text-emerald-400 font-mono text-xs">
          VIF &lt; 5.0 (No Collinearity)
        </Badge>
      </CardHeader>
      <CardContent>
        <div className="overflow-x-auto">
          <table className="w-full text-center text-xs font-mono">
            <thead>
              <tr className="border-b border-border/60 text-muted-foreground font-sans">
                <th className="pb-2 text-left font-medium">Features</th>
                {featureNames.map((name, i) => (
                  <th key={i} className="pb-2 font-medium">{name}</th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-border/30">
              {corrMatrix.map((row, rowIdx) => (
                <tr key={rowIdx} className="hover:bg-muted/30">
                  <td className="py-2.5 font-semibold text-left font-sans text-foreground">
                    {featureNames[rowIdx]}
                  </td>
                  {row.map((val, colIdx) => {
                    const isDiag = rowIdx === colIdx;
                    const isHigh = Math.abs(val) > 0.6 && !isDiag;
                    return (
                      <td
                        key={colIdx}
                        className={`py-2.5 ${
                          isDiag
                            ? 'text-muted-foreground'
                            : isHigh
                            ? 'bg-rose-500/10 text-rose-400 font-bold'
                            : 'text-foreground'
                        }`}
                      >
                        {val.toFixed(2)}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </CardContent>
    </Card>
  );
}
