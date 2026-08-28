'use client';

import React from 'react';
import { Card, CardHeader, CardTitle, CardContent, CardDescription } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';

interface ConfusionMatrixProps {
  matrix?: number[][];
  classNames?: string[];
}

export function ConfusionMatrixHeatmap({
  matrix = [
    [4850, 120, 30],
    [95, 3980, 125],
    [40, 110, 2650],
  ],
  classNames = ['Low Risk', 'Medium Risk', 'High Risk'],
}: ConfusionMatrixProps) {
  const total = matrix.flat().reduce((a, b) => a + b, 0);

  return (
    <Card className="border border-border/60 bg-card/50 backdrop-blur">
      <CardHeader className="flex flex-row items-center justify-between pb-2">
        <div>
          <CardTitle className="text-base font-semibold">Normalized Confusion Matrix</CardTitle>
          <CardDescription className="text-xs">
            Evaluation partitions across {total.toLocaleString()} holdout validation records.
          </CardDescription>
        </div>
        <Badge variant="outline" className="border-primary/40 text-primary font-mono text-xs">
          Overall Accuracy: 96.1%
        </Badge>
      </CardHeader>
      <CardContent>
        <div className="overflow-x-auto">
          <table className="w-full text-center text-xs font-mono">
            <thead>
              <tr className="border-b border-border/60 text-muted-foreground font-sans">
                <th className="pb-2 text-left font-medium">Actual \ Predicted</th>
                {classNames.map((name, i) => (
                  <th key={i} className="pb-2 font-medium">{name}</th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-border/30">
              {matrix.map((row, rowIdx) => {
                const rowTotal = row.reduce((a, b) => a + b, 0);
                return (
                  <tr key={rowIdx} className="hover:bg-muted/30">
                    <td className="py-3 font-semibold text-left font-sans text-foreground">
                      {classNames[rowIdx]}
                    </td>
                    {row.map((val, colIdx) => {
                      const isDiagonal = rowIdx === colIdx;
                      const pct = ((val / rowTotal) * 100).toFixed(1);
                      return (
                        <td
                          key={colIdx}
                          className={`py-3 ${
                            isDiagonal
                              ? 'bg-emerald-500/10 text-emerald-400 font-bold'
                              : 'text-muted-foreground'
                          }`}
                        >
                          <div>{val.toLocaleString()}</div>
                          <div className="text-[10px] opacity-75">({pct}%)</div>
                        </td>
                      );
                    })}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </CardContent>
    </Card>
  );
}
