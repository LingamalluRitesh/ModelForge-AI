'use client';

import React, { useState } from 'react';
import { Card, CardHeader, CardTitle, CardContent, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { ROCComparisonChart } from '@/components/visualizations/ROCComparisonChart';
import { CalibrationCurvePlot } from '@/components/visualizations/CalibrationCurvePlot';
import { Trophy, Zap, Cpu, Award } from 'lucide-react';

export function BenchmarksStudio() {
  const leaderboard = [
    { rank: 1, name: 'XGBoost-Hist-v2.4', auc: 0.942, f1: 0.884, latencyMs: 2.1, memoryMb: 18.4, status: 'CHAMPION' },
    { rank: 2, name: 'TabNet-Attentive-v1.8', auc: 0.928, f1: 0.865, latencyMs: 4.8, memoryMb: 42.1, status: 'CHALLENGER' },
    { rank: 3, name: 'NODE-Oblivious-v1.2', auc: 0.915, f1: 0.851, latencyMs: 5.2, memoryMb: 36.8, status: 'EVALUATED' },
    { rank: 4, name: 'RandomForest-Scratch-v2', auc: 0.898, f1: 0.832, latencyMs: 8.4, memoryMb: 64.0, status: 'BASELINE' },
  ];

  return (
    <div className="space-y-6">
      {/* Top Banner */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <Card className="border border-border/60 bg-card/50 backdrop-blur p-4">
          <div className="flex items-center justify-between">
            <span className="text-xs text-muted-foreground font-medium">Champion Model</span>
            <Trophy className="h-4 w-4 text-amber-400" />
          </div>
          <div className="mt-2 text-xl font-bold text-foreground">XGBoost-Hist-v2.4</div>
          <div className="mt-1 text-[11px] text-emerald-400 font-mono">ROC-AUC 0.942 • F1 0.884</div>
        </Card>

        <Card className="border border-border/60 bg-card/50 backdrop-blur p-4">
          <div className="flex items-center justify-between">
            <span className="text-xs text-muted-foreground font-medium">Fastest Sub-5ms Inference</span>
            <Zap className="h-4 w-4 text-sky-400" />
          </div>
          <div className="mt-2 text-xl font-bold text-sky-400 font-mono">2.1 ms (p99)</div>
          <div className="mt-1 text-[11px] text-muted-foreground">ONNX Runtime graph accelerated</div>
        </Card>

        <Card className="border border-border/60 bg-card/50 backdrop-blur p-4">
          <div className="flex items-center justify-between">
            <span className="text-xs text-muted-foreground font-medium">Memory Footprint</span>
            <Cpu className="h-4 w-4 text-purple-400" />
          </div>
          <div className="mt-2 text-xl font-bold text-purple-400 font-mono">18.4 MB</div>
          <div className="mt-1 text-[11px] text-muted-foreground">Fits on ultra-low edge devices</div>
        </Card>
      </div>

      {/* Leaderboard Table */}
      <Card className="border border-border/60 bg-card/50 backdrop-blur">
        <CardHeader className="flex flex-row items-center justify-between pb-2">
          <div>
            <CardTitle className="text-base font-semibold flex items-center gap-2">
              <Award className="h-4 w-4 text-amber-400" />
              Automated Model Leaderboard & Quality Arena
            </CardTitle>
            <CardDescription className="text-xs">
              Rigorous comparative benchmarking against holdout test partitions with latency and memory profiling.
            </CardDescription>
          </div>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs font-mono">
              <thead>
                <tr className="border-b border-border/60 text-muted-foreground font-sans">
                  <th className="pb-2 font-medium">Rank</th>
                  <th className="pb-2 font-medium">Model Candidate</th>
                  <th className="pb-2 font-medium">ROC-AUC</th>
                  <th className="pb-2 font-medium">F1-Score</th>
                  <th className="pb-2 font-medium">P99 Latency</th>
                  <th className="pb-2 font-medium">RAM Footprint</th>
                  <th className="pb-2 font-medium">Status</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border/30">
                {leaderboard.map((m) => (
                  <tr key={m.rank} className="hover:bg-muted/30">
                    <td className="py-2.5 font-bold text-amber-400">#{m.rank}</td>
                    <td className="py-2.5 font-sans font-semibold text-foreground">{m.name}</td>
                    <td className="py-2.5 text-emerald-400 font-bold">{m.auc.toFixed(3)}</td>
                    <td className="py-2.5 text-primary">{m.f1.toFixed(3)}</td>
                    <td className="py-2.5 text-sky-400">{m.latencyMs} ms</td>
                    <td className="py-2.5 text-muted-foreground">{m.memoryMb} MB</td>
                    <td className="py-2.5">
                      <Badge className={m.status === 'CHAMPION' ? 'bg-amber-500/10 text-amber-400 border-amber-500/20 text-[10px]' : 'bg-muted/60 text-muted-foreground text-[10px]'}>
                        {m.status}
                      </Badge>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>

      {/* Visualizations */}
      <ROCComparisonChart />
      <CalibrationCurvePlot />
    </div>
  );
}
