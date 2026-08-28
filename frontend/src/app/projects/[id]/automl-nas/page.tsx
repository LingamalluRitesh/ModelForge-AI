'use client';

import React from 'react';
import { Card, CardHeader, CardTitle, CardContent, CardDescription } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Cpu, Sparkles, Trophy, GitFork } from 'lucide-react';

export default function AutoMLNASPage({ params }: { params: { id: string } }) {
  const nasTrials = [
    { id: 'darts_trial_14', arch: 'DARTS-Cell-Inception-v3', acc: '94.8%', f1: '0.892', paramsM: '4.2M', latencyMs: '2.4 ms', isPareto: true },
    { id: 'darts_trial_08', arch: 'DARTS-Cell-ResNet-v2', acc: '93.9%', f1: '0.881', paramsM: '2.8M', latencyMs: '1.8 ms', isPareto: true },
    { id: 'darts_trial_22', arch: 'DARTS-Cell-Dense-v1', acc: '95.1%', f1: '0.897', paramsM: '8.6M', latencyMs: '4.8 ms', isPareto: false },
    { id: 'darts_trial_03', arch: 'DARTS-Cell-Mobile-v1', acc: '91.4%', f1: '0.849', paramsM: '1.2M', latencyMs: '0.9 ms', isPareto: true },
  ];

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">Differentiable Architecture Search (DARTS)</h1>
          <p className="text-sm text-muted-foreground">
            Multi-objective Pareto frontier discovery optimizing accuracy against hardware latency budgets.
          </p>
        </div>

        <Button size="sm" className="gap-1.5 text-xs">
          <Sparkles className="h-3.5 w-3.5" />
          Launch NAS Exploration
        </Button>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-4 gap-4">
        <Card className="border border-border/60 bg-card/50 backdrop-blur p-4">
          <div className="flex items-center justify-between">
            <span className="text-xs text-muted-foreground font-medium">Search Space Trials</span>
            <Cpu className="h-4 w-4 text-primary" />
          </div>
          <div className="mt-2 text-2xl font-bold font-mono text-primary">32 Candidates</div>
          <div className="mt-1 text-[11px] text-muted-foreground">Continuous bi-level gradient optimization</div>
        </Card>

        <Card className="border border-border/60 bg-card/50 backdrop-blur p-4">
          <div className="flex items-center justify-between">
            <span className="text-xs text-muted-foreground font-medium">Pareto Optimal Genotypes</span>
            <Trophy className="h-4 w-4 text-amber-400" />
          </div>
          <div className="mt-2 text-2xl font-bold font-mono text-amber-400">3 Architectures</div>
          <div className="mt-1 text-[11px] text-muted-foreground">Non-dominated accuracy-latency frontier</div>
        </Card>

        <Card className="border border-border/60 bg-card/50 backdrop-blur p-4">
          <div className="flex items-center justify-between">
            <span className="text-xs text-muted-foreground font-medium">Best Accuracy Champion</span>
            <Badge className="bg-emerald-500/10 text-emerald-400 border-emerald-500/20 text-[10px]">95.1%</Badge>
          </div>
          <div className="mt-2 text-2xl font-bold font-mono text-emerald-400">DARTS-Dense-v1</div>
          <div className="mt-1 text-[11px] text-muted-foreground">F1 Score: 0.897</div>
        </Card>

        <Card className="border border-border/60 bg-card/50 backdrop-blur p-4">
          <div className="flex items-center justify-between">
            <span className="text-xs text-muted-foreground font-medium">Edge Micro-Model</span>
            <GitFork className="h-4 w-4 text-sky-400" />
          </div>
          <div className="mt-2 text-2xl font-bold font-mono text-sky-400">0.9 ms Latency</div>
          <div className="mt-1 text-[11px] text-muted-foreground">1.2M Parameters (INT8 quantizable)</div>
        </Card>
      </div>

      {/* NAS Trials Table */}
      <Card className="border border-border/60 bg-card/50 backdrop-blur">
        <CardHeader className="pb-2">
          <CardTitle className="text-base font-semibold">Evaluated Neural Architecture Candidates</CardTitle>
          <CardDescription className="text-xs">
            Discovered micro-cell DAG topologies, parameter size scaling, and inference latencies.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs font-mono">
              <thead>
                <tr className="border-b border-border/60 text-muted-foreground font-sans">
                  <th className="pb-2 font-medium">Trial ID</th>
                  <th className="pb-2 font-medium">Genotype Architecture</th>
                  <th className="pb-2 font-medium">Validation Acc</th>
                  <th className="pb-2 font-medium">F1-Score</th>
                  <th className="pb-2 font-medium">Model Size</th>
                  <th className="pb-2 font-medium">P99 Latency</th>
                  <th className="pb-2 font-medium">Pareto Frontier</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border/30">
                {nasTrials.map((t, idx) => (
                  <tr key={idx} className="hover:bg-muted/30">
                    <td className="py-2.5 font-sans font-semibold text-foreground">{t.id}</td>
                    <td className="py-2.5 text-muted-foreground">{t.arch}</td>
                    <td className="py-2.5 text-emerald-400 font-bold">{t.acc}</td>
                    <td className="py-2.5 text-primary">{t.f1}</td>
                    <td className="py-2.5">{t.paramsM}</td>
                    <td className="py-2.5 text-sky-400">{t.latencyMs}</td>
                    <td className="py-2.5">
                      {t.isPareto ? (
                        <Badge className="bg-amber-500/10 text-amber-400 border-amber-500/20 text-[10px]">
                          Pareto Optimal
                        </Badge>
                      ) : (
                        <Badge variant="outline" className="text-muted-foreground text-[10px]">
                          Dominated
                        </Badge>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
