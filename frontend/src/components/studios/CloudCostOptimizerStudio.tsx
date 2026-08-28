'use client';

import React, { useState } from 'react';
import { Card, CardHeader, CardTitle, CardContent, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { DollarSign, TrendingDown, Cpu, Server, Zap, ArrowDownRight } from 'lucide-react';
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid, Legend } from 'recharts';

interface CostBreakdown {
  month: string;
  unoptimizedCost: number;
  optimizedCost: number;
  quantizationSavings: number;
  spotSavings: number;
}

export function CloudCostOptimizerStudio() {
  const data: CostBreakdown[] = [
    { month: 'Apr', unoptimizedCost: 4800, optimizedCost: 2400, quantizationSavings: 1400, spotSavings: 1000 },
    { month: 'May', unoptimizedCost: 5200, optimizedCost: 2600, quantizationSavings: 1500, spotSavings: 1100 },
    { month: 'Jun', unoptimizedCost: 5900, optimizedCost: 2900, quantizationSavings: 1800, spotSavings: 1200 },
    { month: 'Jul', unoptimizedCost: 6400, optimizedCost: 3100, quantizationSavings: 2000, spotSavings: 1300 },
    { month: 'Aug', unoptimizedCost: 7100, optimizedCost: 3300, quantizationSavings: 2300, spotSavings: 1500 },
  ];

  return (
    <div className="space-y-6">
      {/* Top Metrics Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card className="border border-border/60 bg-card/50 backdrop-blur p-4">
          <div className="flex items-center justify-between">
            <span className="text-xs text-muted-foreground font-medium">Monthly Cloud Spend</span>
            <DollarSign className="h-4 w-4 text-emerald-400" />
          </div>
          <div className="mt-2 text-2xl font-bold font-mono">$3,300.00</div>
          <div className="mt-1 flex items-center gap-1 text-[11px] text-emerald-400">
            <TrendingDown className="h-3 w-3" />
            <span>-53.5% vs Unoptimized Baseline</span>
          </div>
        </Card>

        <Card className="border border-border/60 bg-card/50 backdrop-blur p-4">
          <div className="flex items-center justify-between">
            <span className="text-xs text-muted-foreground font-medium">ONNX INT8 Quantization Savings</span>
            <Zap className="h-4 w-4 text-sky-400" />
          </div>
          <div className="mt-2 text-2xl font-bold font-mono text-sky-400">$2,300/mo</div>
          <div className="mt-1 text-[11px] text-muted-foreground">3.8x Throughput Gain • -75% Memory</div>
        </Card>

        <Card className="border border-border/60 bg-card/50 backdrop-blur p-4">
          <div className="flex items-center justify-between">
            <span className="text-xs text-muted-foreground font-medium">Spot & Preemptible GPU Savings</span>
            <Server className="h-4 w-4 text-purple-400" />
          </div>
          <div className="mt-2 text-2xl font-bold font-mono text-purple-400">$1,500/mo</div>
          <div className="mt-1 text-[11px] text-muted-foreground">Automated checkpoint fault-tolerance</div>
        </Card>

        <Card className="border border-border/60 bg-card/50 backdrop-blur p-4">
          <div className="flex items-center justify-between">
            <span className="text-xs text-muted-foreground font-medium">Cost Per 1M Predictions</span>
            <Cpu className="h-4 w-4 text-amber-400" />
          </div>
          <div className="mt-2 text-2xl font-bold font-mono text-amber-400">$0.18</div>
          <div className="mt-1 text-[11px] text-muted-foreground">Sub-5ms serving on AWS G5 Nodes</div>
        </Card>
      </div>

      {/* Monthly Chart */}
      <Card className="border border-border/60 bg-card/50 backdrop-blur">
        <CardHeader className="flex flex-row items-center justify-between pb-2">
          <div>
            <CardTitle className="text-base font-semibold">Cloud Inference Cost Reduction Trajectory</CardTitle>
            <CardDescription className="text-xs">
              Direct infrastructure savings achieved via ONNX Runtime graph optimizations, INT8 Dynamic Quantization, and K8s HPA downscaling.
            </CardDescription>
          </div>
          <Badge variant="outline" className="border-emerald-500/40 text-emerald-400 font-mono text-xs">
            Total Saved YTD: $19,200
          </Badge>
        </CardHeader>
        <CardContent>
          <div className="h-[300px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={data} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.4} />
                <XAxis dataKey="month" stroke="#64748b" />
                <YAxis tickFormatter={(v) => `$${v}`} stroke="#64748b" />
                <Tooltip
                  content={({ active, payload }) => {
                    if (active && payload && payload.length) {
                      const item = payload[0].payload as CostBreakdown;
                      return (
                        <div className="rounded-lg border bg-popover/90 p-2.5 text-xs shadow-xl backdrop-blur">
                          <div className="font-semibold text-foreground">{item.month} Cost Comparison</div>
                          <div className="mt-1 text-rose-400 font-mono">Unoptimized: ${item.unoptimizedCost.toLocaleString()}</div>
                          <div className="text-emerald-400 font-mono font-bold">Optimized: ${item.optimizedCost.toLocaleString()}</div>
                          <div className="mt-1 border-t border-border/40 pt-1 text-sky-400 font-mono">
                            Quantization Savings: +${item.quantizationSavings.toLocaleString()}
                          </div>
                        </div>
                      );
                    }
                    return null;
                  }}
                />
                <Legend verticalAlign="top" height={36} />
                <Bar dataKey="unoptimizedCost" name="Unoptimized Baseline Cost" fill="#f43f5e" opacity={0.6} radius={[4, 4, 0, 0]} />
                <Bar dataKey="optimizedCost" name="ModelForge Optimized Cost" fill="#10b981" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
