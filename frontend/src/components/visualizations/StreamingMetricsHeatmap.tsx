'use client';

import React from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';

interface StreamingWindow {
  windowId: string;
  psi_drift: number;
  ks_stat: number;
  throughput_qps: number;
  p99_latency_ms: number;
  status: 'HEALTHY' | 'WARNING' | 'CRITICAL';
}

export function StreamingMetricsHeatmap() {
  const windows: StreamingWindow[] = [
    { windowId: 'T-00m..05m', psi_drift: 0.024, ks_stat: 0.015, throughput_qps: 1420, p99_latency_ms: 6.8, status: 'HEALTHY' },
    { windowId: 'T-05m..10m', psi_drift: 0.031, ks_stat: 0.022, throughput_qps: 1680, p99_latency_ms: 7.2, status: 'HEALTHY' },
    { windowId: 'T-10m..15m', psi_drift: 0.048, ks_stat: 0.038, throughput_qps: 1950, p99_latency_ms: 8.5, status: 'HEALTHY' },
    { windowId: 'T-15m..20m', psi_drift: 0.112, ks_stat: 0.089, throughput_qps: 2200, p99_latency_ms: 11.4, status: 'WARNING' },
    { windowId: 'T-20m..25m', psi_drift: 0.245, ks_stat: 0.184, throughput_qps: 1850, p99_latency_ms: 14.1, status: 'CRITICAL' },
    { windowId: 'T-25m..30m', psi_drift: 0.082, ks_stat: 0.061, throughput_qps: 1540, p99_latency_ms: 9.0, status: 'HEALTHY' },
  ];

  const getStatusBadge = (status: StreamingWindow['status']) => {
    switch (status) {
      case 'HEALTHY':
        return <Badge className="bg-emerald-500/10 text-emerald-400 border-emerald-500/20">Normal</Badge>;
      case 'WARNING':
        return <Badge className="bg-amber-500/10 text-amber-400 border-amber-500/20">Moderate Shift</Badge>;
      case 'CRITICAL':
        return <Badge className="bg-rose-500/10 text-rose-400 border-rose-500/20">Drift Breach</Badge>;
    }
  };

  return (
    <Card className="border border-border/60 bg-card/50 backdrop-blur">
      <CardHeader className="flex flex-row items-center justify-between pb-2">
        <div>
          <CardTitle className="text-base font-semibold">Tumbling Window Stream Telemetry</CardTitle>
          <p className="text-xs text-muted-foreground">5-Minute Tumbling Aggregations for Real-Time Feature Ingestion Drift</p>
        </div>
        <Badge variant="outline" className="border-sky-500/40 text-sky-400 font-mono text-xs">
          Live Tumbling Windows (N=6)
        </Badge>
      </CardHeader>
      <CardContent>
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead>
              <tr className="border-b border-border/60 text-muted-foreground">
                <th className="pb-2 font-medium">Time Window</th>
                <th className="pb-2 font-medium">PSI Drift Score</th>
                <th className="pb-2 font-medium">KS Statistic</th>
                <th className="pb-2 font-medium">Throughput (QPS)</th>
                <th className="pb-2 font-medium">P99 Latency</th>
                <th className="pb-2 font-medium">Drift Health</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border/30 font-mono">
              {windows.map((w, idx) => (
                <tr key={idx} className="hover:bg-muted/30">
                  <td className="py-2.5 font-semibold text-foreground">{w.windowId}</td>
                  <td className="py-2.5">
                    <span className={w.psi_drift > 0.2 ? 'text-rose-400 font-bold' : w.psi_drift > 0.1 ? 'text-amber-400' : 'text-emerald-400'}>
                      {w.psi_drift.toFixed(3)}
                    </span>
                  </td>
                  <td className="py-2.5 text-muted-foreground">{w.ks_stat.toFixed(3)}</td>
                  <td className="py-2.5 text-primary">{w.throughput_qps.toLocaleString()}</td>
                  <td className="py-2.5 text-foreground">{w.p99_latency_ms} ms</td>
                  <td className="py-2.5">{getStatusBadge(w.status)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </CardContent>
    </Card>
  );
}
