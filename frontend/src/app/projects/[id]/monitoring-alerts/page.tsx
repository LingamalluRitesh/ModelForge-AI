'use client';

import React from 'react';
import { Card, CardHeader, CardTitle, CardContent, CardDescription } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Bell, AlertTriangle, CheckCircle, Radio } from 'lucide-react';

export default function MonitoringAlertsPage({ params }: { params: { id: string } }) {
  const activeAlerts = [
    { rule: 'PSI Population Stability Drift', entity: 'credit_risk_v2', severity: 'WARNING', metric: 'PSI = 0.18 (Threshold 0.20)', time: '10m ago' },
    { rule: 'P99 Inference Latency SLA', entity: 'endpoint /v1/predict', severity: 'NORMAL', metric: '4.2 ms (SLA 10.0 ms)', time: 'Live' },
    { rule: 'Prediction Distribution KL Divergence', entity: 'fraud_detector', severity: 'NORMAL', metric: 'KL = 0.021 (Threshold 0.10)', time: 'Live' },
  ];

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Real-Time Observability & Monitoring Alerts</h1>
        <p className="text-sm text-muted-foreground">
          Autonomous CUSUM, Page-Hinkley, and ADWIN statistical change point triggers with PagerDuty integration.
        </p>
      </div>

      {/* Alerts Table */}
      <Card className="border border-border/60 bg-card/50 backdrop-blur">
        <CardHeader className="pb-2">
          <CardTitle className="text-base font-semibold flex items-center gap-2">
            <Radio className="h-4 w-4 text-emerald-400 animate-pulse" />
            Live Monitored Telemetry Channels
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs font-mono">
              <thead>
                <tr className="border-b border-border/60 text-muted-foreground font-sans">
                  <th className="pb-2 font-medium">Alert Rule</th>
                  <th className="pb-2 font-medium">Target Entity</th>
                  <th className="pb-2 font-medium">Current Metric Value</th>
                  <th className="pb-2 font-medium">Time</th>
                  <th className="pb-2 font-medium">Status</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border/30">
                {activeAlerts.map((a, idx) => (
                  <tr key={idx} className="hover:bg-muted/30">
                    <td className="py-2.5 font-sans font-semibold text-foreground">{a.rule}</td>
                    <td className="py-2.5 text-primary">{a.entity}</td>
                    <td className="py-2.5 text-foreground">{a.metric}</td>
                    <td className="py-2.5 text-muted-foreground">{a.time}</td>
                    <td className="py-2.5">
                      <Badge className={a.severity === 'WARNING' ? 'bg-amber-500/10 text-amber-400 border-amber-500/20 text-[10px]' : 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20 text-[10px]'}>
                        {a.severity}
                      </Badge>
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
