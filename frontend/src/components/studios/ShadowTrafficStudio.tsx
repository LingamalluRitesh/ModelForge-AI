'use client';

import React, { useState } from 'react';
import { Card, CardHeader, CardTitle, CardContent, CardDescription } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Eye, Zap, RefreshCw, CheckCircle2, ArrowRight } from 'lucide-react';

export function ShadowTrafficStudio() {
  const [mirroredRequests, setMirroredRequests] = useState(148200);
  const [discrepancyRate, setDiscrepancyRate] = useState(0.014);

  return (
    <div className="space-y-6">
      {/* Top Banner */}
      <div className="grid grid-cols-1 sm:grid-cols-4 gap-4">
        <Card className="border border-border/60 bg-card/50 backdrop-blur p-4">
          <div className="flex items-center justify-between">
            <span className="text-xs text-muted-foreground font-medium">Dark Shadow Traffic Mirror</span>
            <Eye className="h-4 w-4 text-primary" />
          </div>
          <div className="mt-2 text-2xl font-bold font-mono text-primary">100% Mirroring</div>
          <div className="mt-1 text-[11px] text-muted-foreground">{mirroredRequests.toLocaleString()} total dark requests</div>
        </Card>

        <Card className="border border-border/60 bg-card/50 backdrop-blur p-4">
          <div className="flex items-center justify-between">
            <span className="text-xs text-muted-foreground font-medium">Output Divergence Rate</span>
            <Badge className="bg-emerald-500/10 text-emerald-400 border-emerald-500/20 text-[10px]">Within SLA</Badge>
          </div>
          <div className="mt-2 text-2xl font-bold font-mono text-emerald-400">{(discrepancyRate * 100).toFixed(2)}%</div>
          <div className="mt-1 text-[11px] text-muted-foreground">Target divergence threshold &le; 2.0%</div>
        </Card>

        <Card className="border border-border/60 bg-card/50 backdrop-blur p-4">
          <div className="flex items-center justify-between">
            <span className="text-xs text-muted-foreground font-medium">Primary Champion Latency</span>
            <Zap className="h-4 w-4 text-sky-400" />
          </div>
          <div className="mt-2 text-2xl font-bold font-mono text-sky-400">4.2 ms (p99)</div>
          <div className="mt-1 text-[11px] text-muted-foreground">Risk-XGBoost-v2.3 (Current Prod)</div>
        </Card>

        <Card className="border border-border/60 bg-card/50 backdrop-blur p-4">
          <div className="flex items-center justify-between">
            <span className="text-xs text-muted-foreground font-medium">Shadow Challenger Latency</span>
            <Zap className="h-4 w-4 text-purple-400" />
          </div>
          <div className="mt-2 text-2xl font-bold font-mono text-purple-400">2.1 ms (p99)</div>
          <div className="mt-1 text-[11px] text-emerald-400 font-mono">50% Faster Throughput</div>
        </Card>
      </div>

      {/* Promotion Action Card */}
      <Card className="border border-border/60 bg-card/50 backdrop-blur p-4 flex flex-col sm:flex-row items-center justify-between gap-4">
        <div>
          <h3 className="font-semibold text-sm text-foreground">Challenger Ready for Automated Canary Promotion</h3>
          <p className="text-xs text-muted-foreground">
            Shadow candidate has processed 100,000+ live requests with zero crashes and superior latency characteristics.
          </p>
        </div>
        <Button size="sm" className="gap-1.5 text-xs bg-emerald-600 hover:bg-emerald-500">
          <CheckCircle2 className="h-3.5 w-3.5" />
          Promote to 10% Canary Split
        </Button>
      </Card>
    </div>
  );
}
