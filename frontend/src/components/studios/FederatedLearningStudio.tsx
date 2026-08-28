'use client';

import React, { useState } from 'react';
import { Card, CardHeader, CardTitle, CardContent, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { ShieldAlert, Network, RefreshCw, Smartphone, Laptop, Server } from 'lucide-react';

export function FederatedLearningStudio() {
  const [currentRound, setCurrentRound] = useState(7);
  const [totalRounds] = useState(10);
  const [participatingNodes] = useState(14);
  const [dpEpsilon] = useState(1.0);
  const [averageLoss] = useState(0.0412);
  const [isAggregating, setIsAggregating] = useState(false);

  const edgeNodes = [
    { id: 'edge_us_east_1', deviceType: 'Edge Server', samples: 4500, loss: 0.038, status: 'COMPLETED' },
    { id: 'edge_eu_west_1', deviceType: 'Edge Cluster', samples: 3800, loss: 0.042, status: 'COMPLETED' },
    { id: 'edge_ap_south_1', deviceType: 'On-Prem Worker', samples: 5200, loss: 0.039, status: 'COMPLETED' },
    { id: 'client_mobile_fleet', deviceType: 'Mobile Cluster (12k Devices)', samples: 18000, loss: 0.045, status: 'SYNCED' },
  ];

  const handleAggregate = () => {
    setIsAggregating(true);
    setTimeout(() => {
      setCurrentRound((prev) => Math.min(totalRounds, prev + 1));
      setIsAggregating(false);
    }, 800);
  };

  return (
    <div className="space-y-6">
      {/* Top Metrics Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-4 gap-4">
        <Card className="border border-border/60 bg-card/50 backdrop-blur p-4">
          <div className="flex items-center justify-between">
            <span className="text-xs text-muted-foreground font-medium">Federated Round Progress</span>
            <Network className="h-4 w-4 text-primary" />
          </div>
          <div className="mt-2 text-2xl font-bold font-mono text-primary">
            Round {currentRound} / {totalRounds}
          </div>
          <div className="mt-1 text-[11px] text-muted-foreground">FedAvg Global Model Synchronization</div>
        </Card>

        <Card className="border border-border/60 bg-card/50 backdrop-blur p-4">
          <div className="flex items-center justify-between">
            <span className="text-xs text-muted-foreground font-medium">Decentralized Edge Nodes</span>
            <Server className="h-4 w-4 text-emerald-400" />
          </div>
          <div className="mt-2 text-2xl font-bold font-mono text-emerald-400">{participatingNodes} Nodes</div>
          <div className="mt-1 text-[11px] text-muted-foreground">31,500 total decentralized training samples</div>
        </Card>

        <Card className="border border-border/60 bg-card/50 backdrop-blur p-4">
          <div className="flex items-center justify-between">
            <span className="text-xs text-muted-foreground font-medium">Differential Privacy Guarantee</span>
            <ShieldAlert className="h-4 w-4 text-purple-400" />
          </div>
          <div className="mt-2 text-2xl font-bold font-mono text-purple-400">&epsilon; = {dpEpsilon.toFixed(1)}, &delta; = 10⁻⁵</div>
          <div className="mt-1 text-[11px] text-muted-foreground">Gaussian gradient noise perturbation</div>
        </Card>

        <Card className="border border-border/60 bg-card/50 backdrop-blur p-4">
          <div className="flex items-center justify-between">
            <span className="text-xs text-muted-foreground font-medium">Global Model Converged Loss</span>
            <Badge className="bg-sky-500/10 text-sky-400 border-sky-500/20 text-[10px]">Converged</Badge>
          </div>
          <div className="mt-2 text-2xl font-bold font-mono text-sky-400">{averageLoss.toFixed(4)}</div>
          <div className="mt-1 text-[11px] text-muted-foreground">-48.2% reduction across 7 rounds</div>
        </Card>
      </div>

      {/* Edge Fleet Table */}
      <Card className="border border-border/60 bg-card/50 backdrop-blur">
        <CardHeader className="flex flex-row items-center justify-between pb-2">
          <div>
            <CardTitle className="text-base font-semibold">Decentralized Edge Fleet Status</CardTitle>
            <CardDescription className="text-xs">
              Live federated participants computing local stochastic gradients without transmitting raw user data.
            </CardDescription>
          </div>
          <Button size="sm" onClick={handleAggregate} disabled={isAggregating} className="gap-1.5 text-xs">
            <RefreshCw className={`h-3.5 w-3.5 ${isAggregating ? 'animate-spin' : ''}`} />
            {isAggregating ? 'Aggregating...' : 'Aggregate Next Round'}
          </Button>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs font-mono">
              <thead>
                <tr className="border-b border-border/60 text-muted-foreground font-sans">
                  <th className="pb-2 font-medium">Participant ID</th>
                  <th className="pb-2 font-medium">Device Profile</th>
                  <th className="pb-2 font-medium">Local Training Samples</th>
                  <th className="pb-2 font-medium">Local Loss</th>
                  <th className="pb-2 font-medium">Status</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border/30">
                {edgeNodes.map((node, idx) => (
                  <tr key={idx} className="hover:bg-muted/30">
                    <td className="py-2.5 font-sans font-semibold text-foreground">{node.id}</td>
                    <td className="py-2.5 text-muted-foreground">{node.deviceType}</td>
                    <td className="py-2.5 text-primary font-bold">{node.samples.toLocaleString()}</td>
                    <td className="py-2.5">{node.loss.toFixed(4)}</td>
                    <td className="py-2.5">
                      <Badge className="bg-emerald-500/10 text-emerald-400 border-emerald-500/20 text-[10px]">
                        {node.status}
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
