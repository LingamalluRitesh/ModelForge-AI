'use client';

import React, { useState } from 'react';
import { Card, CardHeader, CardTitle, CardContent, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Cpu, Play, CheckCircle2, Server, Activity, ArrowUpRight } from 'lucide-react';

export function DistributedTrainingStudio() {
  const [activeJobs] = useState([
    { id: 'job_ray_901', name: 'TabNet-Production-Search', framework: 'Ray Train (4 Nodes)', gpus: 8, status: 'RUNNING', elapsed: '14m 20s', samplesPerSec: 14200 },
    { id: 'job_ddp_402', name: 'VisionTransformer-Backbone', framework: 'PyTorch DDP (8 Nodes)', gpus: 16, status: 'COMPLETED', elapsed: '1h 05m', samplesPerSec: 28400 },
    { id: 'job_ds_108', name: 'LLM-Instruction-FineTune', framework: 'DeepSpeed ZeRO-3', gpus: 32, status: 'COMPLETED', elapsed: '3h 45m', samplesPerSec: 4800 },
  ]);

  return (
    <div className="space-y-6">
      {/* Top Banner */}
      <div className="grid grid-cols-1 sm:grid-cols-4 gap-4">
        <Card className="border border-border/60 bg-card/50 backdrop-blur p-4">
          <div className="flex items-center justify-between">
            <span className="text-xs text-muted-foreground font-medium">Distributed GPU Nodes</span>
            <Server className="h-4 w-4 text-primary" />
          </div>
          <div className="mt-2 text-2xl font-bold font-mono text-primary">56 GPUs Active</div>
          <div className="mt-1 text-[11px] text-muted-foreground">Ray Core & PyTorch DDP Cluster</div>
        </Card>

        <Card className="border border-border/60 bg-card/50 backdrop-blur p-4">
          <div className="flex items-center justify-between">
            <span className="text-xs text-muted-foreground font-medium">Cluster Training Throughput</span>
            <Activity className="h-4 w-4 text-emerald-400" />
          </div>
          <div className="mt-2 text-2xl font-bold font-mono text-emerald-400">47,400 samples/s</div>
          <div className="mt-1 text-[11px] text-muted-foreground">Ring-AllReduce bandwidth saturated</div>
        </Card>

        <Card className="border border-border/60 bg-card/50 backdrop-blur p-4">
          <div className="flex items-center justify-between">
            <span className="text-xs text-muted-foreground font-medium">AllReduce Step Latency</span>
            <Badge className="bg-sky-500/10 text-sky-400 border-sky-500/20 text-[10px]">Sub-1ms</Badge>
          </div>
          <div className="mt-2 text-2xl font-bold font-mono text-sky-400">0.82 ms</div>
          <div className="mt-1 text-[11px] text-muted-foreground">AWS EFA / NCCL Direct Interconnect</div>
        </Card>

        <Card className="border border-border/60 bg-card/50 backdrop-blur p-4">
          <div className="flex items-center justify-between">
            <span className="text-xs text-muted-foreground font-medium">Spot GPU Interruption Recovery</span>
            <Cpu className="h-4 w-4 text-purple-400" />
          </div>
          <div className="mt-2 text-2xl font-bold font-mono text-purple-400">&lt; 15s Resumption</div>
          <div className="mt-1 text-[11px] text-muted-foreground">Zero-loss asynchronous checkpointing</div>
        </Card>
      </div>

      {/* Distributed Jobs Table */}
      <Card className="border border-border/60 bg-card/50 backdrop-blur">
        <CardHeader className="flex flex-row items-center justify-between pb-2">
          <div>
            <CardTitle className="text-base font-semibold">Active Distributed Training Clusters</CardTitle>
            <CardDescription className="text-xs">
              Multi-node PyTorch Distributed Data Parallel (DDP), Ray Train, and DeepSpeed ZeRO memory-partitioned runs.
            </CardDescription>
          </div>
          <Button size="sm" className="gap-1.5 text-xs">
            <Play className="h-3.5 w-3.5 fill-current" />
            Launch Multi-Node Job
          </Button>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs font-mono">
              <thead>
                <tr className="border-b border-border/60 text-muted-foreground font-sans">
                  <th className="pb-2 font-medium">Job Name</th>
                  <th className="pb-2 font-medium">Framework</th>
                  <th className="pb-2 font-medium">GPU Allocation</th>
                  <th className="pb-2 font-medium">Throughput</th>
                  <th className="pb-2 font-medium">Elapsed</th>
                  <th className="pb-2 font-medium">Status</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border/30">
                {activeJobs.map((j) => (
                  <tr key={j.id} className="hover:bg-muted/30">
                    <td className="py-2.5 font-sans font-semibold text-foreground">{j.name}</td>
                    <td className="py-2.5 text-muted-foreground">{j.framework}</td>
                    <td className="py-2.5 text-primary font-bold">{j.gpus} GPUs</td>
                    <td className="py-2.5 text-emerald-400">{j.samplesPerSec.toLocaleString()} s/s</td>
                    <td className="py-2.5 text-muted-foreground">{j.elapsed}</td>
                    <td className="py-2.5">
                      <Badge className={j.status === 'RUNNING' ? 'bg-sky-500/10 text-sky-400 border-sky-500/20 text-[10px]' : 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20 text-[10px]'}>
                        {j.status}
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
