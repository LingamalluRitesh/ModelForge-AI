'use client';

import React, { useState } from 'react';
import { Card, CardHeader, CardTitle, CardContent, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { GitBranch, Play, CheckCircle2, Clock, AlertCircle } from 'lucide-react';

interface PipelineStageNode {
  id: string;
  name: string;
  duration: string;
  status: 'COMPLETED' | 'RUNNING' | 'PENDING' | 'FAILED';
  outputs: string[];
}

export function PipelineOrchestrationStudio() {
  const [stages] = useState<PipelineStageNode[]>([
    {
      id: 'stage_1',
      name: 'Data Validation & Profiling',
      duration: '1.2s',
      status: 'COMPLETED',
      outputs: ['Great Expectations Suite', 'Dataset Profile JSON'],
    },
    {
      id: 'stage_2',
      name: 'Feature Store AS-OF Point-In-Time Join',
      duration: '2.4s',
      status: 'COMPLETED',
      outputs: ['Parquet Golden Split', 'Zero Lookahead Leakage'],
    },
    {
      id: 'stage_3',
      name: 'Multi-Algorithm Bayesian HPO',
      duration: '4.8s',
      status: 'COMPLETED',
      outputs: ['XGBoost (AUC 0.942)', 'TabNet (AUC 0.895)'],
    },
    {
      id: 'stage_4',
      name: 'Automated Debiasing & Explainability',
      duration: '2.1s',
      status: 'COMPLETED',
      outputs: ['Equalized Odds Repaired', 'TreeSHAP Attributions'],
    },
    {
      id: 'stage_5',
      name: 'Governance Quality Gate Verification',
      duration: '0.8s',
      status: 'COMPLETED',
      outputs: ['F1 >= 0.85 (Passed)', 'Candidate Approved'],
    },
    {
      id: 'stage_6',
      name: 'Istio Service Mesh Canary Rollout',
      duration: '1.5s',
      status: 'COMPLETED',
      outputs: ['10% Canary Weight', 'Real-time Telemetry Synced'],
    },
  ]);

  return (
    <div className="space-y-6">
      <Card className="border border-border/60 bg-card/50 backdrop-blur">
        <CardHeader className="flex flex-row items-center justify-between pb-2">
          <div>
            <CardTitle className="text-base font-semibold flex items-center gap-2">
              <GitBranch className="h-4 w-4 text-primary" />
              DAG Directed Acyclic Graph Pipeline Execution
            </CardTitle>
            <CardDescription className="text-xs">
              End-to-end continuous training, validation, explainability, and Istio deployment DAG graph.
            </CardDescription>
          </div>
          <Button size="sm" className="gap-1.5 text-xs">
            <Play className="h-3 w-3 fill-current" />
            Trigger Full Pipeline
          </Button>
        </CardHeader>
        <CardContent>
          <div className="space-y-3 py-2">
            {stages.map((stage, idx) => (
              <div
                key={stage.id}
                className="flex items-center justify-between rounded-lg border border-border/50 bg-background/50 p-3 hover:border-primary/40 transition"
              >
                <div className="flex items-center gap-3">
                  <div className="flex h-7 w-7 items-center justify-center rounded-full bg-emerald-500/10 text-emerald-400 text-xs font-bold font-mono">
                    <CheckCircle2 className="h-4 w-4" />
                  </div>
                  <div>
                    <div className="text-xs font-semibold text-foreground">
                      {idx + 1}. {stage.name}
                    </div>
                    <div className="flex gap-2 text-[11px] text-muted-foreground mt-0.5">
                      {stage.outputs.map((out, outIdx) => (
                        <span key={outIdx} className="rounded bg-muted/60 px-1.5 py-0.2 font-mono text-[10px]">
                          {out}
                        </span>
                      ))}
                    </div>
                  </div>
                </div>

                <div className="flex items-center gap-3 text-xs font-mono">
                  <span className="text-muted-foreground flex items-center gap-1">
                    <Clock className="h-3 w-3" />
                    {stage.duration}
                  </span>
                  <Badge className="bg-emerald-500/10 text-emerald-400 border-emerald-500/20 text-[10px]">
                    Success
                  </Badge>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
