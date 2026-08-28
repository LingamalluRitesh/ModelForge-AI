'use client';

import React, { useState } from 'react';
import { Card, CardHeader, CardTitle, CardContent, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Switch } from '@/components/ui/switch';
import { Input } from '@/components/ui/input';
import { AlertCircle, CheckCircle2, Play, RefreshCw, Sparkles, Zap } from 'lucide-react';

interface RetrainingPolicy {
  id: string;
  name: string;
  targetModel: string;
  triggerType: 'DRIFT_BASED' | 'CRON_SCHEDULED' | 'PERFORMANCE_DEGRADATION';
  driftMetric: 'PSI' | 'KS' | 'WASSERSTEIN';
  driftThreshold: number;
  qualityGateF1Threshold: number;
  autoDeployCanary: boolean;
  canaryInitialPercentage: number;
  isActive: boolean;
  lastRetrainedAt: string;
}

export function ContinuousRetrainingStudio() {
  const [policies, setPolicies] = useState<RetrainingPolicy[]>([
    {
      id: 'pol_1',
      name: 'Credit Risk XGBoost Drift Trigger',
      targetModel: 'Risk-XGBoost-v2',
      triggerType: 'DRIFT_BASED',
      driftMetric: 'PSI',
      driftThreshold: 0.20,
      qualityGateF1Threshold: 0.85,
      autoDeployCanary: true,
      canaryInitialPercentage: 10.0,
      isActive: true,
      lastRetrainedAt: '2 hours ago (Auto-Triggered)',
    },
    {
      id: 'pol_2',
      name: 'E-Commerce Weekly Retraining Pipeline',
      targetModel: 'Recommender-SVD-v3',
      triggerType: 'CRON_SCHEDULED',
      driftMetric: 'KS',
      driftThreshold: 0.05,
      qualityGateF1Threshold: 0.80,
      autoDeployCanary: false,
      canaryInitialPercentage: 0.0,
      isActive: true,
      lastRetrainedAt: '3 days ago',
    },
    {
      id: 'pol_3',
      name: 'Transaction Fraud TabNet Quality Gate',
      targetModel: 'Fraud-TabNet-v1',
      triggerType: 'PERFORMANCE_DEGRADATION',
      driftMetric: 'WASSERSTEIN',
      driftThreshold: 0.15,
      qualityGateF1Threshold: 0.90,
      autoDeployCanary: true,
      canaryInitialPercentage: 5.0,
      isActive: false,
      lastRetrainedAt: '1 week ago',
    },
  ]);

  const [isExecuting, setIsExecuting] = useState(false);
  const [executionLog, setExecutionLog] = useState<string[]>([]);

  const handleManualTrigger = (policyName: string) => {
    setIsExecuting(true);
    setExecutionLog([
      `[INIT] Triggering automated retraining workflow for "${policyName}"...`,
      `[STEP 1/5] Ingesting golden validation split & fresh streaming batch...`,
      `[STEP 2/5] Running meta-learning & hyperparameter optimization (Bayesian TPE)...`,
      `[STEP 3/5] Evaluating candidate model against Quality Gate SLAs (F1 >= 0.85)...`,
      `[STEP 4/5] Quality Gate PASSED. Registering candidate model artifact v2.4.1...`,
      `[STEP 5/5] Initializing Istio Canary endpoint routing (10% Traffic Split)...`,
      `[SUCCESS] Continuous Retraining pipeline executed in 4.82s. Zero downtime.`,
    ]);
    setTimeout(() => setIsExecuting(false), 800);
  };

  const togglePolicy = (id: string) => {
    setPolicies(policies.map((p) => (p.id === id ? { ...p, isActive: !p.isActive } : p)));
  };

  return (
    <div className="space-y-6">
      <Card className="border border-border/60 bg-card/50 backdrop-blur">
        <CardHeader className="flex flex-row items-center justify-between pb-2">
          <div>
            <CardTitle className="text-base font-semibold flex items-center gap-2">
              <Zap className="h-4 w-4 text-amber-400" />
              Automated Retraining Policies & Quality Gates
            </CardTitle>
            <CardDescription className="text-xs">
              Autonomous drift-triggered training loops, pre-deployment validation gates, and automated Canary rollouts.
            </CardDescription>
          </div>
          <Badge variant="outline" className="border-emerald-500/40 text-emerald-400 font-mono text-xs">
            3 Active Retraining Daemons
          </Badge>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {policies.map((policy) => (
              <div
                key={policy.id}
                className="flex flex-col md:flex-row items-start md:items-center justify-between rounded-lg border border-border/50 bg-background/50 p-4 gap-4"
              >
                <div className="space-y-1">
                  <div className="flex items-center gap-2">
                    <span className="font-semibold text-sm text-foreground">{policy.name}</span>
                    <Badge variant="secondary" className="font-mono text-[10px]">
                      {policy.triggerType}
                    </Badge>
                    {policy.autoDeployCanary && (
                      <Badge className="bg-sky-500/10 text-sky-400 border-sky-500/20 text-[10px]">
                        Canary Split ({policy.canaryInitialPercentage}%)
                      </Badge>
                    )}
                  </div>
                  <div className="flex flex-wrap items-center gap-3 text-xs text-muted-foreground">
                    <span>Target Model: <strong className="text-foreground">{policy.targetModel}</strong></span>
                    <span>•</span>
                    <span>Drift Threshold: <strong className="text-amber-400 font-mono">{policy.driftMetric} &ge; {policy.driftThreshold}</strong></span>
                    <span>•</span>
                    <span>Quality Gate SLA: <strong className="text-emerald-400 font-mono">F1 &ge; {policy.qualityGateF1Threshold}</strong></span>
                    <span>•</span>
                    <span>Last Run: {policy.lastRetrainedAt}</span>
                  </div>
                </div>

                <div className="flex items-center gap-3 w-full md:w-auto justify-end">
                  <div className="flex items-center gap-2 mr-2">
                    <span className="text-xs text-muted-foreground">{policy.isActive ? 'Enabled' : 'Paused'}</span>
                    <Switch checked={policy.isActive} onCheckedChange={() => togglePolicy(policy.id)} />
                  </div>

                  <Button
                    size="sm"
                    variant="outline"
                    className="gap-1 text-xs border-primary/40 hover:bg-primary/10"
                    onClick={() => handleManualTrigger(policy.name)}
                    disabled={isExecuting}
                  >
                    <Play className="h-3 w-3 text-primary fill-primary" />
                    Run Now
                  </Button>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Execution Console Output */}
      {executionLog.length > 0 && (
        <Card className="border border-border/60 bg-black/60 font-mono text-xs shadow-2xl backdrop-blur">
          <CardHeader className="py-2.5 border-b border-border/40 flex flex-row items-center justify-between">
            <div className="flex items-center gap-2 text-primary font-semibold">
              <RefreshCw className={`h-3.5 w-3.5 ${isExecuting ? 'animate-spin' : ''}`} />
              Orchestrator Execution Console
            </div>
            <Badge variant="outline" className="border-primary/30 text-primary text-[10px]">
              Active Pipeline Stream
            </Badge>
          </CardHeader>
          <CardContent className="py-3 space-y-1.5 text-slate-300">
            {executionLog.map((log, i) => (
              <div key={i} className={log.includes('[SUCCESS]') ? 'text-emerald-400 font-bold' : 'text-slate-300'}>
                {log}
              </div>
            ))}
          </CardContent>
        </Card>
      )}
    </div>
  );
}
