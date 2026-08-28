"use client";

import React, { useState } from "react";
import Link from "next/link";
import {
  Rocket,
  Sliders,
  RotateCcw,
  Activity,
  CheckCircle2,
  AlertTriangle,
  Play,
  Shield,
  Layers,
} from "lucide-react";
import { SectionCard } from "@/components/ui/Cards";
import { StageBadge, StatusBadge } from "@/components/ui/Badges";

interface DeploymentItem {
  id: string;
  name: string;
  endpointPath: string;
  environment: string;
  primaryVersion: string;
  canaryVersion?: string;
  strategy: string;
  canaryPercentage: number;
  replicas: number;
  status: string;
  isHealthy: boolean;
}

const mockDeployments: DeploymentItem[] = [
  {
    id: "dep-1",
    name: "Credit Card Fraud Real-Time Endpoint",
    endpointPath: "fraud-detection-prod",
    environment: "production",
    primaryVersion: "v2.1.0 (XGBoost)",
    canaryVersion: "v2.2.0-candidate (LightGBM)",
    strategy: "canary",
    canaryPercentage: 20,
    replicas: 4,
    status: "active",
    isHealthy: true,
  },
  {
    id: "dep-2",
    name: "Customer Churn Risk Scoring API",
    endpointPath: "churn-risk-staging",
    environment: "staging",
    primaryVersion: "v1.0.0 (Random Forest)",
    strategy: "direct",
    canaryPercentage: 0,
    replicas: 2,
    status: "active",
    isHealthy: true,
  },
];

export default function DeploymentsPage() {
  const [deployments, setDeployments] = useState<DeploymentItem[]>(mockDeployments);
  const [canarySliderVal, setCanarySliderVal] = useState(20);
  const [isUpdating, setIsUpdating] = useState(false);

  const handlePromoteCanary = (pct: number) => {
    setCanarySliderVal(pct);
    setIsUpdating(true);
    setTimeout(() => {
      setDeployments((prev) =>
        prev.map((d) => (d.id === "dep-1" ? { ...d, canaryPercentage: pct } : d))
      );
      setIsUpdating(false);
    }, 500);
  };

  const handleRollback = () => {
    alert("Triggered automated rollback to previous healthy version v2.0.0.");
    setCanarySliderVal(0);
    setDeployments((prev) =>
      prev.map((d) => (d.id === "dep-1" ? { ...d, canaryPercentage: 0, strategy: "direct" } : d))
    );
  };

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 pb-2 border-b border-slate-800">
        <div>
          <h1 className="text-2xl font-bold text-slate-100 tracking-tight">Deployments & Traffic Routing</h1>
          <p className="text-xs text-slate-400 mt-1">
            Manage real-time REST serving, Canary rollouts (5% &rarr; 20% &rarr; 50% &rarr; 100%), and automated rollbacks.
          </p>
        </div>
        <div className="flex items-center gap-2.5">
          <Link
            href="/predictions"
            className="flex items-center gap-2 px-3.5 py-2 rounded-lg bg-teal-500 hover:bg-teal-400 text-slate-950 font-bold text-xs shadow-lg shadow-teal-500/20 transition-all"
          >
            <Activity className="w-4 h-4" />
            Test Live Inference
          </Link>
        </div>
      </div>

      {/* Canary Traffic Routing Control Widget */}
      <SectionCard
        title="Canary Rollout & Live Traffic Splitting Controller"
        description="Dynamically adjust traffic split between Production Champion and Candidate Challenger"
        action={
          <button
            onClick={handleRollback}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-rose-500/20 text-rose-300 border border-rose-500/30 text-xs font-semibold hover:bg-rose-500/30 transition-all"
          >
            <RotateCcw className="w-3.5 h-3.5" />
            Emergency Rollback
          </button>
        }
      >
        <div className="space-y-6 bg-slate-950 p-6 rounded-xl border border-slate-800">
          <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
            <div>
              <span className="text-xs text-slate-400 font-semibold block">Primary Champion (90% Base)</span>
              <span className="text-sm font-bold text-slate-100 font-mono">v2.1.0 — XGBoost Model</span>
            </div>
            <div className="text-right">
              <span className="text-xs text-slate-400 font-semibold block">Canary Challenger ({canarySliderVal}%)</span>
              <span className="text-sm font-bold text-teal-400 font-mono">v2.2.0-candidate — LightGBM Model</span>
            </div>
          </div>

          {/* Traffic Percentage Visualizer Bar */}
          <div className="w-full bg-slate-800 h-6 rounded-lg overflow-hidden flex font-mono text-[11px] font-bold">
            <div
              className="bg-indigo-600 h-full flex items-center justify-center text-white transition-all duration-300"
              style={{ width: `${100 - canarySliderVal}%` }}
            >
              {100 - canarySliderVal > 15 && `Champion: ${100 - canarySliderVal}%`}
            </div>
            <div
              className="bg-teal-500 h-full flex items-center justify-center text-slate-950 transition-all duration-300"
              style={{ width: `${canarySliderVal}%` }}
            >
              {canarySliderVal > 10 && `Canary: ${canarySliderVal}%`}
            </div>
          </div>

          {/* Stage Progression Buttons */}
          <div className="flex flex-wrap items-center gap-2 pt-2">
            <span className="text-xs text-slate-400 font-medium mr-2">Quick Stage Step:</span>
            {[5, 20, 50, 100].map((step) => (
              <button
                key={step}
                onClick={() => handlePromoteCanary(step)}
                className={`px-3 py-1 rounded-lg text-xs font-semibold font-mono border transition-all ${
                  canarySliderVal === step
                    ? "bg-teal-500 text-slate-950 border-teal-400"
                    : "bg-slate-900 text-slate-300 border-slate-700 hover:border-slate-500"
                }`}
              >
                {step}% {step === 100 ? "(Full Promotion)" : ""}
              </button>
            ))}
          </div>
        </div>
      </SectionCard>

      {/* Deployments Table */}
      <SectionCard title="Active Model Deployment Endpoints" description="Kubernetes pods, latency health checks, and endpoints">
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="bg-slate-950/60 text-slate-400 uppercase tracking-wider font-semibold border-b border-slate-800">
              <tr>
                <th className="px-4 py-3">Endpoint</th>
                <th className="px-4 py-3">Environment</th>
                <th className="px-4 py-3">Model Version</th>
                <th className="px-4 py-3">Strategy</th>
                <th className="px-4 py-3">Pod Replicas</th>
                <th className="px-4 py-3">Health Status</th>
                <th className="px-4 py-3 text-right">Inference Tester</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60">
              {deployments.map((d) => (
                <tr key={d.id} className="hover:bg-slate-800/40 transition-colors">
                  <td className="px-4 py-3.5">
                    <div className="font-semibold text-slate-100">{d.name}</div>
                    <div className="text-[11px] font-mono text-slate-500">/api/v1/predictions/{d.endpointPath}</div>
                  </td>
                  <td className="px-4 py-3.5">
                    <StageBadge stage={d.environment} />
                  </td>
                  <td className="px-4 py-3.5 font-mono text-slate-300 font-semibold">{d.primaryVersion}</td>
                  <td className="px-4 py-3.5">
                    <span className="font-semibold uppercase tracking-wider text-[10px] px-2 py-0.5 rounded bg-slate-800 text-teal-300 border border-slate-700">
                      {d.strategy}
                    </span>
                  </td>
                  <td className="px-4 py-3.5 font-mono text-slate-300">{d.replicas} Pods</td>
                  <td className="px-4 py-3.5">
                    <StatusBadge status={d.status} />
                  </td>
                  <td className="px-4 py-3.5 text-right">
                    <Link
                      href="/predictions"
                      className="px-2.5 py-1 rounded bg-slate-800 hover:bg-slate-700 text-teal-300 border border-slate-700 text-xs font-semibold inline-block"
                    >
                      Run Predictions &rarr;
                    </Link>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>
    </div>
  );
}
