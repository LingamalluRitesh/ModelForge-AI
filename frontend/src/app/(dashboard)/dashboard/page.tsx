"use client";

import React, { useState, useEffect } from "react";
import Link from "next/link";
import {
  Rocket,
  Activity,
  TrendingDown,
  Cpu,
  FlaskConical,
  Database,
  ArrowUpRight,
  ShieldAlert,
  Play,
  Sparkles,
  Layers,
  CheckCircle2,
} from "lucide-react";
import { StatCard, SectionCard } from "@/components/ui/Cards";
import { StatusBadge, StageBadge, SeverityBadge } from "@/components/ui/Badges";
import { MetricLineChart, QualityScoreGauge } from "@/components/charts/Charts";
import { useProjectStore } from "@/stores/authStore";
import { apiClient } from "@/lib/api";

const mockThroughputData = [
  { timestamp: "00:00", requests: 1240, latency: 14.2, errors: 2 },
  { timestamp: "04:00", requests: 890, latency: 13.8, errors: 1 },
  { timestamp: "08:00", requests: 3420, latency: 18.5, errors: 6 },
  { timestamp: "12:00", requests: 4890, latency: 22.1, errors: 12 },
  { timestamp: "16:00", requests: 5120, latency: 24.3, errors: 8 },
  { timestamp: "20:00", requests: 3890, latency: 17.6, errors: 4 },
  { timestamp: "23:59", requests: 2150, latency: 15.0, errors: 3 },
];

export default function DashboardPage() {
  const { currentProject } = useProjectStore();
  const [deployments, setDeployments] = useState<any[]>([
    {
      id: "dep-fraud-prod",
      name: "Fraud Detection Real-Time API",
      endpoint_path: "fraud-detection-prod",
      environment: "production",
      status: "active",
      strategy: "canary",
      primary_traffic_percentage: 90,
      canary_stage_percentage: 10,
      min_replicas: 3,
      current_replicas: 4,
      is_healthy: true,
      error_rate_threshold: 0.05,
    },
    {
      id: "dep-churn-staging",
      name: "Customer Churn Batch Scorer",
      endpoint_path: "churn-risk-staging",
      environment: "staging",
      status: "active",
      strategy: "direct",
      primary_traffic_percentage: 100,
      canary_stage_percentage: 0,
      min_replicas: 1,
      current_replicas: 1,
      is_healthy: true,
      error_rate_threshold: 0.05,
    },
  ]);

  return (
    <div className="space-y-8">
      {/* Header Banner */}
      <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4 pb-2 border-b border-slate-800">
        <div>
          <h1 className="text-2xl font-bold text-slate-100 tracking-tight flex items-center gap-2.5">
            <span>Executive ML Operations Overview</span>
            <span className="text-xs px-2.5 py-0.5 rounded-full bg-emerald-500/10 text-emerald-300 border border-emerald-500/20 font-mono">
              Live Production
            </span>
          </h1>
          <p className="text-xs text-slate-400 mt-1">
            Monitoring health, throughput, data drift, and model lifecycle for{" "}
            <span className="text-teal-400 font-semibold">{currentProject?.name || "ModelForge Workspace"}</span>
          </p>
        </div>

        {/* Quick Actions */}
        <div className="flex items-center gap-2.5">
          <Link
            href="/training"
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-teal-500 hover:bg-teal-400 text-slate-950 font-bold text-xs shadow-lg shadow-teal-500/20 transition-all"
          >
            <Cpu className="w-3.5 h-3.5" />
            Train Model
          </Link>
          <Link
            href="/automl"
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white font-semibold text-xs transition-all shadow-lg shadow-indigo-600/20"
          >
            <Sparkles className="w-3.5 h-3.5" />
            AutoML Run
          </Link>
          <Link
            href="/datasets"
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs border border-slate-700 transition-all"
          >
            <Database className="w-3.5 h-3.5" />
            Ingest Dataset
          </Link>
        </div>
      </div>

      {/* KPI Stat Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          title="Production Throughput"
          value="21,550 req"
          change="+18.4%"
          isPositive={true}
          icon={Activity}
          iconColor="text-teal-400"
          subtext="Last 24h &bull; Avg Latency 18.2ms"
        />
        <StatCard
          title="Active Deployments"
          value="4 Endpoints"
          change="100% Healthy"
          isPositive={true}
          icon={Rocket}
          iconColor="text-purple-400"
          subtext="1 Canary (10%) &bull; 3 Direct"
        />
        <StatCard
          title="Data Drift Status"
          value="PSI: 0.08"
          change="Stable (Low)"
          isPositive={true}
          icon={TrendingDown}
          iconColor="text-emerald-400"
          subtext="KS p-value 0.28 &bull; 0 Features Drifted"
        />
        <StatCard
          title="Registered Models"
          value="12 Versions"
          change="v2.1.0 Active"
          isPositive={true}
          icon={FlaskConical}
          iconColor="text-blue-400"
          subtext="F1 Score: 94.6% &bull; AUC 0.98"
        />
      </div>

      {/* Primary Graphs & Real-time Metrics */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Real-time Inference Throughput & Latency */}
        <div className="lg:col-span-2">
          <SectionCard
            title="Real-Time Prediction Volume & Latency Trend"
            description="Hourly requests per second, p95 response time (ms), and error rates"
            action={
              <Link href="/monitoring" className="text-xs text-teal-400 hover:underline flex items-center gap-1">
                Full Metrics <ArrowUpRight className="w-3.5 h-3.5" />
              </Link>
            }
          >
            <MetricLineChart
              data={mockThroughputData}
              lines={[
                { key: "requests", color: "#14b8a6", name: "Inference Volume" },
                { key: "latency", color: "#818cf8", name: "Latency (ms)" },
              ]}
            />
          </SectionCard>
        </div>

        {/* Data Quality & Governance Health */}
        <div>
          <SectionCard
            title="Production Data Quality Gate"
            description="Composite validation score across 15 automated checks"
            action={
              <Link href="/datasets" className="text-xs text-teal-400 hover:underline flex items-center gap-1">
                Quality Rules <ArrowUpRight className="w-3.5 h-3.5" />
              </Link>
            }
          >
            <QualityScoreGauge score={94.6} />
            <div className="mt-4 space-y-2 text-xs border-t border-slate-800 pt-4">
              <div className="flex justify-between text-slate-300">
                <span>Missing Value Rate</span>
                <span className="font-semibold text-emerald-400">0.02% (Passed)</span>
              </div>
              <div className="flex justify-between text-slate-300">
                <span>Duplicate Rows</span>
                <span className="font-semibold text-emerald-400">0 (Passed)</span>
              </div>
              <div className="flex justify-between text-slate-300">
                <span>Distribution Drift</span>
                <span className="font-semibold text-emerald-400">PSI &lt; 0.10 (Passed)</span>
              </div>
            </div>
          </SectionCard>
        </div>
      </div>

      {/* Active Deployments Table */}
      <SectionCard
        title="Deployed Model Endpoints & Traffic Allocation"
        description="Active inference endpoints with Canary stages, scaling replicas, and health probes"
        action={
          <Link href="/deployments" className="text-xs text-teal-400 hover:underline flex items-center gap-1">
            Manage Deployments <ArrowUpRight className="w-3.5 h-3.5" />
          </Link>
        }
      >
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="bg-slate-950/60 text-slate-400 uppercase tracking-wider font-semibold border-b border-slate-800">
              <tr>
                <th className="px-4 py-3">Endpoint Name</th>
                <th className="px-4 py-3">Environment</th>
                <th className="px-4 py-3">Strategy & Traffic</th>
                <th className="px-4 py-3">Replicas</th>
                <th className="px-4 py-3">Status</th>
                <th className="px-4 py-3 text-right">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60">
              {deployments.map((d) => (
                <tr key={d.id} className="hover:bg-slate-800/40 transition-colors">
                  <td className="px-4 py-3.5">
                    <div className="font-semibold text-slate-100">{d.name}</div>
                    <div className="text-[11px] font-mono text-slate-500">/api/v1/predictions/{d.endpoint_path}</div>
                  </td>
                  <td className="px-4 py-3.5">
                    <StageBadge stage={d.environment} />
                  </td>
                  <td className="px-4 py-3.5">
                    <div className="flex items-center gap-2">
                      <span className="font-semibold uppercase tracking-wider text-[10px] text-teal-300">
                        {d.strategy}
                      </span>
                      {d.strategy === "canary" && (
                        <span className="text-[10px] px-1.5 py-0.5 rounded bg-purple-500/20 text-purple-300 border border-purple-500/30">
                          {d.canary_stage_percentage}% Canary
                        </span>
                      )}
                    </div>
                  </td>
                  <td className="px-4 py-3.5 text-slate-300 font-mono">
                    {d.current_replicas} / {d.min_replicas}-{d.min_replicas + 4} Pods
                  </td>
                  <td className="px-4 py-3.5">
                    <StatusBadge status={d.status} />
                  </td>
                  <td className="px-4 py-3.5 text-right">
                    <Link
                      href="/predictions"
                      className="px-2.5 py-1 rounded bg-slate-800 hover:bg-slate-700 text-teal-300 border border-slate-700 text-xs font-semibold inline-block"
                    >
                      Test Inference &rarr;
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
