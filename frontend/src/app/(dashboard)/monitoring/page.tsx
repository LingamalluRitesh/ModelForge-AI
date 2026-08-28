"use client";

import React, { useState } from "react";
import { Activity, Clock, ShieldAlert, Cpu, HardDrive, BarChart2, RefreshCw } from "lucide-react";
import { StatCard, SectionCard } from "@/components/ui/Cards";
import { MetricLineChart } from "@/components/charts/Charts";

const liveTimeseriesData = [
  { timestamp: "10:00", rps: 180, p50: 12.1, p95: 18.4, p99: 26.2, error_rate: 0.01 },
  { timestamp: "10:10", rps: 220, p50: 12.8, p95: 19.1, p99: 28.5, error_rate: 0.02 },
  { timestamp: "10:20", rps: 310, p50: 14.5, p95: 22.0, p99: 32.1, error_rate: 0.01 },
  { timestamp: "10:30", rps: 290, p50: 13.9, p95: 20.8, p99: 30.4, error_rate: 0.00 },
  { timestamp: "10:40", rps: 420, p50: 16.2, p95: 24.5, p99: 35.8, error_rate: 0.03 },
  { timestamp: "10:50", rps: 380, p50: 15.1, p95: 23.2, p99: 33.6, error_rate: 0.01 },
];

export default function LiveMonitoringPage() {
  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 pb-2 border-b border-slate-800">
        <div>
          <h1 className="text-2xl font-bold text-slate-100 tracking-tight flex items-center gap-2">
            <span>Model Observability & Telemetry</span>
            <span className="w-2.5 h-2.5 rounded-full bg-emerald-400 animate-pulse"></span>
          </h1>
          <p className="text-xs text-slate-400 mt-1">
            Real-time Prometheus metrics exporter, OpenTelemetry traces, latency percentiles, and hardware health.
          </p>
        </div>
      </div>

      {/* Observability Stats */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          title="Current Throughput"
          value="380 req/s"
          change="+12% Spike"
          isPositive={true}
          icon={Activity}
          iconColor="text-teal-400"
          subtext="Kubernetes Ingress Gateway"
        />
        <StatCard
          title="p95 Latency"
          value="23.2 ms"
          change="SLA Met (<50ms)"
          isPositive={true}
          icon={Clock}
          iconColor="text-indigo-400"
          subtext="p50: 15.1ms &bull; p99: 33.6ms"
        />
        <StatCard
          title="Error Rate"
          value="0.01%"
          change="Optimal (<0.1%)"
          isPositive={true}
          icon={ShieldAlert}
          iconColor="text-emerald-400"
          subtext="4xx/5xx HTTP Error Ratio"
        />
        <StatCard
          title="Cluster Pod CPU"
          value="28.4%"
          change="4 of 8 Pods"
          isPositive={true}
          icon={Cpu}
          iconColor="text-cyan-400"
          subtext="Memory Util: 2.1 GB / 8 GB"
        />
      </div>

      {/* Latency Percentiles & Throughput Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <SectionCard title="Latency Percentiles (p50, p95, p99 ms)" description="Response time distribution over time">
          <MetricLineChart
            data={liveTimeseriesData}
            lines={[
              { key: "p50", color: "#14b8a6", name: "p50 Median" },
              { key: "p95", color: "#f59e0b", name: "p95 Percentile" },
              { key: "p99", color: "#ef4444", name: "p99 Tail Latency" },
            ]}
          />
        </SectionCard>

        <SectionCard title="Requests Per Second (RPS Throughput)" description="Real-time traffic load across active inference pods">
          <MetricLineChart
            data={liveTimeseriesData}
            lines={[{ key: "rps", color: "#818cf8", name: "RPS Throughput" }]}
          />
        </SectionCard>
      </div>
    </div>
  );
}
