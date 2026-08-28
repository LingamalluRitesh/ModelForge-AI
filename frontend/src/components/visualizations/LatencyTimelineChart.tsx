"use client";

import React from "react";
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from "recharts";

interface LatencyTimelineProps {
  data?: Array<{ timestamp: string; p50: number; p95: number; p99: number }>;
}

const defaultTimeline = Array.from({ length: 24 }).map((_, i) => ({
  timestamp: `${String(i).padStart(2, "0")}:00`,
  p50: Math.round(12 + Math.random() * 4),
  p95: Math.round(28 + Math.random() * 8),
  p99: Math.round(45 + Math.random() * 15),
}));

export function LatencyTimelineChart({ data = defaultTimeline }: LatencyTimelineProps) {
  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 space-y-3">
      <div className="flex items-center justify-between pb-2 border-b border-slate-800">
        <div>
          <h4 className="font-semibold text-xs text-slate-200">24-Hour Latency Distribution (ms)</h4>
          <p className="text-[11px] text-slate-400">p50, p95, and p99 real-time endpoint latency percentiles</p>
        </div>
        <div className="flex items-center gap-2">
          <span className="w-2.5 h-2.5 rounded-full bg-emerald-500 animate-pulse" />
          <span className="text-[11px] font-mono text-emerald-400">SLA: 100ms</span>
        </div>
      </div>

      <div style={{ width: "100%", height: 260 }}>
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={data} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
            <defs>
              <linearGradient id="p99Grad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#f43f5e" stopOpacity={0.4} />
                <stop offset="95%" stopColor="#f43f5e" stopOpacity={0.0} />
              </linearGradient>
              <linearGradient id="p95Grad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#f59e0b" stopOpacity={0.4} />
                <stop offset="95%" stopColor="#f59e0b" stopOpacity={0.0} />
              </linearGradient>
              <linearGradient id="p50Grad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#14b8a6" stopOpacity={0.4} />
                <stop offset="95%" stopColor="#14b8a6" stopOpacity={0.0} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
            <XAxis dataKey="timestamp" stroke="#64748b" fontSize={10} />
            <YAxis stroke="#64748b" fontSize={10} unit="ms" />
            <Tooltip
              contentStyle={{
                backgroundColor: "#0f172a",
                borderColor: "#334155",
                borderRadius: "8px",
                fontSize: "12px",
              }}
            />
            <Legend wrapperStyle={{ fontSize: "11px", color: "#94a3b8" }} />
            <Area type="monotone" dataKey="p99" name="p99 Latency" stroke="#f43f5e" fill="url(#p99Grad)" strokeWidth={2} />
            <Area type="monotone" dataKey="p95" name="p95 Latency" stroke="#f59e0b" fill="url(#p95Grad)" strokeWidth={2} />
            <Area type="monotone" dataKey="p50" name="p50 Latency" stroke="#14b8a6" fill="url(#p50Grad)" strokeWidth={2} />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
