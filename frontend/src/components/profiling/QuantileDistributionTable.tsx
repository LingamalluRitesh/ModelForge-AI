"use client";

import React from "react";

interface ColumnStat {
  name: string;
  type: string;
  mean: number;
  std: number;
  min: number;
  p25: number;
  median: number;
  p75: number;
  max: number;
  skew: number;
}

const mockStats: ColumnStat[] = [
  { name: "amount", type: "float64", mean: 78.4, std: 142.1, min: 2.5, p25: 18.2, median: 45.0, p75: 98.4, max: 2450.0, skew: 3.84 },
  { name: "customer_age", type: "int64", mean: 41.2, std: 14.8, min: 18.0, p25: 29.0, median: 40.0, p75: 52.0, max: 80.0, skew: 0.22 },
  { name: "credit_utilization", type: "float64", mean: 0.38, std: 0.24, min: 0.01, p25: 0.18, median: 0.34, p75: 0.54, max: 0.99, skew: 0.58 },
  { name: "device_risk_score", type: "float64", mean: 0.21, std: 0.15, min: 0.00, p25: 0.10, median: 0.19, p75: 0.30, max: 0.95, skew: 1.42 },
];

export function QuantileDistributionTable({ stats = mockStats }: { stats?: ColumnStat[] }) {
  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 space-y-3 overflow-hidden">
      <div className="flex items-center justify-between pb-2 border-b border-slate-800">
        <h3 className="font-semibold text-sm text-slate-100">Statistical Quantiles & Moments Summary</h3>
        <span className="text-xs text-slate-500 font-mono">4 Continuous Features Profiled</span>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-left text-xs font-mono">
          <thead className="bg-slate-950/60 text-slate-400 uppercase tracking-wider font-semibold border-b border-slate-800 font-sans">
            <tr>
              <th className="px-3 py-2.5">Feature</th>
              <th className="px-3 py-2.5">Mean &plusmn; Std</th>
              <th className="px-3 py-2.5">Min</th>
              <th className="px-3 py-2.5">p25</th>
              <th className="px-3 py-2.5">Median</th>
              <th className="px-3 py-2.5">p75</th>
              <th className="px-3 py-2.5">Max</th>
              <th className="px-3 py-2.5">Skewness</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800/60">
            {stats.map((s) => (
              <tr key={s.name} className="hover:bg-slate-800/40 transition-colors">
                <td className="px-3 py-3 text-slate-200 font-bold font-sans">{s.name}</td>
                <td className="px-3 py-3 text-teal-400">{s.mean.toFixed(2)} &plusmn; {s.std.toFixed(2)}</td>
                <td className="px-3 py-3 text-slate-400">{s.min.toFixed(2)}</td>
                <td className="px-3 py-3 text-slate-300">{s.p25.toFixed(2)}</td>
                <td className="px-3 py-3 text-slate-100 font-bold">{s.median.toFixed(2)}</td>
                <td className="px-3 py-3 text-slate-300">{s.p75.toFixed(2)}</td>
                <td className="px-3 py-3 text-slate-400">{s.max.toFixed(2)}</td>
                <td className="px-3 py-3 text-amber-400">{s.skew.toFixed(2)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
