"use client";

import React from "react";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  AreaChart,
  Area,
} from "recharts";

export function MetricLineChart({
  data,
  lines,
  xAxisKey = "timestamp",
  height = 260,
}: {
  data: any[];
  lines: { key: string; color: string; name?: string }[];
  xAxisKey?: string;
  height?: number;
}) {
  return (
    <div style={{ width: "100%", height }}>
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={data} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
          <defs>
            {lines.map((l) => (
              <linearGradient key={l.key} id={`gradient-${l.key}`} x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor={l.color} stopOpacity={0.3} />
                <stop offset="95%" stopColor={l.color} stopOpacity={0.0} />
              </linearGradient>
            ))}
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
          <XAxis dataKey={xAxisKey} stroke="#64748b" fontSize={11} />
          <YAxis stroke="#64748b" fontSize={11} />
          <Tooltip
            contentStyle={{
              backgroundColor: "#0f172a",
              borderColor: "#334155",
              borderRadius: "8px",
              color: "#f8fafc",
              fontSize: "12px",
            }}
          />
          <Legend wrapperStyle={{ fontSize: "11px", color: "#94a3b8" }} />
          {lines.map((l) => (
            <Area
              key={l.key}
              type="monotone"
              dataKey={l.key}
              name={l.name || l.key}
              stroke={l.color}
              strokeWidth={2}
              fillOpacity={1}
              fill={`url(#gradient-${l.key})`}
            />
          ))}
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}

export function ConfusionMatrixView({
  matrix,
  labels,
}: {
  matrix: number[][];
  labels: string[];
}) {
  if (!matrix || matrix.length === 0) return <div className="text-xs text-slate-500">No matrix data</div>;

  const maxVal = Math.max(...matrix.flat());

  return (
    <div className="flex flex-col items-center">
      <div className="text-xs font-semibold text-slate-400 mb-2 uppercase tracking-wider">
        Predicted Class &rarr;
      </div>
      <div className="flex items-center">
        <div className="text-xs font-semibold text-slate-400 -rotate-90 mr-2 uppercase tracking-wider">
          Actual Class &rarr;
        </div>
        <div className="grid gap-1.5" style={{ gridTemplateColumns: `repeat(${labels.length}, minmax(60px, 1fr))` }}>
          {matrix.map((row, i) =>
            row.map((val, j) => {
              const intensity = maxVal > 0 ? val / maxVal : 0;
              const isDiagonal = i === j;
              return (
                <div
                  key={`${i}-${j}`}
                  className="h-14 rounded-lg flex flex-col items-center justify-center font-bold text-xs transition-all border border-slate-700/50"
                  style={{
                    backgroundColor: isDiagonal
                      ? `rgba(20, 184, 166, ${Math.max(0.2, intensity)})`
                      : `rgba(244, 63, 94, ${Math.max(0.1, intensity * 0.7)})`,
                    color: isDiagonal ? "#ccfbf1" : "#fecdd3",
                  }}
                  title={`Actual: ${labels[i]}, Predicted: ${labels[j]}, Count: ${val}`}
                >
                  <span>{val}</span>
                  <span className="text-[9px] font-normal opacity-70">
                    {labels[i]}&rarr;{labels[j]}
                  </span>
                </div>
              );
            })
          )}
        </div>
      </div>
    </div>
  );
}

export function SHAPBarPlot({
  importances,
}: {
  importances: Record<string, number>;
}) {
  const data = Object.entries(importances)
    .map(([feature, val]) => ({
      feature: feature.length > 18 ? feature.substring(0, 16) + "..." : feature,
      importance: Number((val * 100).toFixed(1)),
    }))
    .slice(0, 8);

  return (
    <div style={{ width: "100%", height: 240 }}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} layout="vertical" margin={{ top: 5, right: 20, left: 40, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" horizontal={false} />
          <XAxis type="number" stroke="#64748b" fontSize={11} unit="%" />
          <YAxis dataKey="feature" type="category" stroke="#94a3b8" fontSize={11} width={80} />
          <Tooltip
            contentStyle={{ backgroundColor: "#0f172a", borderColor: "#334155", borderRadius: "8px", fontSize: "12px" }}
            formatter={(value: any) => [`${value}%`, "Attribution Impact"]}
          />
          <Bar dataKey="importance" fill="#14b8a6" radius={[0, 4, 4, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

export function QualityScoreGauge({ score }: { score: number }) {
  const color = score >= 90 ? "#10b981" : score >= 75 ? "#f59e0b" : "#ef4444";
  return (
    <div className="flex flex-col items-center justify-center p-4">
      <div className="relative w-28 h-28 flex items-center justify-center">
        <svg className="w-full h-full transform -rotate-90">
          <circle cx="56" cy="56" r="46" stroke="#1e293b" strokeWidth="8" fill="transparent" />
          <circle
            cx="56"
            cy="56"
            r="46"
            stroke={color}
            strokeWidth="8"
            strokeDasharray={289}
            strokeDashoffset={289 - (289 * score) / 100}
            strokeLinecap="round"
            fill="transparent"
            className="transition-all duration-1000 ease-out"
          />
        </svg>
        <div className="absolute flex flex-col items-center">
          <span className="text-2xl font-bold text-slate-100">{score.toFixed(1)}%</span>
          <span className="text-[10px] uppercase font-bold text-slate-400">Score</span>
        </div>
      </div>
    </div>
  );
}
