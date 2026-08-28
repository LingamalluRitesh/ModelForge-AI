"use client";

import React from "react";
import {
  ResponsiveContainer,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  Tooltip,
} from "recharts";

interface QualityDimension {
  dimension: string;
  score: number;
  threshold: number;
}

const defaultDimensions: QualityDimension[] = [
  { dimension: "Completeness", score: 98.5, threshold: 90 },
  { dimension: "Uniqueness", score: 99.2, threshold: 95 },
  { dimension: "Consistency", score: 94.0, threshold: 85 },
  { dimension: "Validity", score: 96.8, threshold: 90 },
  { dimension: "Freshness", score: 91.5, threshold: 80 },
  { dimension: "Integrity", score: 95.0, threshold: 90 },
];

export function DataQualityRadarChart({ data = defaultDimensions }: { data?: QualityDimension[] }) {
  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 space-y-3">
      <div className="flex items-center justify-between pb-2 border-b border-slate-800">
        <div>
          <h4 className="font-semibold text-xs text-slate-200">Data Quality Dimensions Radar</h4>
          <p className="text-[11px] text-slate-400">Multi-dimensional dataset integrity and hygiene</p>
        </div>
        <span className="text-xs font-bold font-mono text-emerald-400">Overall: 95.8%</span>
      </div>

      <div style={{ width: "100%", height: 260 }}>
        <ResponsiveContainer width="100%" height="100%">
          <RadarChart data={data} margin={{ top: 10, right: 20, left: 20, bottom: 10 }}>
            <PolarGrid stroke="#1e293b" />
            <PolarAngleAxis dataKey="dimension" stroke="#94a3b8" fontSize={11} />
            <PolarRadiusAxis domain={[0, 100]} stroke="#475569" fontSize={9} />
            <Tooltip
              contentStyle={{
                backgroundColor: "#0f172a",
                borderColor: "#334155",
                borderRadius: "8px",
                fontSize: "12px",
              }}
              formatter={(v: any) => [`${v}%`, "Quality Score"]}
            />
            <Radar
              name="Dataset Quality"
              dataKey="score"
              stroke="#14b8a6"
              fill="#14b8a6"
              fillOpacity={0.4}
            />
          </RadarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
