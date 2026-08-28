"use client";

import React from "react";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from "recharts";

const rocPoints = [
  { fpr: 0.0, tpr: 0.0, baseline: 0.0 },
  { fpr: 0.02, tpr: 0.45, baseline: 0.02 },
  { fpr: 0.05, tpr: 0.78, baseline: 0.05 },
  { fpr: 0.1, tpr: 0.89, baseline: 0.1 },
  { fpr: 0.2, tpr: 0.95, baseline: 0.2 },
  { fpr: 0.4, tpr: 0.98, baseline: 0.4 },
  { fpr: 0.7, tpr: 0.99, baseline: 0.7 },
  { fpr: 1.0, tpr: 1.0, baseline: 1.0 },
];

export function ROCPlotCanvas({ auc = 0.985 }: { auc?: number }) {
  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between text-xs pb-1">
        <span className="text-slate-400 font-semibold">Receiver Operating Characteristic (ROC)</span>
        <span className="text-teal-400 font-bold font-mono">AUC = {auc.toFixed(3)}</span>
      </div>
      <div style={{ width: "100%", height: 260 }}>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={rocPoints} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
            <XAxis
              dataKey="fpr"
              type="number"
              domain={[0, 1]}
              stroke="#64748b"
              fontSize={11}
              unit=""
              tickFormatter={(v) => v.toFixed(1)}
            />
            <YAxis
              type="number"
              domain={[0, 1]}
              stroke="#64748b"
              fontSize={11}
              tickFormatter={(v) => v.toFixed(1)}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: "#0f172a",
                borderColor: "#334155",
                borderRadius: "8px",
                fontSize: "12px",
              }}
              formatter={(val: any) => [Number(val).toFixed(2), "Rate"]}
            />
            <Legend wrapperStyle={{ fontSize: "11px", color: "#94a3b8" }} />
            <Line
              type="monotone"
              dataKey="tpr"
              name="Model Classifier (TPR vs FPR)"
              stroke="#14b8a6"
              strokeWidth={2.5}
              dot={{ r: 3, fill: "#14b8a6" }}
            />
            <Line
              type="monotone"
              dataKey="baseline"
              name="Random Chance"
              stroke="#475569"
              strokeDasharray="4 4"
              strokeWidth={1.5}
              dot={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
