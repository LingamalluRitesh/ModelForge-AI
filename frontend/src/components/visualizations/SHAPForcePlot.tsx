"use client";

import React from "react";
import { ArrowUpRight, ArrowDownRight } from "lucide-react";

interface SHAPFactor {
  feature_name: string;
  feature_value: any;
  shap_value: number;
  percentage_contribution: number;
  direction: "positive" | "negative";
}

interface SHAPForcePlotProps {
  baseValue?: number;
  predictionValue?: number;
  factors?: SHAPFactor[];
}

const defaultFactors: SHAPFactor[] = [
  { feature_name: "device_risk_score", feature_value: 0.89, shap_value: 0.38, percentage_contribution: 45.2, direction: "positive" },
  { feature_name: "amount", feature_value: 450.0, shap_value: 0.24, percentage_contribution: 28.5, direction: "positive" },
  { feature_name: "num_failed_logins", feature_value: 3, shap_value: 0.18, percentage_contribution: 21.4, direction: "positive" },
  { feature_name: "account_age_months", feature_value: 42, shap_value: -0.12, percentage_contribution: 14.2, direction: "negative" },
];

export function SHAPForcePlot({ baseValue = 0.5, predictionValue = 0.94, factors = defaultFactors }: SHAPForcePlotProps) {
  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 space-y-4">
      <div className="flex items-center justify-between pb-2 border-b border-slate-800">
        <div>
          <h4 className="font-semibold text-xs text-slate-200">SHAP Local Instance Attribution</h4>
          <p className="text-[11px] text-slate-400">Force decomposition of individual prediction probability</p>
        </div>
        <div className="flex items-center gap-3 text-xs font-mono">
          <span className="text-slate-500">Base: {baseValue.toFixed(2)}</span>
          <span className="text-slate-600">&rarr;</span>
          <span className="text-teal-400 font-bold">Prediction: {predictionValue.toFixed(2)}</span>
        </div>
      </div>

      <div className="space-y-2.5">
        {factors.map((f) => (
          <div key={f.feature_name} className="flex items-center justify-between p-2.5 rounded-lg bg-slate-950/60 border border-slate-800 text-xs">
            <div className="flex items-center gap-2">
              {f.direction === "positive" ? (
                <div className="p-1 rounded bg-rose-500/10 text-rose-400 border border-rose-500/20">
                  <ArrowUpRight className="w-3.5 h-3.5" />
                </div>
              ) : (
                <div className="p-1 rounded bg-teal-500/10 text-teal-400 border border-teal-500/20">
                  <ArrowDownRight className="w-3.5 h-3.5" />
                </div>
              )}
              <div>
                <span className="font-semibold text-slate-200">{f.feature_name}</span>
                <span className="text-slate-500 text-[11px] ml-2 font-mono">= {String(f.feature_value)}</span>
              </div>
            </div>

            <div className="flex items-center gap-3 font-mono">
              <span className={`font-bold ${f.direction === "positive" ? "text-rose-400" : "text-teal-400"}`}>
                {f.shap_value > 0 ? `+${f.shap_value.toFixed(3)}` : f.shap_value.toFixed(3)}
              </span>
              <span className="text-slate-400 text-[11px] w-12 text-right">
                {f.percentage_contribution.toFixed(1)}%
              </span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
