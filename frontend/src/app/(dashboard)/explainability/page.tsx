"use client";

import React, { useState } from "react";
import { Eye, ShieldCheck, BarChart3, HelpCircle, CheckCircle2, AlertTriangle } from "lucide-react";
import { SectionCard } from "@/components/ui/Cards";
import { SHAPBarPlot } from "@/components/charts/Charts";

const mockSHAPImportances = {
  transaction_amount: 0.35,
  credit_utilization: 0.28,
  num_failed_logins: 0.21,
  account_age_months: 0.14,
  ip_risk_score: 0.08,
  is_international: 0.04,
};

const mockLocalFactors = [
  { name: "credit_utilization (0.88)", impact: "+35.2%", direction: "positive", desc: "Significantly higher than account historical mean (0.24)" },
  { name: "num_failed_logins (3 attempts)", impact: "+28.1%", direction: "positive", desc: "3 consecutive invalid credentials in last 1 hour" },
  { name: "is_international (True)", impact: "+21.4%", direction: "positive", desc: "Foreign IP address mismatch with billing country" },
  { name: "account_age_months (14)", impact: "-10.5%", direction: "negative", desc: "Established account tenure reduces risk" },
];

export default function ExplainabilityPage() {
  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 pb-2 border-b border-slate-800">
        <div>
          <h1 className="text-2xl font-bold text-slate-100 tracking-tight flex items-center gap-2">
            <span>Explainable AI & Algorithmic Fairness</span>
          </h1>
          <p className="text-xs text-slate-400 mt-1">
            Global SHAP feature attribution, local prediction explanations, and demographic parity bias audits.
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Global SHAP Feature Importances */}
        <SectionCard
          title="Global SHAP Feature Importance Attribution"
          description="Mean absolute Shapley impact across entire test validation population"
        >
          <SHAPBarPlot importances={mockSHAPImportances} />
        </SectionCard>

        {/* Local Prediction Explanation Breakdown */}
        <SectionCard
          title="Local Prediction Attribution Breakdown"
          description="Why did the model classify the latest sample as High Risk (Fraud)?"
        >
          <div className="space-y-3">
            {mockLocalFactors.map((f) => (
              <div
                key={f.name}
                className="p-3 rounded-xl bg-slate-950 border border-slate-800 flex items-center justify-between"
              >
                <div>
                  <span className="font-semibold text-xs text-slate-200 block">{f.name}</span>
                  <span className="text-[11px] text-slate-400">{f.desc}</span>
                </div>
                <span
                  className={`text-xs font-bold font-mono px-2 py-1 rounded ${
                    f.direction === "positive"
                      ? "bg-rose-500/20 text-rose-300 border border-rose-500/30"
                      : "bg-emerald-500/20 text-emerald-300 border border-emerald-500/30"
                  }`}
                >
                  {f.impact}
                </span>
              </div>
            ))}
          </div>
        </SectionCard>
      </div>

      {/* Algorithmic Fairness & Bias Audit Card */}
      <SectionCard
        title="Model Algorithmic Fairness & Bias Assessment"
        description="Evaluating demographic parity ratio, disparate impact (4/5ths rule), and equalized odds"
      >
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
          <div className="p-4 rounded-xl bg-slate-950 border border-slate-800">
            <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider block">Disparate Impact Ratio</span>
            <span className="text-xl font-bold text-emerald-400 font-mono">0.94</span>
            <span className="text-[11px] text-slate-400 block mt-0.5">&ge; 0.80 (4/5ths Rule Passed)</span>
          </div>
          <div className="p-4 rounded-xl bg-slate-950 border border-slate-800">
            <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider block">Equal Opportunity Difference</span>
            <span className="text-xl font-bold text-emerald-400 font-mono">0.03</span>
            <span className="text-[11px] text-slate-400 block mt-0.5">&le; 0.10 (Passed)</span>
          </div>
          <div className="p-4 rounded-xl bg-slate-950 border border-slate-800">
            <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider block">Demographic Parity</span>
            <span className="text-xl font-bold text-emerald-400 font-mono">0.92</span>
            <span className="text-[11px] text-slate-400 block mt-0.5">Balanced Across Subgroups</span>
          </div>
        </div>

        <div className="p-4 rounded-xl bg-emerald-500/10 border border-emerald-500/20 text-emerald-300 text-xs flex items-center gap-3">
          <CheckCircle2 className="w-5 h-5 shrink-0" />
          <span>Model satisfies enterprise algorithmic fairness and ethical AI standards for all protected sensitive attributes.</span>
        </div>
      </SectionCard>
    </div>
  );
}
