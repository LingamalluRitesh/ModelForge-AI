"use client";

import React from "react";
import { CheckCircle2, XCircle, ShieldCheck, AlertCircle } from "lucide-react";

interface RuleCheck {
  id: string;
  name: string;
  category: "Performance" | "Data Quality" | "Drift" | "Security" | "Fairness";
  threshold: string;
  observed: string;
  passed: boolean;
}

const defaultRules: RuleCheck[] = [
  { id: "r1", name: "F1 Score Minimal Gate", category: "Performance", threshold: "F1 >= 0.85", observed: "0.948", passed: true },
  { id: "r2", name: "Data Quality Score Gate", category: "Data Quality", threshold: "Score >= 90.0%", observed: "94.6%", passed: true },
  { id: "r3", name: "Population Drift PSI", category: "Drift", threshold: "PSI < 0.25", observed: "0.065", passed: true },
  { id: "r4", name: "Disparate Impact Ratio", category: "Fairness", threshold: "DIR >= 0.80", observed: "0.94", passed: true },
  { id: "r5", name: "Inference Latency SLA", category: "Performance", threshold: "p95 <= 50ms", observed: "18.2ms", passed: true },
  { id: "r6", name: "Security & Vulnerability Audit", category: "Security", threshold: "Zero High/Critical CVEs", observed: "Passed (0 CVEs)", passed: true },
];

export function QualityGateChecklist({ rules = defaultRules }: { rules?: RuleCheck[] }) {
  const allPassed = rules.every((r) => r.passed);

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 space-y-4">
      <div className="flex items-center justify-between pb-3 border-b border-slate-800">
        <div className="flex items-center gap-2">
          <ShieldCheck className="w-5 h-5 text-teal-400" />
          <h3 className="font-semibold text-sm text-slate-100">Automated Quality Gate Compliance</h3>
        </div>
        <span
          className={`text-xs font-bold px-2.5 py-0.5 rounded-full border ${
            allPassed
              ? "bg-emerald-500/20 text-emerald-300 border-emerald-500/40"
              : "bg-rose-500/20 text-rose-300 border-rose-500/40"
          }`}
        >
          {allPassed ? "100% GATES PASSED" : "GATE VIOLATIONS DETECTED"}
        </span>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-left text-xs">
          <thead className="bg-slate-950/60 text-slate-400 uppercase tracking-wider font-semibold border-b border-slate-800">
            <tr>
              <th className="px-4 py-2.5">Check Name</th>
              <th className="px-4 py-2.5">Category</th>
              <th className="px-4 py-2.5">Enforced Threshold</th>
              <th className="px-4 py-2.5">Observed Value</th>
              <th className="px-4 py-2.5">Status</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800/60">
            {rules.map((r) => (
              <tr key={r.id} className="hover:bg-slate-800/40 transition-colors">
                <td className="px-4 py-3 font-semibold text-slate-200">{r.name}</td>
                <td className="px-4 py-3">
                  <span className="text-[10px] uppercase font-mono px-2 py-0.5 rounded bg-slate-800 text-slate-300 border border-slate-700">
                    {r.category}
                  </span>
                </td>
                <td className="px-4 py-3 font-mono text-slate-400">{r.threshold}</td>
                <td className="px-4 py-3 font-mono font-bold text-slate-200">{r.observed}</td>
                <td className="px-4 py-3">
                  {r.passed ? (
                    <span className="flex items-center gap-1 text-emerald-400 font-semibold text-[11px]">
                      <CheckCircle2 className="w-3.5 h-3.5" />
                      Passed
                    </span>
                  ) : (
                    <span className="flex items-center gap-1 text-rose-400 font-semibold text-[11px]">
                      <XCircle className="w-3.5 h-3.5" />
                      Failed
                    </span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
