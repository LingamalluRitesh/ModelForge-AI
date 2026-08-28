"use client";

import React, { useState } from "react";
import Link from "next/link";
import { FlaskConical, GitCompare, ArrowRight, Play, CheckCircle2, Trophy } from "lucide-react";
import { SectionCard } from "@/components/ui/Cards";
import { StatusBadge } from "@/components/ui/Badges";

interface RunItem {
  id: string;
  name: string;
  algorithm: string;
  accuracy: number;
  f1: number;
  precision: number;
  recall: number;
  auc: number;
  duration: string;
  status: string;
  isChampion?: boolean;
}

const mockRuns: RunItem[] = [
  { id: "run-xgb-01", name: "xgboost-tpe-tuned-v2", algorithm: "XGBoost", accuracy: 0.942, f1: 0.946, precision: 0.938, recall: 0.954, auc: 0.984, duration: "32.4s", status: "completed", isChampion: true },
  { id: "run-lgb-02", name: "lightgbm-default-v1", algorithm: "LightGBM", accuracy: 0.938, f1: 0.941, precision: 0.931, recall: 0.951, auc: 0.981, duration: "24.1s", status: "completed" },
  { id: "run-rf-03", name: "random-forest-150trees", algorithm: "Random Forest", accuracy: 0.925, f1: 0.928, precision: 0.919, recall: 0.937, auc: 0.972, duration: "48.2s", status: "completed" },
  { id: "run-torch-04", name: "pytorch-deep-mlp-res", algorithm: "PyTorch MLP", accuracy: 0.931, f1: 0.934, precision: 0.925, recall: 0.943, auc: 0.978, duration: "84.5s", status: "completed" },
  { id: "run-lr-05", name: "logistic-regression-l2", algorithm: "Logistic Regression", accuracy: 0.884, f1: 0.887, precision: 0.871, recall: 0.903, auc: 0.931, duration: "8.2s", status: "completed" },
];

export default function ExperimentsPage() {
  const [selectedRuns, setSelectedRuns] = useState<string[]>(["run-xgb-01", "run-lgb-02", "run-rf-03"]);

  const toggleSelect = (id: string) => {
    if (selectedRuns.includes(id)) {
      setSelectedRuns(selectedRuns.filter((r) => r !== id));
    } else {
      setSelectedRuns([...selectedRuns, id]);
    }
  };

  const comparedRunsList = mockRuns.filter((r) => selectedRuns.includes(r.id));

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 pb-2 border-b border-slate-800">
        <div>
          <h1 className="text-2xl font-bold text-slate-100 tracking-tight">Experiment Tracking & Multi-Model Matrix</h1>
          <p className="text-xs text-slate-400 mt-1">Track training runs, compare hyperparameters and evaluation metrics side-by-side, and promote champions.</p>
        </div>
        <div className="flex items-center gap-2.5">
          <Link
            href="/training"
            className="flex items-center gap-2 px-3.5 py-2 rounded-lg bg-teal-500 hover:bg-teal-400 text-slate-950 font-bold text-xs shadow-lg shadow-teal-500/20 transition-all"
          >
            <Play className="w-4 h-4 fill-current" />
            New Training Run
          </Link>
        </div>
      </div>

      {/* Side-by-Side Comparison Matrix */}
      {comparedRunsList.length > 0 && (
        <SectionCard
          title="Side-by-Side Model Run Comparison"
          description={`Comparing ${comparedRunsList.length} selected runs across primary metrics`}
        >
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead className="bg-slate-950/60 text-slate-400 uppercase tracking-wider font-semibold border-b border-slate-800">
                <tr>
                  <th className="px-4 py-3">Metric / Property</th>
                  {comparedRunsList.map((r) => (
                    <th key={r.id} className="px-4 py-3 text-slate-200">
                      <div className="flex items-center gap-1.5 font-bold">
                        {r.isChampion && <Trophy className="w-3.5 h-3.5 text-amber-400" />}
                        <span>{r.name}</span>
                      </div>
                      <div className="text-[10px] text-teal-400 uppercase font-mono font-normal">{r.algorithm}</div>
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60 font-mono">
                <tr>
                  <td className="px-4 py-3 font-sans font-semibold text-slate-300">F1 Score (Primary)</td>
                  {comparedRunsList.map((r) => (
                    <td key={r.id} className="px-4 py-3 text-slate-100 font-bold text-sm">
                      <span className={r.isChampion ? "text-emerald-400" : ""}>{(r.f1 * 100).toFixed(2)}%</span>
                    </td>
                  ))}
                </tr>
                <tr>
                  <td className="px-4 py-3 font-sans font-semibold text-slate-300">Accuracy</td>
                  {comparedRunsList.map((r) => (
                    <td key={r.id} className="px-4 py-3 text-slate-200">
                      {(r.accuracy * 100).toFixed(2)}%
                    </td>
                  ))}
                </tr>
                <tr>
                  <td className="px-4 py-3 font-sans font-semibold text-slate-300">Precision</td>
                  {comparedRunsList.map((r) => (
                    <td key={r.id} className="px-4 py-3 text-slate-200">
                      {(r.precision * 100).toFixed(2)}%
                    </td>
                  ))}
                </tr>
                <tr>
                  <td className="px-4 py-3 font-sans font-semibold text-slate-300">Recall</td>
                  {comparedRunsList.map((r) => (
                    <td key={r.id} className="px-4 py-3 text-slate-200">
                      {(r.recall * 100).toFixed(2)}%
                    </td>
                  ))}
                </tr>
                <tr>
                  <td className="px-4 py-3 font-sans font-semibold text-slate-300">ROC-AUC</td>
                  {comparedRunsList.map((r) => (
                    <td key={r.id} className="px-4 py-3 text-slate-200">
                      {r.auc.toFixed(4)}
                    </td>
                  ))}
                </tr>
                <tr>
                  <td className="px-4 py-3 font-sans font-semibold text-slate-300">Training Duration</td>
                  {comparedRunsList.map((r) => (
                    <td key={r.id} className="px-4 py-3 text-slate-400">
                      {r.duration}
                    </td>
                  ))}
                </tr>
              </tbody>
            </table>
          </div>
        </SectionCard>
      )}

      {/* All Runs Table */}
      <SectionCard title="Experiment Runs History" description="Select runs using checkboxes to compare side-by-side above">
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="bg-slate-950/60 text-slate-400 uppercase tracking-wider font-semibold border-b border-slate-800">
              <tr>
                <th className="px-4 py-3 w-10">Select</th>
                <th className="px-4 py-3">Run Name</th>
                <th className="px-4 py-3">Algorithm</th>
                <th className="px-4 py-3">F1 Score</th>
                <th className="px-4 py-3">Accuracy</th>
                <th className="px-4 py-3">ROC-AUC</th>
                <th className="px-4 py-3">Duration</th>
                <th className="px-4 py-3">Status</th>
                <th className="px-4 py-3 text-right">Action</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 font-mono">
              {mockRuns.map((r) => {
                const isChecked = selectedRuns.includes(r.id);
                return (
                  <tr key={r.id} className="hover:bg-slate-800/40 transition-colors">
                    <td className="px-4 py-3">
                      <input
                        type="checkbox"
                        checked={isChecked}
                        onChange={() => toggleSelect(r.id)}
                        className="rounded bg-slate-800 border-slate-700 text-teal-500 focus:ring-teal-500"
                      />
                    </td>
                    <td className="px-4 py-3 text-slate-100 font-semibold flex items-center gap-1.5">
                      {r.isChampion && <Trophy className="w-3.5 h-3.5 text-amber-400 shrink-0 font-sans" />}
                      <span>{r.name}</span>
                    </td>
                    <td className="px-4 py-3 text-teal-400">{r.algorithm}</td>
                    <td className="px-4 py-3 text-slate-100 font-bold">{(r.f1 * 100).toFixed(2)}%</td>
                    <td className="px-4 py-3 text-slate-300">{(r.accuracy * 100).toFixed(2)}%</td>
                    <td className="px-4 py-3 text-slate-300">{r.auc.toFixed(3)}</td>
                    <td className="px-4 py-3 text-slate-400">{r.duration}</td>
                    <td className="px-4 py-3 font-sans">
                      <StatusBadge status={r.status} />
                    </td>
                    <td className="px-4 py-3 text-right font-sans">
                      <Link
                        href="/registry"
                        className="px-2 py-1 rounded bg-slate-800 hover:bg-slate-700 text-teal-300 border border-slate-700 text-[11px] font-semibold"
                      >
                        Register Model &rarr;
                      </Link>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </SectionCard>
    </div>
  );
}
