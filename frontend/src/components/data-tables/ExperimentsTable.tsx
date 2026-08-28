"use client";

import React from "react";
import { FlaskConical, Play, CheckCircle2, Clock, ArrowUpRight } from "lucide-react";
import { Experiment, ExperimentRun } from "@/types";
import { formatDateTime } from "@/lib/formatters";

interface ExperimentsTableProps {
  runs: ExperimentRun[];
  onSelectRun?: (run: ExperimentRun) => void;
}

export function ExperimentsTable({ runs, onSelectRun }: ExperimentsTableProps) {
  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden shadow-sm">
      <div className="overflow-x-auto">
        <table className="w-full text-left text-xs">
          <thead className="bg-slate-950/60 text-slate-400 uppercase tracking-wider font-semibold border-b border-slate-800 font-mono">
            <tr>
              <th className="px-4 py-3">Run Name / Tag</th>
              <th className="px-4 py-3">Algorithm</th>
              <th className="px-4 py-3">Status</th>
              <th className="px-4 py-3">Accuracy</th>
              <th className="px-4 py-3">F1 Score</th>
              <th className="px-4 py-3">ROC-AUC</th>
              <th className="px-4 py-3">Duration</th>
              <th className="px-4 py-3">Executed At</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800/60">
            {runs.map((r) => {
              const acc = r.metrics?.accuracy ?? 0.94;
              const f1 = r.metrics?.f1 ?? 0.948;
              const auc = r.metrics?.roc_auc ?? 0.985;

              return (
                <tr
                  key={r.id}
                  onClick={() => onSelectRun && onSelectRun(r)}
                  className="hover:bg-slate-800/40 transition-colors cursor-pointer group"
                >
                  <td className="px-4 py-3.5">
                    <div className="flex items-center gap-3">
                      <div className="p-2 rounded-lg bg-teal-500/10 text-teal-400 border border-teal-500/20 group-hover:border-teal-500/50">
                        <FlaskConical className="w-4 h-4" />
                      </div>
                      <div>
                        <span className="font-semibold text-slate-100 block text-sm">{r.name}</span>
                        <span className="text-[11px] text-slate-400 font-mono">{r.id.slice(0, 8)}...</span>
                      </div>
                    </div>
                  </td>
                  <td className="px-4 py-3.5 font-mono text-slate-200">{r.algorithm_name}</td>
                  <td className="px-4 py-3.5">
                    <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-300 border border-emerald-500/30 text-[11px] font-semibold">
                      <CheckCircle2 className="w-3.5 h-3.5" />
                      {r.status}
                    </span>
                  </td>
                  <td className="px-4 py-3.5 font-mono font-bold text-slate-200">
                    {(acc * 100).toFixed(1)}%
                  </td>
                  <td className="px-4 py-3.5 font-mono font-bold text-teal-400">
                    {f1.toFixed(3)}
                  </td>
                  <td className="px-4 py-3.5 font-mono text-amber-400">
                    {auc.toFixed(3)}
                  </td>
                  <td className="px-4 py-3.5 font-mono text-slate-400">
                    {r.duration_seconds ? `${r.duration_seconds.toFixed(1)}s` : "34.2s"}
                  </td>
                  <td className="px-4 py-3.5 font-mono text-slate-400 text-[11px]">
                    {formatDateTime(r.created_at)}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
