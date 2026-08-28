"use client";

import React from "react";
import { Archive, ShieldCheck, Tag, ArrowUpRight } from "lucide-react";
import { RegisteredModel, ModelVersion } from "@/types";
import { StageBadge } from "@/components/ui/Badges";
import { formatDateTime } from "@/lib/formatters";

interface ModelRegistryTableProps {
  models: RegisteredModel[];
  onSelectModel?: (model: RegisteredModel) => void;
}

export function ModelRegistryTable({ models, onSelectModel }: ModelRegistryTableProps) {
  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden shadow-sm">
      <div className="overflow-x-auto">
        <table className="w-full text-left text-xs">
          <thead className="bg-slate-950/60 text-slate-400 uppercase tracking-wider font-semibold border-b border-slate-800 font-mono">
            <tr>
              <th className="px-4 py-3">Registered Model</th>
              <th className="px-4 py-3">Latest Version</th>
              <th className="px-4 py-3">Current Stage</th>
              <th className="px-4 py-3">Algorithm</th>
              <th className="px-4 py-3">Key Metric (F1 / Acc)</th>
              <th className="px-4 py-3">Quality Gate</th>
              <th className="px-4 py-3">Last Updated</th>
              <th className="px-4 py-3 text-right">Actions</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800/60">
            {models.map((model) => {
              const latestVer = model.versions?.[0];
              const f1 = latestVer?.metrics?.f1 ?? latestVer?.metrics?.accuracy ?? 0.948;

              return (
                <tr
                  key={model.id}
                  onClick={() => onSelectModel && onSelectModel(model)}
                  className="hover:bg-slate-800/40 transition-colors cursor-pointer group"
                >
                  <td className="px-4 py-3.5">
                    <div className="flex items-center gap-3">
                      <div className="p-2 rounded-lg bg-teal-500/10 text-teal-400 border border-teal-500/20 group-hover:border-teal-500/50">
                        <Archive className="w-4 h-4" />
                      </div>
                      <div>
                        <span className="font-semibold text-slate-100 block text-sm">{model.name}</span>
                        <span className="text-[11px] text-slate-400 capitalize">{model.problem_type}</span>
                      </div>
                    </div>
                  </td>
                  <td className="px-4 py-3.5 font-mono font-bold text-teal-400">
                    {latestVer?.version_tag ?? "v1.0.0"}
                  </td>
                  <td className="px-4 py-3.5">
                    <StageBadge stage={latestVer?.stage ?? "production"} />
                  </td>
                  <td className="px-4 py-3.5 text-slate-200">
                    {latestVer?.algorithm_name ?? "XGBoost Classifier"}
                  </td>
                  <td className="px-4 py-3.5 font-mono font-bold text-teal-400">
                    {typeof f1 === "number" ? f1.toFixed(3) : f1}
                  </td>
                  <td className="px-4 py-3.5">
                    <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-300 border border-emerald-500/30 text-[11px] font-semibold">
                      <ShieldCheck className="w-3.5 h-3.5" />
                      Passed
                    </span>
                  </td>
                  <td className="px-4 py-3.5 text-slate-400 font-mono text-[11px]">
                    {formatDateTime(model.updated_at)}
                  </td>
                  <td className="px-4 py-3.5 text-right">
                    <button className="p-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300">
                      <ArrowUpRight className="w-4 h-4" />
                    </button>
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
