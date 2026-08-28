"use client";

import React from "react";
import { Rocket, Sliders, RefreshCw, Activity, ArrowUpRight } from "lucide-react";
import { Deployment } from "@/types";
import { StatusBadge } from "@/components/ui/Badges";
import { formatDateTime } from "@/lib/formatters";

interface DeploymentsTableProps {
  deployments: Deployment[];
  onOpenCanaryModal?: (deployment: Deployment) => void;
}

export function DeploymentsTable({ deployments, onOpenCanaryModal }: DeploymentsTableProps) {
  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden shadow-sm">
      <div className="overflow-x-auto">
        <table className="w-full text-left text-xs">
          <thead className="bg-slate-950/60 text-slate-400 uppercase tracking-wider font-semibold border-b border-slate-800 font-mono">
            <tr>
              <th className="px-4 py-3">Endpoint Name</th>
              <th className="px-4 py-3">Environment</th>
              <th className="px-4 py-3">Status</th>
              <th className="px-4 py-3">Traffic Strategy</th>
              <th className="px-4 py-3">Replicas</th>
              <th className="px-4 py-3">Endpoint URL</th>
              <th className="px-4 py-3 text-right">Actions</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800/60">
            {deployments.map((dep) => (
              <tr key={dep.id} className="hover:bg-slate-800/40 transition-colors">
                <td className="px-4 py-3.5">
                  <div className="flex items-center gap-3">
                    <div className="p-2 rounded-lg bg-teal-500/10 text-teal-400 border border-teal-500/20">
                      <Rocket className="w-4 h-4" />
                    </div>
                    <div>
                      <span className="font-semibold text-slate-100 block text-sm">{dep.name}</span>
                      <span className="text-[11px] text-slate-400 font-mono">{dep.id.slice(0, 8)}...</span>
                    </div>
                  </div>
                </td>
                <td className="px-4 py-3.5 capitalize font-semibold text-slate-300">
                  <span className="px-2 py-0.5 rounded bg-slate-800 border border-slate-700 text-[11px]">
                    {dep.environment}
                  </span>
                </td>
                <td className="px-4 py-3.5">
                  <StatusBadge status={dep.status} />
                </td>
                <td className="px-4 py-3.5">
                  <div className="flex items-center gap-2">
                    <span className="font-mono text-slate-200 capitalize">{dep.strategy}</span>
                    {dep.strategy === "canary" && (
                      <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-teal-500/10 text-teal-300 border border-teal-500/20">
                        {dep.canary_stage_percentage}% Canary
                      </span>
                    )}
                  </div>
                </td>
                <td className="px-4 py-3.5 font-mono text-slate-300">
                  {dep.current_replicas} / {dep.max_replicas} pods
                </td>
                <td className="px-4 py-3.5 font-mono text-xs text-slate-400">
                  /v1/predict/{dep.endpoint_path}
                </td>
                <td className="px-4 py-3.5 text-right">
                  <div className="flex items-center justify-end gap-2">
                    {dep.strategy === "canary" && onOpenCanaryModal && (
                      <button
                        onClick={() => onOpenCanaryModal(dep)}
                        className="flex items-center gap-1 px-2.5 py-1 rounded-lg bg-teal-500/10 hover:bg-teal-500/20 text-teal-300 text-xs font-semibold border border-teal-500/20 transition-colors"
                      >
                        <Sliders className="w-3.5 h-3.5" />
                        Split
                      </button>
                    )}
                    <button className="p-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300">
                      <ArrowUpRight className="w-4 h-4" />
                    </button>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
