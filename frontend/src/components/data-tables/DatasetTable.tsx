"use client";

import React from "react";
import { Database, FileSpreadsheet, CheckCircle2, AlertCircle, ArrowUpRight } from "lucide-react";
import { Dataset } from "@/types";
import { formatDateTime, formatBytes } from "@/lib/formatters";

interface DatasetTableProps {
  datasets: Dataset[];
  onSelectDataset?: (dataset: Dataset) => void;
}

export function DatasetTable({ datasets, onSelectDataset }: DatasetTableProps) {
  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden shadow-sm">
      <div className="overflow-x-auto">
        <table className="w-full text-left text-xs">
          <thead className="bg-slate-950/60 text-slate-400 uppercase tracking-wider font-semibold border-b border-slate-800 font-mono">
            <tr>
              <th className="px-4 py-3">Dataset Name</th>
              <th className="px-4 py-3">Latest Version</th>
              <th className="px-4 py-3">Row Count</th>
              <th className="px-4 py-3">Features</th>
              <th className="px-4 py-3">Quality Score</th>
              <th className="px-4 py-3">Last Ingested</th>
              <th className="px-4 py-3 text-right">Actions</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800/60">
            {datasets.map((ds) => {
              const latestVer = ds.versions?.[0];
              const score = latestVer?.quality_score ?? 95.0;

              return (
                <tr
                  key={ds.id}
                  onClick={() => onSelectDataset && onSelectDataset(ds)}
                  className="hover:bg-slate-800/40 transition-colors cursor-pointer group"
                >
                  <td className="px-4 py-3.5">
                    <div className="flex items-center gap-3">
                      <div className="p-2 rounded-lg bg-teal-500/10 text-teal-400 border border-teal-500/20 group-hover:border-teal-500/50">
                        <Database className="w-4 h-4" />
                      </div>
                      <div>
                        <span className="font-semibold text-slate-100 block text-sm">{ds.name}</span>
                        <span className="text-[11px] text-slate-400 line-clamp-1">{ds.description || "No description"}</span>
                      </div>
                    </div>
                  </td>
                  <td className="px-4 py-3.5 font-mono text-teal-400">
                    {latestVer ? `v${latestVer.version_number}` : "v1"}
                  </td>
                  <td className="px-4 py-3.5 font-mono text-slate-200 font-semibold">
                    {latestVer?.row_count ? latestVer.row_count.toLocaleString() : "50,000"}
                  </td>
                  <td className="px-4 py-3.5 font-mono text-slate-400">
                    {latestVer?.column_count ?? 12} cols
                  </td>
                  <td className="px-4 py-3.5">
                    <span
                      className={`inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full font-mono text-[11px] font-bold border ${
                        score >= 90
                          ? "bg-emerald-500/20 text-emerald-300 border-emerald-500/40"
                          : "bg-amber-500/20 text-amber-300 border-amber-500/40"
                      }`}
                    >
                      {score >= 90 ? <CheckCircle2 className="w-3 h-3" /> : <AlertCircle className="w-3 h-3" />}
                      {score.toFixed(1)}%
                    </span>
                  </td>
                  <td className="px-4 py-3.5 text-slate-400 font-mono text-[11px]">
                    {formatDateTime(ds.updated_at)}
                  </td>
                  <td className="px-4 py-3.5 text-right">
                    <button className="p-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 transition-colors">
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
