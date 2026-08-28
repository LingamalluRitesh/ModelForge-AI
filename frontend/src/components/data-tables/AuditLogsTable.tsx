"use client";

import React from "react";
import { ShieldCheck, ShieldAlert, Key, User, Clock } from "lucide-react";
import { AuditLog } from "@/types";
import { formatDateTime } from "@/lib/formatters";

interface AuditLogsTableProps {
  logs: AuditLog[];
}

export function AuditLogsTable({ logs }: AuditLogsTableProps) {
  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden shadow-sm">
      <div className="overflow-x-auto">
        <table className="w-full text-left text-xs">
          <thead className="bg-slate-950/60 text-slate-400 uppercase tracking-wider font-semibold border-b border-slate-800 font-mono">
            <tr>
              <th className="px-4 py-3">Timestamp</th>
              <th className="px-4 py-3">Action</th>
              <th className="px-4 py-3">Resource Type</th>
              <th className="px-4 py-3">Resource ID</th>
              <th className="px-4 py-3">User / Identity</th>
              <th className="px-4 py-3">Status</th>
              <th className="px-4 py-3">IP Address</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800/60">
            {logs.map((log) => (
              <tr key={log.id} className="hover:bg-slate-800/40 transition-colors font-mono">
                <td className="px-4 py-3.5 text-slate-400 text-[11px]">
                  {formatDateTime(log.timestamp)}
                </td>
                <td className="px-4 py-3.5 font-bold text-slate-200">
                  {log.action}
                </td>
                <td className="px-4 py-3.5">
                  <span className="px-2 py-0.5 rounded bg-slate-800 border border-slate-700 text-[10px] uppercase text-slate-300">
                    {log.resource_type}
                  </span>
                </td>
                <td className="px-4 py-3.5 text-slate-400 text-[11px]">
                  {log.resource_id ? `${log.resource_id.slice(0, 8)}...` : "—"}
                </td>
                <td className="px-4 py-3.5 text-slate-300 font-sans">
                  {log.user?.email ?? "System Service Worker"}
                </td>
                <td className="px-4 py-3.5">
                  <span
                    className={`inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-bold uppercase border ${
                      log.status === "SUCCESS"
                        ? "bg-emerald-500/10 text-emerald-300 border-emerald-500/30"
                        : "bg-rose-500/10 text-rose-300 border-rose-500/30"
                    }`}
                  >
                    {log.status === "SUCCESS" ? <ShieldCheck className="w-3 h-3" /> : <ShieldAlert className="w-3 h-3" />}
                    {log.status}
                  </span>
                </td>
                <td className="px-4 py-3.5 text-slate-500 text-[11px]">
                  {log.ip_address || "127.0.0.1"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
