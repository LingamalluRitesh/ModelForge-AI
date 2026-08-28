"use client";

import React, { useState } from "react";
import { ShieldCheck, Search, Filter, CheckCircle2, XCircle } from "lucide-react";
import { SectionCard } from "@/components/ui/Cards";

interface AuditItem {
  id: string;
  timestamp: string;
  user: string;
  action: string;
  resource: string;
  ip: string;
  status: "SUCCESS" | "FAILURE" | "FORBIDDEN";
}

const mockAuditLogs: AuditItem[] = [
  { id: "aud-01", timestamp: "2026-08-28 15:42:10", user: "admin@modelforge.ai", action: "deployment.canary_promoted", resource: "Deployment: fraud-detection-prod (20%)", ip: "192.168.1.45", status: "SUCCESS" },
  { id: "aud-02", timestamp: "2026-08-28 14:35:22", user: "system-retraining-worker", action: "retraining.execution_completed", resource: "RetrainingPolicy: pol-1", ip: "10.0.4.12", status: "SUCCESS" },
  { id: "aud-03", timestamp: "2026-08-28 12:10:05", user: "reviewer@modelforge.ai", action: "model_registry.model_approved", resource: "ModelVersion: v2.1.0 (Production)", ip: "192.168.1.88", status: "SUCCESS" },
  { id: "aud-04", timestamp: "2026-08-28 09:20:18", user: "analyst@modelforge.ai", action: "dataset.uploaded", resource: "Dataset: Fraud Ingested Transactions (v1.0.0)", ip: "192.168.1.92", status: "SUCCESS" },
];

export default function AuditLogsPage() {
  const [search, setSearch] = useState("");

  const filtered = mockAuditLogs.filter(
    (l) =>
      l.action.toLowerCase().includes(search.toLowerCase()) ||
      l.user.toLowerCase().includes(search.toLowerCase()) ||
      l.resource.toLowerCase().includes(search.toLowerCase())
  );

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 pb-2 border-b border-slate-800">
        <div>
          <h1 className="text-2xl font-bold text-slate-100 tracking-tight flex items-center gap-2">
            <span>Immutable Audit Trail & Compliance</span>
          </h1>
          <p className="text-xs text-slate-400 mt-1">Cryptographically stamped immutable audit logs recording every governance, deployment, and security action.</p>
        </div>
      </div>

      <div className="relative max-w-md">
        <Search className="w-4 h-4 text-slate-400 absolute left-3 top-3 pointer-events-none" />
        <input
          type="text"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="Filter audit logs by user, action, or target resource..."
          className="w-full bg-slate-900 border border-slate-800 rounded-lg pl-9 pr-3 py-2 text-xs text-slate-200 focus:outline-none focus:ring-1 focus:ring-teal-500"
        />
      </div>

      <SectionCard title="Enterprise Audit Log Stream" description="Append-only immutable record stream">
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="bg-slate-950/60 text-slate-400 uppercase tracking-wider font-semibold border-b border-slate-800">
              <tr>
                <th className="px-4 py-3">Timestamp (UTC)</th>
                <th className="px-4 py-3">User Principal</th>
                <th className="px-4 py-3">Action Event</th>
                <th className="px-4 py-3">Target Resource</th>
                <th className="px-4 py-3">Client IP</th>
                <th className="px-4 py-3">Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 font-mono">
              {filtered.map((l) => (
                <tr key={l.id} className="hover:bg-slate-800/40 transition-colors">
                  <td className="px-4 py-3 text-slate-400">{l.timestamp}</td>
                  <td className="px-4 py-3 text-slate-200 font-semibold">{l.user}</td>
                  <td className="px-4 py-3 text-teal-400 font-bold">{l.action}</td>
                  <td className="px-4 py-3 text-slate-300 font-sans">{l.resource}</td>
                  <td className="px-4 py-3 text-slate-400">{l.ip}</td>
                  <td className="px-4 py-3 font-sans">
                    <span className="inline-flex items-center gap-1 text-[11px] font-bold text-emerald-400">
                      <CheckCircle2 className="w-3.5 h-3.5" />
                      {l.status}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>
    </div>
  );
}
