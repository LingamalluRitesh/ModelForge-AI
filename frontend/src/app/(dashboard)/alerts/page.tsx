"use client";

import React, { useState } from "react";
import { Bell, CheckCircle2, ShieldAlert, AlertTriangle, Info, Send, Plus } from "lucide-react";
import { SectionCard } from "@/components/ui/Cards";
import { SeverityBadge } from "@/components/ui/Badges";

interface AlertItem {
  id: string;
  title: string;
  message: string;
  severity: string;
  source: string;
  timestamp: string;
  isAck: boolean;
}

const mockAlerts: AlertItem[] = [
  {
    id: "alt-01",
    title: "Moderate Data Drift Warning on 'ip_risk_score'",
    message: "Population Stability Index (PSI) reached 0.142 on live inference traffic (Warning threshold: 0.10).",
    severity: "warning",
    source: "Deployment: fraud-detection-prod",
    timestamp: "10 minutes ago",
    isAck: false,
  },
  {
    id: "alt-02",
    title: "Quality Gate Verification Succeeded",
    message: "Candidate model v2.2.0 passed all 5 automated governance quality gates (F1 94.8% >= 85%).",
    severity: "info",
    source: "Model Registry: Fraud Detection",
    timestamp: "1 hour ago",
    isAck: true,
  },
];

export default function AlertsPage() {
  const [alerts, setAlerts] = useState<AlertItem[]>(mockAlerts);

  const acknowledgeAlert = (id: string) => {
    setAlerts((prev) =>
      prev.map((a) => (a.id === id ? { ...a, isAck: true } : a))
    );
  };

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 pb-2 border-b border-slate-800">
        <div>
          <h1 className="text-2xl font-bold text-slate-100 tracking-tight">Alerts & Notification Webhooks</h1>
          <p className="text-xs text-slate-400 mt-1">Configure threshold rules, webhooks, and manage real-time platform incident alerts.</p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left: Alerts Stream */}
        <div className="lg:col-span-2 space-y-4">
          <SectionCard title="Active Incident Feed" description="Real-time alerts triggered by drift, latencies, or failures">
            <div className="space-y-3">
              {alerts.map((a) => (
                <div
                  key={a.id}
                  className={`p-4 rounded-xl border flex items-start justify-between gap-4 transition-all ${
                    a.isAck
                      ? "bg-slate-950/40 border-slate-800 opacity-60"
                      : "bg-slate-900 border-slate-700/80 shadow-md"
                  }`}
                >
                  <div className="space-y-1">
                    <div className="flex items-center gap-2.5">
                      <SeverityBadge severity={a.severity} />
                      <h4 className="font-semibold text-xs text-slate-100">{a.title}</h4>
                    </div>
                    <p className="text-xs text-slate-300">{a.message}</p>
                    <div className="flex items-center gap-3 text-[11px] text-slate-500 pt-1">
                      <span>{a.source}</span>
                      <span>&bull;</span>
                      <span>{a.timestamp}</span>
                    </div>
                  </div>

                  <div>
                    {!a.isAck ? (
                      <button
                        onClick={() => acknowledgeAlert(a.id)}
                        className="px-2.5 py-1 rounded bg-slate-800 hover:bg-slate-700 text-teal-300 border border-slate-700 text-xs font-semibold shrink-0"
                      >
                        Acknowledge
                      </button>
                    ) : (
                      <span className="text-[11px] text-slate-500 font-semibold flex items-center gap-1">
                        <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
                        Acknowledged
                      </span>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </SectionCard>
        </div>

        {/* Right: Webhook Channels */}
        <div>
          <SectionCard title="Notification Channels" description="Configured Slack, PagerDuty, and Webhook dispatchers">
            <div className="space-y-3 text-xs">
              <div className="p-3 rounded-lg bg-slate-950 border border-slate-800 flex items-center justify-between">
                <div>
                  <span className="font-semibold text-slate-200 block">Slack Operations Channel</span>
                  <span className="text-slate-500 font-mono text-[10px]">#mlops-alerts</span>
                </div>
                <span className="text-emerald-400 font-semibold text-[11px]">Active</span>
              </div>

              <div className="p-3 rounded-lg bg-slate-950 border border-slate-800 flex items-center justify-between">
                <div>
                  <span className="font-semibold text-slate-200 block">PagerDuty On-Call</span>
                  <span className="text-slate-500 font-mono text-[10px]">Critical Severity</span>
                </div>
                <span className="text-emerald-400 font-semibold text-[11px]">Active</span>
              </div>

              <button className="w-full py-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 font-semibold text-xs border border-slate-700 flex items-center justify-center gap-1.5 mt-4">
                <Plus className="w-3.5 h-3.5" />
                Add Notification Channel
              </button>
            </div>
          </SectionCard>
        </div>
      </div>
    </div>
  );
}
