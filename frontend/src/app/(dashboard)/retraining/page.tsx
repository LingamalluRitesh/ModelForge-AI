"use client";

import React, { useState } from "react";
import Link from "next/link";
import { RefreshCw, Play, CheckCircle2, Shield, Plus, ArrowRight, Trophy } from "lucide-react";
import { SectionCard } from "@/components/ui/Cards";
import { StatusBadge } from "@/components/ui/Badges";

interface RetrainingPolicyItem {
  id: string;
  name: string;
  triggerType: string;
  driftThreshold: number;
  performanceDrop: number;
  autoPromote: boolean;
  status: string;
}

interface ExecutionHistoryItem {
  id: string;
  date: string;
  reason: string;
  championF1: number;
  challengerF1: number;
  improvement: string;
  promoted: boolean;
  status: string;
}

const mockPolicies: RetrainingPolicyItem[] = [
  {
    id: "pol-1",
    name: "Automated Fraud Model Drift Retrainer",
    triggerType: "Drift PSI Threshold &ge; 0.25",
    driftThreshold: 0.25,
    performanceDrop: 0.05,
    autoPromote: false,
    status: "active",
  },
  {
    id: "pol-2",
    name: "Weekly Scheduled Refresh",
    triggerType: "Cron (Every Sunday Midnight)",
    driftThreshold: 0.20,
    performanceDrop: 0.03,
    autoPromote: true,
    status: "active",
  },
];

const mockExecutions: ExecutionHistoryItem[] = [
  {
    id: "exec-994",
    date: "2026-08-28 14:30",
    reason: "Feature Drift (ip_risk_score PSI exceeded 0.25)",
    championF1: 0.941,
    challengerF1: 0.948,
    improvement: "+0.70% (Challenger Won)",
    promoted: true,
    status: "completed",
  },
  {
    id: "exec-912",
    date: "2026-08-21 00:00",
    reason: "Weekly Scheduled Refresh",
    championF1: 0.939,
    challengerF1: 0.941,
    improvement: "+0.20% (Challenger Won)",
    promoted: false,
    status: "completed",
  },
];

export default function RetrainingPage() {
  const [policies, setPolicies] = useState<RetrainingPolicyItem[]>(mockPolicies);
  const [executions, setExecutions] = useState<ExecutionHistoryItem[]>(mockExecutions);
  const [isTriggering, setIsTriggering] = useState(false);

  const handleManualTrigger = () => {
    setIsTriggering(true);
    setTimeout(() => {
      const newExec: ExecutionHistoryItem = {
        id: `exec-${Date.now().toString().slice(-4)}`,
        date: "Just now",
        reason: "Manual Retraining Trigger",
        championF1: 0.946,
        challengerF1: 0.952,
        improvement: "+0.60% (Challenger Won)",
        promoted: false,
        status: "completed",
      };
      setExecutions([newExec, ...executions]);
      setIsTriggering(false);
    }, 1500);
  };

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 pb-2 border-b border-slate-800">
        <div>
          <h1 className="text-2xl font-bold text-slate-100 tracking-tight">Automated Retraining Engine</h1>
          <p className="text-xs text-slate-400 mt-1">
            Trigger automated pipelines upon data drift, accuracy drops, or cron schedules. Compare candidate models against active champions before promotion.
          </p>
        </div>
        <div className="flex items-center gap-2.5">
          <button
            onClick={handleManualTrigger}
            disabled={isTriggering}
            className="flex items-center gap-2 px-3.5 py-2 rounded-lg bg-teal-500 hover:bg-teal-400 text-slate-950 font-bold text-xs shadow-lg shadow-teal-500/20 transition-all disabled:opacity-50"
          >
            <Play className="w-4 h-4 fill-current" />
            {isTriggering ? "Training New Challenger..." : "Trigger Retraining Pipeline"}
          </button>
        </div>
      </div>

      {/* Retraining Policies */}
      <SectionCard title="Active Retraining Policies" description="Autonomous trigger rules and promotion constraints">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {policies.map((p) => (
            <div key={p.id} className="bg-slate-950 border border-slate-800 rounded-xl p-4 flex flex-col justify-between">
              <div>
                <div className="flex items-center justify-between mb-2">
                  <span className="font-semibold text-xs text-slate-100">{p.name}</span>
                  <StatusBadge status={p.status} />
                </div>
                <p className="text-xs text-teal-400 font-mono mt-1">{p.triggerType}</p>
              </div>
              <div className="mt-4 pt-3 border-t border-slate-800/80 flex items-center justify-between text-xs text-slate-400">
                <span>Auto-Promote: <strong className="text-slate-200">{p.autoPromote ? "Enabled" : "Approval Required"}</strong></span>
                <span className="text-[11px] font-mono text-slate-500">{p.id}</span>
              </div>
            </div>
          ))}
        </div>
      </SectionCard>

      {/* Execution History & Comparison Table */}
      <SectionCard title="Retraining Execution History & Model Comparisons" description="Challenger vs Champion validation scores">
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="bg-slate-950/60 text-slate-400 uppercase tracking-wider font-semibold border-b border-slate-800">
              <tr>
                <th className="px-4 py-3">Timestamp</th>
                <th className="px-4 py-3">Trigger Reason</th>
                <th className="px-4 py-3">Champion Baseline</th>
                <th className="px-4 py-3">Challenger Score</th>
                <th className="px-4 py-3">Comparison Delta</th>
                <th className="px-4 py-3">Outcome</th>
                <th className="px-4 py-3">Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 font-mono">
              {executions.map((e) => (
                <tr key={e.id} className="hover:bg-slate-800/40 transition-colors">
                  <td className="px-4 py-3 text-slate-400">{e.date}</td>
                  <td className="px-4 py-3 text-slate-200 font-sans">{e.reason}</td>
                  <td className="px-4 py-3 text-slate-300">{(e.championF1 * 100).toFixed(2)}%</td>
                  <td className="px-4 py-3 text-emerald-400 font-bold">{(e.challengerF1 * 100).toFixed(2)}%</td>
                  <td className="px-4 py-3 text-teal-300">{e.improvement}</td>
                  <td className="px-4 py-3 font-sans">
                    {e.promoted ? (
                      <span className="text-[11px] font-semibold text-purple-300 flex items-center gap-1">
                        <Trophy className="w-3.5 h-3.5 text-amber-400" />
                        Promoted to Canary
                      </span>
                    ) : (
                      <span className="text-[11px] font-semibold text-slate-400">Candidate Registered</span>
                    )}
                  </td>
                  <td className="px-4 py-3 font-sans">
                    <StatusBadge status={e.status} />
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
