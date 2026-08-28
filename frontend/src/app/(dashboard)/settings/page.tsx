"use client";

import React, { useState } from "react";
import { Shield, Key, Users, Building, Plus, Copy, CheckCircle2 } from "lucide-react";
import { SectionCard } from "@/components/ui/Cards";
import { StatusBadge } from "@/components/ui/Badges";
import { useAuthStore } from "@/stores/authStore";

export default function SettingsPage() {
  const { user } = useAuthStore();
  const [copiedKey, setCopiedKey] = useState(false);

  const copySampleKey = () => {
    setCopiedKey(true);
    setTimeout(() => setCopiedKey(false), 2000);
  };

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 pb-2 border-b border-slate-800">
        <div>
          <h1 className="text-2xl font-bold text-slate-100 tracking-tight">Organization & RBAC Security Settings</h1>
          <p className="text-xs text-slate-400 mt-1">Manage team members, roles, API credentials, and multi-tenant quotas.</p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Organization Info */}
        <SectionCard title="Enterprise Workspace Details" description="Organization metadata and resource quotas">
          <div className="space-y-4 text-xs">
            <div className="grid grid-cols-2 gap-4">
              <div className="p-3 rounded-lg bg-slate-950 border border-slate-800">
                <span className="text-slate-500 block text-[10px] uppercase font-semibold">Organization Plan</span>
                <span className="font-bold text-teal-400 text-sm">Enterprise Tier</span>
              </div>
              <div className="p-3 rounded-lg bg-slate-950 border border-slate-800">
                <span className="text-slate-500 block text-[10px] uppercase font-semibold">Storage Quota</span>
                <span className="font-bold text-slate-200 text-sm">4.8 GB / 5,000 GB</span>
              </div>
            </div>

            <div>
              <label className="block font-semibold text-slate-300 mb-1">Organization Slug</label>
              <input
                type="text"
                disabled
                defaultValue="modelforge-enterprise"
                className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-slate-400 font-mono"
              />
            </div>
          </div>
        </SectionCard>

        {/* API Keys Management */}
        <SectionCard
          title="Programmatic API Keys (SDK & CLI)"
          description="Authenticate ModelForge Python SDK, Airflow, and REST pipelines"
          action={
            <button className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-teal-500 hover:bg-teal-400 text-slate-950 font-bold text-xs shadow-md transition-all">
              <Plus className="w-3.5 h-3.5" />
              Generate API Key
            </button>
          }
        >
          <div className="space-y-3 text-xs">
            <div className="p-3.5 rounded-xl bg-slate-950 border border-slate-800 flex items-center justify-between">
              <div>
                <span className="font-semibold text-slate-200 block">Production Service Key</span>
                <span className="text-slate-500 font-mono text-[11px]">mf_live_99a842f...</span>
              </div>
              <div className="flex items-center gap-2">
                <StatusBadge status="active" />
                <button
                  onClick={copySampleKey}
                  className="p-1.5 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700 text-xs flex items-center gap-1"
                >
                  {copiedKey ? <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
                  {copiedKey ? "Copied" : "Copy"}
                </button>
              </div>
            </div>
          </div>
        </SectionCard>
      </div>

      {/* Team Members & Granular Roles */}
      <SectionCard title="Team Members & RBAC Permissions" description="Configured users and assigned enterprise roles">
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="bg-slate-950/60 text-slate-400 uppercase tracking-wider font-semibold border-b border-slate-800">
              <tr>
                <th className="px-4 py-3">Member Name</th>
                <th className="px-4 py-3">Email Address</th>
                <th className="px-4 py-3">Assigned Role</th>
                <th className="px-4 py-3">MFA Status</th>
                <th className="px-4 py-3">Account Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60">
              <tr className="hover:bg-slate-800/40 transition-colors">
                <td className="px-4 py-3.5 text-slate-100 font-semibold">{user?.full_name || "System Admin"}</td>
                <td className="px-4 py-3.5 text-slate-400 font-mono">{user?.email || "admin@modelforge.ai"}</td>
                <td className="px-4 py-3.5">
                  <span className="px-2 py-0.5 rounded font-bold uppercase text-[10px] bg-purple-500/20 text-purple-300 border border-purple-500/30">
                    SUPER_ADMIN
                  </span>
                </td>
                <td className="px-4 py-3.5 text-emerald-400 font-semibold text-[11px]">Enforced</td>
                <td className="px-4 py-3.5">
                  <StatusBadge status="active" />
                </td>
              </tr>
              <tr className="hover:bg-slate-800/40 transition-colors">
                <td className="px-4 py-3.5 text-slate-100 font-semibold">Lead ML Engineer</td>
                <td className="px-4 py-3.5 text-slate-400 font-mono">ml-lead@modelforge.ai</td>
                <td className="px-4 py-3.5">
                  <span className="px-2 py-0.5 rounded font-bold uppercase text-[10px] bg-teal-500/20 text-teal-300 border border-teal-500/30">
                    ML_ENGINEER
                  </span>
                </td>
                <td className="px-4 py-3.5 text-emerald-400 font-semibold text-[11px]">Enforced</td>
                <td className="px-4 py-3.5">
                  <StatusBadge status="active" />
                </td>
              </tr>
              <tr className="hover:bg-slate-800/40 transition-colors">
                <td className="px-4 py-3.5 text-slate-100 font-semibold">Lead Model Reviewer</td>
                <td className="px-4 py-3.5 text-slate-400 font-mono">reviewer@modelforge.ai</td>
                <td className="px-4 py-3.5">
                  <span className="px-2 py-0.5 rounded font-bold uppercase text-[10px] bg-amber-500/20 text-amber-300 border border-amber-500/30">
                    MODEL_REVIEWER
                  </span>
                </td>
                <td className="px-4 py-3.5 text-emerald-400 font-semibold text-[11px]">Enforced</td>
                <td className="px-4 py-3.5">
                  <StatusBadge status="active" />
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </SectionCard>
    </div>
  );
}
