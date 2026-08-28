"use client";

import React, { useState } from "react";
import { Layers, Plus, Search, Database, ArrowRight, Tag, Activity } from "lucide-react";
import { SectionCard } from "@/components/ui/Cards";
import { StatusBadge } from "@/components/ui/Badges";

interface FeatureItem {
  id: string;
  name: string;
  entity: string;
  dataType: string;
  transformation: string;
  status: string;
  sourceDataset: string;
}

const mockFeatures: FeatureItem[] = [
  { id: "f1", name: "scaled_amount", entity: "transaction_id", dataType: "float64", transformation: "StandardScaler", status: "active", sourceDataset: "Fraud Ingested Transactions" },
  { id: "f2", name: "credit_utilization_log", entity: "customer_id", dataType: "float64", transformation: "Log1p", status: "active", sourceDataset: "Fraud Ingested Transactions" },
  { id: "f3", name: "failed_login_velocity_24h", entity: "customer_id", dataType: "int64", transformation: "RollingWindowAgg", status: "active", sourceDataset: "Fraud Ingested Transactions" },
  { id: "f4", name: "device_ip_risk_score", entity: "transaction_id", dataType: "float64", transformation: "TargetEncoder", status: "active", sourceDataset: "Fraud Ingested Transactions" },
  { id: "f5", name: "transaction_hour_cyclical_sin", entity: "transaction_id", dataType: "float64", transformation: "CyclicalSinCos", status: "active", sourceDataset: "Fraud Ingested Transactions" },
  { id: "f6", name: "is_international_flag", entity: "transaction_id", dataType: "int64", transformation: "BinaryIndicator", status: "active", sourceDataset: "Fraud Ingested Transactions" },
];

export default function FeatureStorePage() {
  const [search, setSearch] = useState("");
  const [features, setFeatures] = useState<FeatureItem[]>(mockFeatures);

  const filtered = features.filter((f) =>
    f.name.toLowerCase().includes(search.toLowerCase()) || f.entity.toLowerCase().includes(search.toLowerCase())
  );

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 pb-2 border-b border-slate-800">
        <div>
          <h1 className="text-2xl font-bold text-slate-100 tracking-tight">Enterprise Feature Store</h1>
          <p className="text-xs text-slate-400 mt-1">Catalog, discover, and reuse versioned features across offline training and low-latency online serving.</p>
        </div>
        <button className="flex items-center gap-2 px-3.5 py-2 rounded-lg bg-teal-500 hover:bg-teal-400 text-slate-950 font-bold text-xs shadow-lg shadow-teal-500/20 transition-all">
          <Plus className="w-4 h-4" />
          Register Feature
        </button>
      </div>

      <div className="relative max-w-md">
        <Search className="w-4 h-4 text-slate-400 absolute left-3 top-3 pointer-events-none" />
        <input
          type="text"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="Search features by name or entity key..."
          className="w-full bg-slate-900 border border-slate-800 rounded-lg pl-9 pr-3 py-2 text-xs text-slate-200 focus:outline-none focus:ring-1 focus:ring-teal-500"
        />
      </div>

      <SectionCard title="Feature Catalog Registry" description="6 registered entity features ready for model training and online hydration">
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="bg-slate-950/60 text-slate-400 uppercase tracking-wider font-semibold border-b border-slate-800">
              <tr>
                <th className="px-4 py-3">Feature Name</th>
                <th className="px-4 py-3">Entity Key</th>
                <th className="px-4 py-3">Data Type</th>
                <th className="px-4 py-3">Transformation</th>
                <th className="px-4 py-3">Source Dataset</th>
                <th className="px-4 py-3">Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 font-mono">
              {filtered.map((f) => (
                <tr key={f.id} className="hover:bg-slate-800/40 transition-colors">
                  <td className="px-4 py-3.5 text-slate-100 font-semibold flex items-center gap-2">
                    <Layers className="w-3.5 h-3.5 text-teal-400 font-sans" />
                    <span>{f.name}</span>
                  </td>
                  <td className="px-4 py-3.5 text-slate-300">{f.entity}</td>
                  <td className="px-4 py-3.5 text-teal-400">{f.dataType}</td>
                  <td className="px-4 py-3.5 text-slate-300 font-sans">{f.transformation}</td>
                  <td className="px-4 py-3.5 text-slate-400 font-sans">{f.sourceDataset}</td>
                  <td className="px-4 py-3.5 font-sans">
                    <StatusBadge status={f.status} />
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
