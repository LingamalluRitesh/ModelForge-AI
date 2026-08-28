"use client";

import React, { useState } from "react";
import { Database, Layers, ArrowRight, RefreshCw, Key, CheckCircle2, Clock, Zap, Search, Plus } from "lucide-react";
import { Feature } from "@/types";

export function FeatureStoreStudio() {
  const [activeTab, setActiveTab] = useState<"catalog" | "entities" | "asof_joins" | "online_cache">("catalog");
  const [searchQuery, setSearchQuery] = useState("");

  const mockFeatures: Feature[] = [
    {
      id: "f1",
      project_id: "p1",
      name: "avg_transaction_amount_7d",
      data_type: "float64",
      transformation_logic: "AVG(amount) OVER (PARTITION BY customer_id ORDER BY tx_time RANGE BETWEEN INTERVAL 7 DAYS PRECEDING AND CURRENT ROW)",
      description: "Rolling 7-day average spend per customer account",
      tags: ["transactions", "rolling-window", "fraud"],
      validation_rules: { min: 0.0, max: 100000.0, max_null_pct: 0.01 },
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
    },
    {
      id: "f2",
      project_id: "p1",
      name: "failed_login_velocity_1h",
      data_type: "int64",
      transformation_logic: "COUNT(login_id) FILTER (WHERE status = 'FAILED') OVER (PARTITION BY user_id ORDER BY event_time RANGE BETWEEN INTERVAL 1 HOUR PRECEDING AND CURRENT ROW)",
      description: "High-frequency failed authentication attempts in the preceding hour",
      tags: ["security", "velocity", "real-time"],
      validation_rules: { min: 0, max: 100 },
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
    },
    {
      id: "f3",
      project_id: "p1",
      name: "merchant_risk_category_encoded",
      data_type: "int64",
      transformation_logic: "TargetEncoding(merchant_mcc, smoothing=10.0, cv=5)",
      description: "Out-of-fold target encoded merchant risk category index",
      tags: ["merchant", "encoding"],
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
    },
  ];

  const filteredFeatures = mockFeatures.filter((f) =>
    f.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
    f.tags?.some((t) => t.toLowerCase().includes(searchQuery.toLowerCase()))
  );

  return (
    <div className="space-y-6">
      {/* Top Header & Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="p-4 rounded-xl bg-slate-900 border border-slate-800">
          <span className="text-xs text-slate-400 font-medium">Registered Features</span>
          <p className="text-2xl font-bold font-mono text-slate-100 mt-1">128</p>
          <span className="text-[11px] text-teal-400 mt-1 block">Across 14 Feature Views</span>
        </div>
        <div className="p-4 rounded-xl bg-slate-900 border border-slate-800">
          <span className="text-xs text-slate-400 font-medium">Online Cache (Redis)</span>
          <p className="text-2xl font-bold font-mono text-emerald-400 mt-1">0.42 ms</p>
          <span className="text-[11px] text-slate-400 mt-1 block">p99 Read Latency</span>
        </div>
        <div className="p-4 rounded-xl bg-slate-900 border border-slate-800">
          <span className="text-xs text-slate-400 font-medium">Point-In-Time Joins</span>
          <p className="text-2xl font-bold font-mono text-teal-400 mt-1">100%</p>
          <span className="text-[11px] text-slate-400 mt-1 block">Zero Temporal Leakage</span>
        </div>
        <div className="p-4 rounded-xl bg-slate-900 border border-slate-800">
          <span className="text-xs text-slate-400 font-medium">Sync Schedule</span>
          <p className="text-2xl font-bold font-mono text-slate-100 mt-1">Hourly</p>
          <span className="text-[11px] text-emerald-400 mt-1 block">Active Streaming Sync</span>
        </div>
      </div>

      {/* Tabs */}
      <div className="flex items-center gap-2 border-b border-slate-800 pb-2">
        <button
          onClick={() => setActiveTab("catalog")}
          className={`px-4 py-2 rounded-xl text-xs font-semibold transition-all ${
            activeTab === "catalog" ? "bg-teal-500/10 text-teal-300 border border-teal-500/30" : "text-slate-400 hover:text-slate-200"
          }`}
        >
          Feature Catalog
        </button>
        <button
          onClick={() => setActiveTab("entities")}
          className={`px-4 py-2 rounded-xl text-xs font-semibold transition-all ${
            activeTab === "entities" ? "bg-teal-500/10 text-teal-300 border border-teal-500/30" : "text-slate-400 hover:text-slate-200"
          }`}
        >
          Primary Entities
        </button>
        <button
          onClick={() => setActiveTab("asof_joins")}
          className={`px-4 py-2 rounded-xl text-xs font-semibold transition-all ${
            activeTab === "asof_joins" ? "bg-teal-500/10 text-teal-300 border border-teal-500/30" : "text-slate-400 hover:text-slate-200"
          }`}
        >
          Point-In-Time Join Simulator
        </button>
        <button
          onClick={() => setActiveTab("online_cache")}
          className={`px-4 py-2 rounded-xl text-xs font-semibold transition-all ${
            activeTab === "online_cache" ? "bg-teal-500/10 text-teal-300 border border-teal-500/30" : "text-slate-400 hover:text-slate-200"
          }`}
        >
          Online Store KV Viewer
        </button>
      </div>

      {/* Tab: Catalog */}
      {activeTab === "catalog" && (
        <div className="space-y-4">
          <div className="flex items-center justify-between gap-4">
            <div className="relative flex-1">
              <Search className="w-4 h-4 text-slate-500 absolute left-3 top-1/2 -translate-y-1/2" />
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Search features by name, entity, or tag..."
                className="w-full pl-9 pr-4 py-2 rounded-xl bg-slate-900 border border-slate-800 text-xs text-slate-100 placeholder-slate-500 focus:outline-none focus:border-teal-500"
              />
            </div>
            <button className="flex items-center gap-2 px-4 py-2 rounded-xl bg-teal-500 hover:bg-teal-400 text-slate-950 text-xs font-semibold transition-colors">
              <Plus className="w-4 h-4" />
              Register Feature
            </button>
          </div>

          <div className="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden">
            <table className="w-full text-left text-xs">
              <thead className="bg-slate-950/60 text-slate-400 uppercase tracking-wider font-semibold border-b border-slate-800 font-mono">
                <tr>
                  <th className="px-4 py-3">Feature Name</th>
                  <th className="px-4 py-3">Type</th>
                  <th className="px-4 py-3">Transformation Logic</th>
                  <th className="px-4 py-3">Tags</th>
                  <th className="px-4 py-3">Validation Rules</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60">
                {filteredFeatures.map((f) => (
                  <tr key={f.id} className="hover:bg-slate-800/40 transition-colors">
                    <td className="px-4 py-3.5 font-bold font-mono text-teal-400">{f.name}</td>
                    <td className="px-4 py-3.5">
                      <span className="px-2 py-0.5 rounded bg-slate-800 border border-slate-700 text-[10px] font-mono text-slate-300">
                        {f.data_type}
                      </span>
                    </td>
                    <td className="px-4 py-3.5 font-mono text-slate-300 text-[11px] max-w-md truncate">
                      {f.transformation_logic}
                    </td>
                    <td className="px-4 py-3.5">
                      <div className="flex gap-1 flex-wrap">
                        {f.tags?.map((t) => (
                          <span key={t} className="px-1.5 py-0.5 rounded bg-slate-800 text-[10px] text-slate-400">
                            {t}
                          </span>
                        ))}
                      </div>
                    </td>
                    <td className="px-4 py-3.5 text-slate-400 font-mono text-[11px]">
                      {f.validation_rules ? JSON.stringify(f.validation_rules) : "None"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Tab: AS-OF Join Simulator */}
      {activeTab === "asof_joins" && (
        <div className="p-6 rounded-2xl bg-slate-900 border border-slate-800 space-y-4">
          <div className="flex items-center gap-3">
            <Zap className="w-5 h-5 text-teal-400" />
            <div>
              <h4 className="font-semibold text-sm text-slate-100">Point-In-Time (AS-OF) Join Engine</h4>
              <p className="text-xs text-slate-400">Prevents lookahead data leakage by matching observations strictly to past feature state</p>
            </div>
          </div>

          <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 font-mono text-xs text-slate-300 space-y-2">
            <div className="text-teal-400 font-bold">SELECT</div>
            <div className="pl-4">e.customer_id, e.observation_timestamp, e.target_label,</div>
            <div className="pl-4">f.avg_transaction_amount_7d, f.failed_login_velocity_1h</div>
            <div className="text-teal-400 font-bold">FROM</div>
            <div className="pl-4">observation_entities e</div>
            <div className="text-teal-400 font-bold">ASOF JOIN</div>
            <div className="pl-4">feature_view_transactions f</div>
            <div className="pl-4">ON e.customer_id = f.customer_id AND f.feature_timestamp &le; e.observation_timestamp</div>
            <div className="text-emerald-400 text-[11px] pt-2">✓ Verified: 0 future rows leaked across 50,000 observations</div>
          </div>
        </div>
      )}
    </div>
  );
}
