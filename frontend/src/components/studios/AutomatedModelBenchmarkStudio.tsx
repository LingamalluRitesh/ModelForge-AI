"use client";

import React, { useState } from "react";
import { Trophy, Award, Zap, Cpu, BarChart3, ArrowUpRight, CheckCircle2 } from "lucide-react";

export function AutomatedModelBenchmarkStudio() {
  const [selectedMetric, setSelectedMetric] = useState<"f1" | "accuracy" | "latency" | "throughput">("f1");

  const benchmarkModels = [
    { name: "XGBoost Tuned v2.1", algorithm: "Extreme Gradient Boosting", f1: 0.954, acc: 0.962, latencyMs: 3.2, throughputQps: 1850, isChampion: true },
    { name: "FT-Transformer v1.0", algorithm: "Feature Tokenizer Transformer", f1: 0.948, acc: 0.958, latencyMs: 8.4, throughputQps: 720, isChampion: false },
    { name: "TabNet Attentive v1.2", algorithm: "Attentive Interpretable TabNet", f1: 0.942, acc: 0.951, latencyMs: 6.8, throughputQps: 940, isChampion: false },
    { name: "LightGBM Fast v3.0", algorithm: "Light Gradient Boosting", f1: 0.951, acc: 0.959, latencyMs: 2.1, throughputQps: 2400, isChampion: false },
    { name: "Random Forest Baseline", algorithm: "Random Forest Ensemble", f1: 0.925, acc: 0.938, latencyMs: 4.5, throughputQps: 1400, isChampion: false },
  ];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="p-2.5 rounded-xl bg-teal-500/10 text-teal-400 border border-teal-500/20">
            <Trophy className="w-5 h-5" />
          </div>
          <div>
            <h3 className="text-lg font-bold text-slate-100">Automated Model Benchmark & Leaderboard</h3>
            <p className="text-xs text-slate-400">Head-to-head comparison across statistical accuracy and serving latency SLA</p>
          </div>
        </div>
      </div>

      {/* Champion Banner */}
      <div className="p-5 rounded-2xl bg-gradient-to-r from-teal-500/10 via-slate-900 to-slate-900 border border-teal-500/30 flex items-center justify-between">
        <div className="flex items-center gap-4">
          <div className="p-3 rounded-2xl bg-teal-500/20 text-teal-400 border border-teal-500/40">
            <Award className="w-8 h-8" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="px-2 py-0.5 rounded bg-teal-500/20 text-teal-300 text-[10px] font-bold uppercase tracking-wider">
                Current Champion
              </span>
              <h4 className="text-base font-bold text-slate-100">XGBoost Tuned v2.1</h4>
            </div>
            <p className="text-xs text-slate-400 mt-0.5">Top composite score: 0.954 F1 Score at 3.2ms inference latency</p>
          </div>
        </div>

        <button className="px-4 py-2 rounded-xl bg-teal-500 hover:bg-teal-400 text-slate-950 text-xs font-semibold transition-colors">
          Promote to Production
        </button>
      </div>

      {/* Leaderboard Table */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden shadow-sm">
        <table className="w-full text-left text-xs">
          <thead className="bg-slate-950/60 text-slate-400 uppercase tracking-wider font-semibold border-b border-slate-800 font-mono">
            <tr>
              <th className="px-4 py-3">Rank</th>
              <th className="px-4 py-3">Candidate Model</th>
              <th className="px-4 py-3">Architecture</th>
              <th className="px-4 py-3">F1 Score</th>
              <th className="px-4 py-3">Accuracy</th>
              <th className="px-4 py-3">p95 Latency</th>
              <th className="px-4 py-3">Throughput (QPS)</th>
              <th className="px-4 py-3">Status</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800/60 font-mono">
            {benchmarkModels.map((m, idx) => (
              <tr key={m.name} className={`hover:bg-slate-800/40 transition-colors ${m.isChampion ? "bg-teal-500/5" : ""}`}>
                <td className="px-4 py-3.5 font-bold text-slate-400">#{idx + 1}</td>
                <td className="px-4 py-3.5 font-sans font-bold text-slate-100 flex items-center gap-2">
                  {m.name}
                  {m.isChampion && (
                    <span className="w-2 h-2 rounded-full bg-teal-400 animate-pulse" />
                  )}
                </td>
                <td className="px-4 py-3.5 text-slate-400 font-sans">{m.algorithm}</td>
                <td className="px-4 py-3.5 font-bold text-teal-400">{m.f1.toFixed(3)}</td>
                <td className="px-4 py-3.5 text-slate-200">{(m.acc * 100).toFixed(1)}%</td>
                <td className="px-4 py-3.5 text-emerald-400">{m.latencyMs} ms</td>
                <td className="px-4 py-3.5 text-slate-200">{m.throughputQps.toLocaleString()}</td>
                <td className="px-4 py-3.5 font-sans">
                  <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-300 border border-emerald-500/30 text-[11px] font-semibold">
                    <CheckCircle2 className="w-3 h-3" />
                    Passed Gate
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
