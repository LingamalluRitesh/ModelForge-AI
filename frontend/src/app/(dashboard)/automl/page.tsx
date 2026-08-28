"use client";

import React, { useState } from "react";
import Link from "next/link";
import { Sparkles, Play, Trophy, CheckCircle2, Sliders, ArrowUpRight, Cpu } from "lucide-react";
import { SectionCard } from "@/components/ui/Cards";
import { StatusBadge } from "@/components/ui/Badges";
import { useProjectStore } from "@/stores/authStore";

interface LeaderboardItem {
  rank: number;
  algorithm: string;
  score: number;
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  duration: string;
  isBest?: boolean;
}

const mockLeaderboard: LeaderboardItem[] = [
  { rank: 1, algorithm: "XGBoost Classifier", score: 0.948, accuracy: 0.942, precision: 0.938, recall: 0.957, f1: 0.948, duration: "32.4s", isBest: true },
  { rank: 2, algorithm: "LightGBM Classifier", score: 0.941, accuracy: 0.938, precision: 0.931, recall: 0.951, f1: 0.941, duration: "24.1s" },
  { rank: 3, algorithm: "PyTorch Deep MLP", score: 0.934, accuracy: 0.931, precision: 0.925, recall: 0.943, f1: 0.934, duration: "84.5s" },
  { rank: 4, algorithm: "Random Forest", score: 0.928, accuracy: 0.925, precision: 0.919, recall: 0.937, f1: 0.928, duration: "48.2s" },
  { rank: 5, algorithm: "Logistic Regression", score: 0.887, accuracy: 0.884, precision: 0.871, recall: 0.903, f1: 0.887, duration: "8.2s" },
];

export default function AutoMLStudioPage() {
  const { currentProject } = useProjectStore();
  const [jobName, setJobName] = useState("automl-fraud-optimization-v1");
  const [metric, setMetric] = useState("f1");
  const [maxTrials, setMaxTrials] = useState(20);
  const [isSearching, setIsSearching] = useState(false);
  const [leaderboard, setLeaderboard] = useState<LeaderboardItem[]>(mockLeaderboard);

  const handleRunAutoML = (e: React.FormEvent) => {
    e.preventDefault();
    setIsSearching(true);
    setTimeout(() => {
      setIsSearching(false);
    }, 2000);
  };

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 pb-2 border-b border-slate-800">
        <div>
          <h1 className="text-2xl font-bold text-slate-100 tracking-tight flex items-center gap-2">
            <span>AutoML Studio & Leaderboard</span>
            <span className="text-xs px-2 py-0.5 rounded bg-indigo-500/20 text-indigo-300 border border-indigo-500/30">
              Optuna TPE
            </span>
          </h1>
          <p className="text-xs text-slate-400 mt-1">
            Automated feature preprocessing, multi-algorithm Bayesian hyperparameter optimization, and champion model selection.
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Setup Config Card */}
        <div>
          <SectionCard title="AutoML Job Setup" description="Configure optimization metric and budget">
            <form onSubmit={handleRunAutoML} className="space-y-4 text-xs">
              <div>
                <label className="block font-semibold text-slate-300 mb-1">Job Name</label>
                <input
                  type="text"
                  required
                  value={jobName}
                  onChange={(e) => setJobName(e.target.value)}
                  className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-slate-100 focus:outline-none focus:ring-1 focus:ring-teal-500"
                />
              </div>

              <div>
                <label className="block font-semibold text-slate-300 mb-1">Optimization Metric</label>
                <select
                  value={metric}
                  onChange={(e) => setMetric(e.target.value)}
                  className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-slate-100"
                >
                  <option value="f1">F1 Score (Harmonic Mean)</option>
                  <option value="accuracy">Accuracy (Correct Classifications)</option>
                  <option value="roc_auc">ROC-AUC (Area Under Curve)</option>
                  <option value="precision">Precision (Minimizing False Positives)</option>
                </select>
              </div>

              <div>
                <label className="block font-semibold text-slate-300 mb-1">Max Bayesian Trials</label>
                <input
                  type="number"
                  value={maxTrials}
                  onChange={(e) => setMaxTrials(Number(e.target.value))}
                  className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-slate-100"
                />
              </div>

              <div className="pt-2">
                <button
                  type="submit"
                  disabled={isSearching}
                  className="w-full py-2.5 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white font-bold text-xs shadow-lg shadow-indigo-600/20 transition-all flex items-center justify-center gap-2 disabled:opacity-50"
                >
                  <Sparkles className="w-4 h-4" />
                  {isSearching ? "Optimizing Bayesian Search..." : "Start AutoML Exploration"}
                </button>
              </div>
            </form>
          </SectionCard>
        </div>

        {/* Leaderboard Table Card */}
        <div className="lg:col-span-2">
          <SectionCard
            title="AutoML Model Leaderboard"
            description="Evaluated models ranked by primary validation metric"
            action={
              <Link href="/registry" className="text-xs text-teal-400 hover:underline flex items-center gap-1">
                Model Registry <ArrowUpRight className="w-3.5 h-3.5" />
              </Link>
            }
          >
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead className="bg-slate-950/60 text-slate-400 uppercase tracking-wider font-semibold border-b border-slate-800">
                  <tr>
                    <th className="px-4 py-3">Rank</th>
                    <th className="px-4 py-3">Model Candidate</th>
                    <th className="px-4 py-3">F1 Score</th>
                    <th className="px-4 py-3">Accuracy</th>
                    <th className="px-4 py-3">Precision</th>
                    <th className="px-4 py-3">Recall</th>
                    <th className="px-4 py-3">Duration</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60 font-mono">
                  {leaderboard.map((item) => (
                    <tr
                      key={item.rank}
                      className={`hover:bg-slate-800/40 transition-colors ${
                        item.isBest ? "bg-teal-500/5 font-semibold" : ""
                      }`}
                    >
                      <td className="px-4 py-3.5 text-slate-300">
                        {item.isBest ? (
                          <div className="flex items-center gap-1.5 font-bold text-amber-400 font-sans">
                            <Trophy className="w-4 h-4" />
                            <span>#1 Champion</span>
                          </div>
                        ) : (
                          `#${item.rank}`
                        )}
                      </td>
                      <td className="px-4 py-3.5 text-slate-100 font-sans font-semibold flex items-center gap-2">
                        <Cpu className="w-3.5 h-3.5 text-teal-400 shrink-0" />
                        <span>{item.algorithm}</span>
                      </td>
                      <td className="px-4 py-3.5 text-emerald-400 font-bold text-sm">
                        {(item.f1 * 100).toFixed(2)}%
                      </td>
                      <td className="px-4 py-3.5 text-slate-200">{(item.accuracy * 100).toFixed(2)}%</td>
                      <td className="px-4 py-3.5 text-slate-300">{(item.precision * 100).toFixed(2)}%</td>
                      <td className="px-4 py-3.5 text-slate-300">{(item.recall * 100).toFixed(2)}%</td>
                      <td className="px-4 py-3.5 text-slate-400 font-sans">{item.duration}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </SectionCard>
        </div>
      </div>
    </div>
  );
}
