"use client";

import React, { useState } from "react";
import Link from "next/link";
import { TrendingDown, AlertTriangle, CheckCircle2, Play, RefreshCw, Layers } from "lucide-react";
import { StatCard, SectionCard } from "@/components/ui/Cards";
import { SeverityBadge } from "@/components/ui/Badges";

interface DriftFeatureItem {
  featureName: string;
  psiScore: number;
  ksPValue: number;
  wassersteinDist: number;
  status: string;
  severity: string;
}

const mockDriftFeatures: DriftFeatureItem[] = [
  { featureName: "amount", psiScore: 0.042, ksPValue: 0.42, wassersteinDist: 0.018, status: "Stable", severity: "none" },
  { featureName: "credit_utilization", psiScore: 0.089, ksPValue: 0.18, wassersteinDist: 0.034, status: "Stable", severity: "none" },
  { featureName: "account_age_months", psiScore: 0.061, ksPValue: 0.28, wassersteinDist: 0.022, status: "Stable", severity: "none" },
  { featureName: "num_failed_logins", psiScore: 0.035, ksPValue: 0.55, wassersteinDist: 0.012, status: "Stable", severity: "none" },
  { featureName: "ip_risk_score", psiScore: 0.142, ksPValue: 0.04, wassersteinDist: 0.082, status: "Moderate Shift", severity: "warning" },
  { featureName: "is_international", psiScore: 0.021, ksPValue: 0.72, wassersteinDist: 0.008, status: "Stable", severity: "none" },
];

export default function DriftDashboardPage() {
  const [features, setFeatures] = useState<DriftFeatureItem[]>(mockDriftFeatures);
  const [isScanning, setIsScanning] = useState(false);

  const handleRunDriftScan = () => {
    setIsScanning(true);
    setTimeout(() => {
      setIsScanning(false);
    }, 1500);
  };

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 pb-2 border-b border-slate-800">
        <div>
          <h1 className="text-2xl font-bold text-slate-100 tracking-tight flex items-center gap-2">
            <span>Statistical Data & Concept Drift Center</span>
          </h1>
          <p className="text-xs text-slate-400 mt-1">
            Detect population distribution shifts using PSI, Kolmogorov-Smirnov (KS-test), and Wasserstein distance.
          </p>
        </div>
        <div className="flex items-center gap-2.5">
          <button
            onClick={handleRunDriftScan}
            disabled={isScanning}
            className="flex items-center gap-2 px-3.5 py-2 rounded-lg bg-teal-500 hover:bg-teal-400 text-slate-950 font-bold text-xs shadow-lg shadow-teal-500/20 transition-all disabled:opacity-50"
          >
            <RefreshCw className={`w-4 h-4 ${isScanning ? "animate-spin" : ""}`} />
            {isScanning ? "Evaluating Statistical Tests..." : "Run Statistical Drift Scan"}
          </button>
        </div>
      </div>

      {/* Drift Summary Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          title="Overall Drift Score (PSI)"
          value="0.065"
          change="Green / Stable"
          isPositive={true}
          icon={TrendingDown}
          iconColor="text-emerald-400"
          subtext="Threshold: PSI &ge; 0.25 (Critical)"
        />
        <StatCard
          title="Drifted Features"
          value="1 of 6"
          change="Moderate Alert"
          isPositive={false}
          icon={AlertTriangle}
          iconColor="text-amber-400"
          subtext="Feature: ip_risk_score (PSI 0.14)"
        />
        <StatCard
          title="Concept Drift (F1 Drop)"
          value="-0.4%"
          change="Acceptable (<5%)"
          isPositive={true}
          icon={CheckCircle2}
          iconColor="text-teal-400"
          subtext="Moving 7-day F1 Evaluation"
        />
        <StatCard
          title="Auto Retraining Policy"
          value="Standby"
          change="Trigger at PSI &ge; 0.25"
          isPositive={true}
          icon={Layers}
          iconColor="text-purple-400"
          subtext="Automatic Candidate Generation"
        />
      </div>

      {/* Feature-Level Drift Table */}
      <SectionCard
        title="Feature-by-Feature Statistical Drift Metrics"
        description="Continuous monitoring comparing baseline training dataset distribution vs last 10,000 live inferences"
        action={
          <Link href="/retraining" className="text-xs text-teal-400 hover:underline">
            Configure Retraining Policies &rarr;
          </Link>
        }
      >
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="bg-slate-950/60 text-slate-400 uppercase tracking-wider font-semibold border-b border-slate-800">
              <tr>
                <th className="px-4 py-3">Feature Name</th>
                <th className="px-4 py-3">PSI Metric</th>
                <th className="px-4 py-3">KS-Test (p-value)</th>
                <th className="px-4 py-3">Wasserstein Dist</th>
                <th className="px-4 py-3">Statistical Status</th>
                <th className="px-4 py-3">Severity</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 font-mono">
              {features.map((f) => (
                <tr key={f.featureName} className="hover:bg-slate-800/40 transition-colors">
                  <td className="px-4 py-3.5 text-slate-100 font-semibold font-sans">{f.featureName}</td>
                  <td className="px-4 py-3.5 font-bold text-slate-200">
                    <span className={f.psiScore >= 0.1 ? "text-amber-400" : "text-emerald-400"}>
                      {f.psiScore.toFixed(4)}
                    </span>
                  </td>
                  <td className="px-4 py-3.5 text-slate-300">{f.ksPValue.toFixed(4)}</td>
                  <td className="px-4 py-3.5 text-slate-300">{f.wassersteinDist.toFixed(4)}</td>
                  <td className="px-4 py-3.5 font-sans text-slate-300">{f.status}</td>
                  <td className="px-4 py-3.5 font-sans">
                    <SeverityBadge severity={f.severity} />
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
