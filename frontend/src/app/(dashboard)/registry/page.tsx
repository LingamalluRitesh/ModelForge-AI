"use client";

import React, { useState } from "react";
import Link from "next/link";
import {
  Archive,
  CheckCircle2,
  XCircle,
  Clock,
  ShieldCheck,
  Rocket,
  ArrowRight,
  GitBranch,
} from "lucide-react";
import { SectionCard } from "@/components/ui/Cards";
import { StageBadge, StatusBadge } from "@/components/ui/Badges";

interface RegisteredModelItem {
  id: string;
  name: string;
  problemType: string;
  versions: {
    id: string;
    versionTag: string;
    stage: string;
    algorithm: string;
    f1: number;
    accuracy: number;
    qualityGatePassed: boolean;
    createdAt: string;
  }[];
}

const mockRegisteredModels: RegisteredModelItem[] = [
  {
    id: "model-fraud-01",
    name: "Enterprise Fraud Detection Model",
    problemType: "classification",
    versions: [
      {
        id: "v2.1.0",
        versionTag: "v2.1.0",
        stage: "production",
        algorithm: "XGBoost Classifier",
        f1: 0.948,
        accuracy: 0.946,
        qualityGatePassed: true,
        createdAt: "2026-08-20",
      },
      {
        id: "v2.0.0",
        versionTag: "v2.0.0",
        stage: "staging",
        algorithm: "LightGBM Classifier",
        f1: 0.941,
        accuracy: 0.938,
        qualityGatePassed: true,
        createdAt: "2026-08-15",
      },
      {
        id: "v1.1.0",
        versionTag: "v1.1.0",
        stage: "archived",
        algorithm: "Random Forest",
        f1: 0.928,
        accuracy: 0.925,
        qualityGatePassed: true,
        createdAt: "2026-08-01",
      },
    ],
  },
];

export default function ModelRegistryPage() {
  const [models, setModels] = useState<RegisteredModelItem[]>(mockRegisteredModels);
  const [selectedModel, setSelectedModel] = useState<RegisteredModelItem>(models[0]);
  const [showApprovalModal, setShowApprovalModal] = useState(false);
  const [targetStage, setTargetStage] = useState("production");

  const handleApprove = () => {
    alert("Model version approved and promoted to Staging/Production!");
    setShowApprovalModal(false);
  };

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 pb-2 border-b border-slate-800">
        <div>
          <h1 className="text-2xl font-bold text-slate-100 tracking-tight">Model Registry & Governance</h1>
          <p className="text-xs text-slate-400 mt-1">
            Enterprise model artifact versioning, quality gate checks, and multi-stage promotion workflows.
          </p>
        </div>
        <button
          onClick={() => setShowApprovalModal(true)}
          className="flex items-center gap-2 px-3.5 py-2 rounded-lg bg-teal-500 hover:bg-teal-400 text-slate-950 font-bold text-xs shadow-lg shadow-teal-500/20 transition-all"
        >
          <ShieldCheck className="w-4 h-4" />
          Governance Approval Gate
        </button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Model Selector */}
        <div>
          <SectionCard title="Registered Models Catalog" description="Select model entity to view version tree">
            <div className="space-y-2">
              {models.map((m) => (
                <div
                  key={m.id}
                  onClick={() => setSelectedModel(m)}
                  className={`p-3.5 rounded-xl border cursor-pointer transition-all ${
                    selectedModel.id === m.id
                      ? "bg-slate-800/80 border-teal-500 ring-1 ring-teal-500/30"
                      : "bg-slate-950/40 border-slate-800 hover:border-slate-700"
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <span className="font-semibold text-xs text-slate-100">{m.name}</span>
                    <span className="text-[10px] uppercase font-bold text-teal-400 font-mono">
                      {m.versions.length} Versions
                    </span>
                  </div>
                  <span className="text-[10px] uppercase font-mono text-slate-500 mt-1 block">
                    {m.problemType}
                  </span>
                </div>
              ))}
            </div>
          </SectionCard>
        </div>

        {/* Versions Tree & Quality Gates */}
        <div className="lg:col-span-2 space-y-6">
          <SectionCard
            title={`${selectedModel.name} — Lifecycle Version Tree`}
            description="Traceable model lineage: Development &rarr; Candidate &rarr; Approved &rarr; Staging &rarr; Production"
          >
            <div className="space-y-4">
              {selectedModel.versions.map((ver) => (
                <div
                  key={ver.id}
                  className="bg-slate-950 border border-slate-800 rounded-xl p-4 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4"
                >
                  <div className="space-y-1">
                    <div className="flex items-center gap-3">
                      <span className="font-bold text-sm text-slate-100 font-mono">{ver.versionTag}</span>
                      <StageBadge stage={ver.stage} />
                      {ver.qualityGatePassed && (
                        <span className="flex items-center gap-1 text-[11px] font-semibold text-emerald-400">
                          <CheckCircle2 className="w-3.5 h-3.5" />
                          Quality Gate Passed
                        </span>
                      )}
                    </div>
                    <div className="text-xs text-slate-400">
                      Algorithm: <span className="text-slate-200 font-semibold">{ver.algorithm}</span> &bull; F1:{" "}
                      <span className="text-teal-400 font-bold font-mono">{(ver.f1 * 100).toFixed(2)}%</span>
                    </div>
                  </div>

                  <div className="flex items-center gap-2">
                    <Link
                      href="/deployments"
                      className="px-3 py-1.5 rounded-lg bg-teal-500/20 text-teal-300 border border-teal-500/30 text-xs font-semibold hover:bg-teal-500/30 transition-all flex items-center gap-1.5"
                    >
                      <Rocket className="w-3.5 h-3.5" />
                      Deploy Version &rarr;
                    </Link>
                  </div>
                </div>
              ))}
            </div>
          </SectionCard>
        </div>
      </div>

      {/* Governance Approval Modal */}
      {showApprovalModal && (
        <div className="fixed inset-0 bg-slate-950/80 backdrop-blur-sm flex items-center justify-center p-4 z-50">
          <div className="bg-slate-900 border border-slate-800 rounded-2xl max-w-md w-full p-6 shadow-2xl">
            <h3 className="text-lg font-bold text-slate-100 mb-1">Model Governance Quality Gate</h3>
            <p className="text-xs text-slate-400 mb-4">Review model evaluation metrics before production promotion.</p>

            <div className="space-y-3 text-xs bg-slate-950 p-4 rounded-xl border border-slate-800 mb-4">
              <div className="flex justify-between">
                <span className="text-slate-400">Validation F1 Score</span>
                <span className="text-emerald-400 font-bold font-mono">94.8% (&ge; 85% Gate Passed)</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Data Quality Score</span>
                <span className="text-emerald-400 font-bold font-mono">94.6% (&ge; 90% Gate Passed)</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Drift Risk Assessment</span>
                <span className="text-emerald-400 font-bold font-mono">Low (PSI 0.08 &lt; 0.25)</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Security & Dependency Audit</span>
                <span className="text-emerald-400 font-bold font-mono">Clean &bull; Passed</span>
              </div>
            </div>

            <div className="flex justify-end gap-2 pt-2 border-t border-slate-800">
              <button
                onClick={() => setShowApprovalModal(false)}
                className="px-3.5 py-1.5 rounded-lg bg-slate-800 text-slate-300 hover:bg-slate-700 text-xs"
              >
                Reject / Cancel
              </button>
              <button
                onClick={handleApprove}
                className="px-4 py-1.5 rounded-lg bg-teal-500 text-slate-950 font-bold text-xs hover:bg-teal-400 flex items-center gap-1.5"
              >
                <CheckCircle2 className="w-4 h-4" />
                Approve & Promote Model
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
