"use client";

import React, { useState } from "react";
import { X, Rocket, Shield, Activity, RefreshCw } from "lucide-react";

interface DeployModelModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSubmit: (config: any) => Promise<void>;
  modelVersions?: Array<{ id: string; version_tag: string; algorithm_name: string }>;
}

export function DeployModelModal({ isOpen, onClose, onSubmit, modelVersions = [] }: DeployModelModalProps) {
  const [name, setName] = useState("");
  const [endpointPath, setEndpointPath] = useState("");
  const [environment, setEnvironment] = useState("production");
  const [strategy, setStrategy] = useState("canary");
  const [canaryPercentage, setCanaryPercentage] = useState(10.0);
  const [minReplicas, setMinReplicas] = useState(2);
  const [maxReplicas, setMaxReplicas] = useState(6);
  const [autoRollback, setAutoRollback] = useState(true);
  const [loading, setLoading] = useState(false);

  if (!isOpen) return null;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!name.trim() || !endpointPath.trim()) return;
    setLoading(true);
    try {
      await onSubmit({
        name,
        endpoint_path: endpointPath,
        environment,
        strategy,
        canary_stage_percentage: strategy === "canary" ? canaryPercentage : 0,
        primary_traffic_percentage: strategy === "canary" ? 100 - canaryPercentage : 100,
        min_replicas: minReplicas,
        max_replicas: maxReplicas,
        auto_rollback_enabled: autoRollback,
      });
      onClose();
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/80 backdrop-blur-sm p-4">
      <div className="bg-slate-900 border border-slate-800 rounded-2xl w-full max-w-lg shadow-2xl overflow-hidden">
        <div className="flex items-center justify-between p-6 border-b border-slate-800">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-xl bg-teal-500/10 text-teal-400 border border-teal-500/20">
              <Rocket className="w-5 h-5" />
            </div>
            <div>
              <h3 className="font-bold text-slate-100 text-lg">Create Real-Time Serving Deployment</h3>
              <p className="text-xs text-slate-400">Deploy candidate model to autoscaling Kubernetes pods</p>
            </div>
          </div>
          <button onClick={onClose} className="p-1 rounded-lg text-slate-400 hover:text-slate-200">
            <X className="w-5 h-5" />
          </button>
        </div>

        <form onSubmit={handleSubmit} className="p-6 space-y-4">
          <div>
            <label className="block text-xs font-semibold text-slate-300 mb-1.5">Deployment Name</label>
            <input
              type="text"
              required
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="e.g., Fraud Detection Real-Time Endpoint"
              className="w-full px-3.5 py-2.5 rounded-xl bg-slate-950 border border-slate-800 text-sm text-slate-100 placeholder-slate-500 focus:outline-none focus:border-teal-500"
            />
          </div>

          <div>
            <label className="block text-xs font-semibold text-slate-300 mb-1.5">HTTP Endpoint Path</label>
            <div className="flex items-center rounded-xl bg-slate-950 border border-slate-800 px-3 py-2 text-sm">
              <span className="text-slate-500 font-mono text-xs">/v1/predict/</span>
              <input
                type="text"
                required
                value={endpointPath}
                onChange={(e) => setEndpointPath(e.target.value)}
                placeholder="fraud-prod-v2"
                className="w-full bg-transparent border-none text-slate-100 placeholder-slate-500 focus:outline-none font-mono text-xs"
              />
            </div>
          </div>

          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="block text-xs font-semibold text-slate-300 mb-1.5">Environment</label>
              <select
                value={environment}
                onChange={(e) => setEnvironment(e.target.value)}
                className="w-full px-3 py-2 rounded-xl bg-slate-950 border border-slate-800 text-xs text-slate-100 focus:outline-none"
              >
                <option value="production">Production</option>
                <option value="staging">Staging</option>
                <option value="development">Development</option>
              </select>
            </div>
            <div>
              <label className="block text-xs font-semibold text-slate-300 mb-1.5">Rollout Strategy</label>
              <select
                value={strategy}
                onChange={(e) => setStrategy(e.target.value)}
                className="w-full px-3 py-2 rounded-xl bg-slate-950 border border-slate-800 text-xs text-slate-100 focus:outline-none"
              >
                <option value="canary">Canary Rollout (Weighted)</option>
                <option value="blue_green">Blue/Green Zero-Downtime</option>
                <option value="shadow">Shadow Traffic (Mirrored)</option>
                <option value="recreate">Direct Replacement</option>
              </select>
            </div>
          </div>

          {strategy === "canary" && (
            <div className="p-3.5 rounded-xl bg-slate-950 border border-slate-800 space-y-2">
              <div className="flex items-center justify-between text-xs font-semibold">
                <span className="text-slate-300">Initial Canary Traffic Split</span>
                <span className="text-teal-400 font-mono">{canaryPercentage}% Canary</span>
              </div>
              <input
                type="range"
                min="1"
                max="50"
                step="1"
                value={canaryPercentage}
                onChange={(e) => setCanaryPercentage(Number(e.target.value))}
                className="w-full accent-teal-500"
              />
            </div>
          )}

          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="block text-xs font-semibold text-slate-300 mb-1.5">Min Replicas</label>
              <input
                type="number"
                min="1"
                max="10"
                value={minReplicas}
                onChange={(e) => setMinReplicas(Number(e.target.value))}
                className="w-full px-3 py-2 rounded-xl bg-slate-950 border border-slate-800 text-xs text-slate-100 font-mono focus:outline-none"
              />
            </div>
            <div>
              <label className="block text-xs font-semibold text-slate-300 mb-1.5">Max Replicas (HPA)</label>
              <input
                type="number"
                min="1"
                max="50"
                value={maxReplicas}
                onChange={(e) => setMaxReplicas(Number(e.target.value))}
                className="w-full px-3 py-2 rounded-xl bg-slate-950 border border-slate-800 text-xs text-slate-100 font-mono focus:outline-none"
              />
            </div>
          </div>

          <div className="flex items-center gap-2 pt-2">
            <input
              type="checkbox"
              id="auto-rollback"
              checked={autoRollback}
              onChange={(e) => setAutoRollback(e.target.checked)}
              className="rounded bg-slate-950 border-slate-800 text-teal-500 focus:ring-0"
            />
            <label htmlFor="auto-rollback" className="text-xs text-slate-300">
              Enable automated rollback if error rate exceeds 5% or latency exceeds 100ms SLA
            </label>
          </div>

          <div className="flex items-center justify-end gap-3 pt-4 border-t border-slate-800">
            <button type="button" onClick={onClose} className="px-4 py-2.5 rounded-xl text-sm font-semibold text-slate-400 hover:text-slate-200">
              Cancel
            </button>
            <button
              type="submit"
              disabled={loading || !name.trim() || !endpointPath.trim()}
              className="flex items-center gap-2 px-5 py-2.5 rounded-xl text-sm font-semibold bg-teal-500 hover:bg-teal-400 text-slate-950 transition-colors disabled:opacity-50"
            >
              <Rocket className="w-4 h-4" />
              {loading ? "Deploying..." : "Launch Deployment"}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
