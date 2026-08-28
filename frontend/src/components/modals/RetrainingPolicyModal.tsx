"use client";

import React, { useState } from "react";
import { X, RefreshCw, ShieldAlert, Sparkles } from "lucide-react";

interface RetrainingPolicyModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSubmit: (policyData: any) => Promise<void>;
}

export function RetrainingPolicyModal({ isOpen, onClose, onSubmit }: RetrainingPolicyModalProps) {
  const [name, setName] = useState("");
  const [triggerType, setTriggerType] = useState("drift_threshold");
  const [driftThreshold, setDriftThreshold] = useState(0.25);
  const [perfDropThreshold, setPerfDropThreshold] = useState(0.05);
  const [cronExpression, setCronExpression] = useState("0 0 * * 0");
  const [autoPromote, setAutoPromote] = useState(false);
  const [loading, setLoading] = useState(false);

  if (!isOpen) return null;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!name.trim()) return;
    setLoading(true);
    try {
      await onSubmit({
        name,
        trigger_type: triggerType,
        drift_score_threshold: driftThreshold,
        performance_drop_threshold: perfDropThreshold,
        cron_expression: triggerType === "schedule" ? cronExpression : null,
        auto_promote_if_passed: autoPromote,
        is_active: true,
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
              <RefreshCw className="w-5 h-5" />
            </div>
            <div>
              <h3 className="font-bold text-slate-100 text-lg">Create Automated Retraining Policy</h3>
              <p className="text-xs text-slate-400">Continuous retraining triggered by statistical drift or schedule</p>
            </div>
          </div>
          <button onClick={onClose} className="p-1 rounded-lg text-slate-400 hover:text-slate-200">
            <X className="w-5 h-5" />
          </button>
        </div>

        <form onSubmit={handleSubmit} className="p-6 space-y-4">
          <div>
            <label className="block text-xs font-semibold text-slate-300 mb-1.5">Policy Name</label>
            <input
              type="text"
              required
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="e.g., Weekly Fraud Retrain Policy"
              className="w-full px-3.5 py-2.5 rounded-xl bg-slate-950 border border-slate-800 text-sm text-slate-100 placeholder-slate-500 focus:outline-none focus:border-teal-500"
            />
          </div>

          <div>
            <label className="block text-xs font-semibold text-slate-300 mb-1.5">Retraining Trigger Condition</label>
            <select
              value={triggerType}
              onChange={(e) => setTriggerType(e.target.value)}
              className="w-full px-3.5 py-2.5 rounded-xl bg-slate-950 border border-slate-800 text-sm text-slate-100 focus:outline-none"
            >
              <option value="drift_threshold">Statistical Data Drift (PSI / KS-test exceedance)</option>
              <option value="performance_drop">Production Performance Metric Degradation</option>
              <option value="schedule">Scheduled Cron Job</option>
              <option value="data_volume">New Labeled Data Volume Threshold</option>
            </select>
          </div>

          {triggerType === "drift_threshold" && (
            <div>
              <label className="block text-xs font-semibold text-slate-300 mb-1.5">
                Population Stability Index (PSI) Threshold: <span className="text-teal-400 font-mono">{driftThreshold}</span>
              </label>
              <input
                type="range"
                min="0.10"
                max="0.50"
                step="0.01"
                value={driftThreshold}
                onChange={(e) => setDriftThreshold(Number(e.target.value))}
                className="w-full accent-teal-500"
              />
              <p className="text-[11px] text-slate-500 mt-1">PSI &ge; 0.25 indicates significant population distribution shift.</p>
            </div>
          )}

          {triggerType === "schedule" && (
            <div>
              <label className="block text-xs font-semibold text-slate-300 mb-1.5">Cron Expression (UTC)</label>
              <input
                type="text"
                value={cronExpression}
                onChange={(e) => setCronExpression(e.target.value)}
                placeholder="0 0 * * 0 (Every Sunday at Midnight)"
                className="w-full px-3.5 py-2 rounded-xl bg-slate-950 border border-slate-800 text-sm text-slate-100 font-mono focus:outline-none focus:border-teal-500"
              />
            </div>
          )}

          <div className="flex items-center gap-2 pt-2">
            <input
              type="checkbox"
              id="auto-promote"
              checked={autoPromote}
              onChange={(e) => setAutoPromote(e.target.checked)}
              className="rounded bg-slate-950 border-slate-800 text-teal-500"
            />
            <label htmlFor="auto-promote" className="text-xs text-slate-300">
              Auto-promote candidate to Canary serving if it beats champion model F1 score
            </label>
          </div>

          <div className="flex items-center justify-end gap-3 pt-4 border-t border-slate-800">
            <button type="button" onClick={onClose} className="px-4 py-2.5 rounded-xl text-sm font-semibold text-slate-400 hover:text-slate-200">
              Cancel
            </button>
            <button
              type="submit"
              disabled={loading || !name.trim()}
              className="flex items-center gap-2 px-5 py-2.5 rounded-xl text-sm font-semibold bg-teal-500 hover:bg-teal-400 text-slate-950 transition-colors disabled:opacity-50"
            >
              <Sparkles className="w-4 h-4" />
              {loading ? "Creating..." : "Save Retraining Policy"}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
