"use client";

import React, { useState } from "react";
import { X, Sliders, ShieldCheck, AlertTriangle } from "lucide-react";

interface CanarySplitModalProps {
  isOpen: boolean;
  onClose: () => void;
  deploymentName: string;
  currentCanary: number;
  onUpdateSplit: (canaryPercentage: number) => Promise<void>;
}

export function CanarySplitModal({
  isOpen,
  onClose,
  deploymentName,
  currentCanary,
  onUpdateSplit,
}: CanarySplitModalProps) {
  const [split, setSplit] = useState(currentCanary);
  const [loading, setLoading] = useState(false);

  if (!isOpen) return null;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    try {
      await onUpdateSplit(split);
      onClose();
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/80 backdrop-blur-sm p-4">
      <div className="bg-slate-900 border border-slate-800 rounded-2xl w-full max-w-md shadow-2xl overflow-hidden">
        <div className="flex items-center justify-between p-6 border-b border-slate-800">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-xl bg-teal-500/10 text-teal-400 border border-teal-500/20">
              <Sliders className="w-5 h-5" />
            </div>
            <div>
              <h3 className="font-bold text-slate-100 text-lg">Adjust Canary Traffic Split</h3>
              <p className="text-xs text-slate-400 font-mono">{deploymentName}</p>
            </div>
          </div>
          <button onClick={onClose} className="p-1 rounded-lg text-slate-400 hover:text-slate-200">
            <X className="w-5 h-5" />
          </button>
        </div>

        <form onSubmit={handleSubmit} className="p-6 space-y-6">
          <div className="space-y-3">
            <div className="flex items-center justify-between text-sm">
              <div className="flex flex-col">
                <span className="font-semibold text-slate-200">Champion (Prod)</span>
                <span className="text-xs text-slate-400 font-mono">{(100 - split).toFixed(0)}% Traffic</span>
              </div>
              <div className="flex flex-col text-right">
                <span className="font-semibold text-teal-400">Challenger (Canary)</span>
                <span className="text-xs text-teal-400/80 font-mono">{split.toFixed(0)}% Traffic</span>
              </div>
            </div>

            <input
              type="range"
              min="0"
              max="100"
              step="5"
              value={split}
              onChange={(e) => setSplit(Number(e.target.value))}
              className="w-full h-2.5 bg-slate-950 rounded-lg appearance-none cursor-pointer accent-teal-500"
            />

            <div className="flex justify-between text-[11px] text-slate-500 font-mono">
              <span>0% (Halted)</span>
              <span>25% (Trial)</span>
              <span>50% (A/B)</span>
              <span>100% (Full Promotion)</span>
            </div>
          </div>

          {split >= 50 && (
            <div className="p-3 rounded-xl bg-amber-500/10 border border-amber-500/20 flex items-start gap-2.5">
              <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
              <p className="text-xs text-amber-300">
                You are routing &ge; 50% of live production traffic to the challenger model. Ensure error rate alarms are monitored.
              </p>
            </div>
          )}

          <div className="flex items-center justify-end gap-3 pt-4 border-t border-slate-800">
            <button type="button" onClick={onClose} className="px-4 py-2.5 rounded-xl text-sm font-semibold text-slate-400 hover:text-slate-200">
              Cancel
            </button>
            <button
              type="submit"
              disabled={loading}
              className="flex items-center gap-2 px-5 py-2.5 rounded-xl text-sm font-semibold bg-teal-500 hover:bg-teal-400 text-slate-950 transition-colors disabled:opacity-50"
            >
              {loading ? "Updating Routing..." : "Apply Traffic Split"}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
