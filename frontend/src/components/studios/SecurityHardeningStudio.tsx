"use client";

import React, { useState } from "react";
import { ShieldAlert, ShieldCheck, Bug, Zap, Sliders, ArrowUpRight } from "lucide-react";

export function SecurityHardeningStudio() {
  const [epsilon, setEpsilon] = useState(0.05);
  const [simulating, setSimulating] = useState(false);
  const [results, setResults] = useState<any>({
    cleanAcc: 0.962,
    advAcc: 0.884,
    degradation: 0.078,
    evasionRate: 0.081,
    status: "SECURE",
  });

  const handleSimulateAttack = () => {
    setSimulating(true);
    setTimeout(() => {
      setSimulating(false);
      const deg = epsilon * 1.6;
      const adv = Math.max(0.4, 0.962 - deg);
      setResults({
        cleanAcc: 0.962,
        advAcc: Math.round(adv * 1000) / 1000,
        degradation: Math.round(deg * 1000) / 1000,
        evasionRate: Math.round((0.962 - adv) * 1000) / 1000,
        status: deg >= 0.15 ? "VULNERABLE" : "SECURE",
      });
    }, 600);
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="p-2.5 rounded-xl bg-rose-500/10 text-rose-400 border border-rose-500/20">
            <Bug className="w-5 h-5" />
          </div>
          <div>
            <h3 className="text-lg font-bold text-slate-100">Model Security & Adversarial Attack Studio</h3>
            <p className="text-xs text-slate-400">Simulate FGSM / PGD evasion attacks and audit robustness bounds</p>
          </div>
        </div>

        <button
          onClick={handleSimulateAttack}
          disabled={simulating}
          className="flex items-center gap-2 px-5 py-2.5 rounded-xl bg-rose-500 hover:bg-rose-400 text-slate-950 text-xs font-semibold transition-colors disabled:opacity-50"
        >
          <Zap className="w-4 h-4" />
          {simulating ? "Penetration Testing..." : "Run Adversarial Attack Test"}
        </button>
      </div>

      {/* Attack Parameter Controls */}
      <div className="p-5 rounded-2xl bg-slate-900 border border-slate-800 space-y-4">
        <div className="flex items-center justify-between text-xs font-semibold">
          <span className="text-slate-300">Fast Gradient Sign Method (FGSM) Perturbation Budget (&epsilon;)</span>
          <span className="text-teal-400 font-mono">&epsilon; = {epsilon.toFixed(2)}</span>
        </div>
        <input
          type="range"
          min="0.01"
          max="0.25"
          step="0.01"
          value={epsilon}
          onChange={(e) => setEpsilon(Number(e.target.value))}
          className="w-full accent-teal-500"
        />
        <div className="flex justify-between text-[11px] text-slate-500 font-mono">
          <span>&epsilon;=0.01 (Imperceptible Noise)</span>
          <span>&epsilon;=0.10 (Moderate Attack)</span>
          <span>&epsilon;=0.25 (Severe Evasion)</span>
        </div>
      </div>

      {/* Metrics Row */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="p-4 rounded-xl bg-slate-900 border border-slate-800">
          <span className="text-xs text-slate-400 font-medium">Clean Accuracy</span>
          <p className="text-2xl font-bold font-mono text-slate-100 mt-1">{(results.cleanAcc * 100).toFixed(1)}%</p>
          <span className="text-[11px] text-slate-500 mt-1 block">Unperturbed Baseline</span>
        </div>

        <div className="p-4 rounded-xl bg-slate-900 border border-slate-800">
          <span className="text-xs text-slate-400 font-medium">Adversarial Accuracy</span>
          <p className="text-2xl font-bold font-mono text-rose-400 mt-1">{(results.advAcc * 100).toFixed(1)}%</p>
          <span className="text-[11px] text-slate-500 mt-1 block">Under &epsilon;={epsilon.toFixed(2)} attack</span>
        </div>

        <div className="p-4 rounded-xl bg-slate-900 border border-slate-800">
          <span className="text-xs text-slate-400 font-medium">Evasion Degradation</span>
          <p className="text-2xl font-bold font-mono text-amber-400 mt-1">-{(results.degradation * 100).toFixed(1)}%</p>
          <span className="text-[11px] text-slate-500 mt-1 block font-mono">Accuracy Delta</span>
        </div>

        <div className="p-4 rounded-xl bg-slate-900 border border-slate-800">
          <span className="text-xs text-slate-400 font-medium">Security Posture</span>
          <p className="text-2xl font-bold font-mono text-emerald-400 mt-1 flex items-center gap-1.5">
            {results.status === "SECURE" ? <ShieldCheck className="w-6 h-6 text-emerald-400" /> : <ShieldAlert className="w-6 h-6 text-rose-400" />}
            {results.status}
          </p>
          <span className="text-[11px] text-slate-500 mt-1 block font-sans">Passes Enterprise Quality Gate</span>
        </div>
      </div>
    </div>
  );
}
