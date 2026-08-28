import React from "react";
import { clsx } from "clsx";

export function StatusBadge({ status }: { status: string }) {
  const s = status.toLowerCase();
  const styles: Record<string, string> = {
    active: "bg-emerald-500/15 text-emerald-300 border-emerald-500/30",
    ready: "bg-emerald-500/15 text-emerald-300 border-emerald-500/30",
    completed: "bg-emerald-500/15 text-emerald-300 border-emerald-500/30",
    running: "bg-blue-500/15 text-blue-300 border-blue-500/30 animate-pulse",
    queued: "bg-amber-500/15 text-amber-300 border-amber-500/30",
    processing: "bg-blue-500/15 text-blue-300 border-blue-500/30",
    failed: "bg-rose-500/15 text-rose-300 border-rose-500/30",
    terminated: "bg-slate-700/30 text-slate-400 border-slate-700",
    paused: "bg-slate-700/30 text-slate-400 border-slate-700",
  };

  const currentStyle = styles[s] || "bg-slate-800 text-slate-300 border-slate-700";

  return (
    <span
      className={clsx(
        "inline-flex items-center px-2 py-0.5 rounded-full text-[11px] font-semibold uppercase tracking-wider border",
        currentStyle
      )}
    >
      <span className="w-1.5 h-1.5 rounded-full bg-current mr-1.5 opacity-80" />
      {status}
    </span>
  );
}

export function StageBadge({ stage }: { stage: string }) {
  const s = stage.toLowerCase();
  const styles: Record<string, string> = {
    production: "bg-purple-500/20 text-purple-300 border-purple-500/40",
    staging: "bg-amber-500/20 text-amber-300 border-amber-500/40",
    candidate: "bg-blue-500/20 text-blue-300 border-blue-500/40",
    approved: "bg-emerald-500/20 text-emerald-300 border-emerald-500/40",
    validation: "bg-cyan-500/20 text-cyan-300 border-cyan-500/40",
    development: "bg-slate-700/40 text-slate-300 border-slate-600",
    archived: "bg-slate-800 text-slate-500 border-slate-700",
  };

  return (
    <span
      className={clsx(
        "inline-flex items-center px-2.5 py-0.5 rounded-md text-xs font-bold uppercase tracking-wider border",
        styles[s] || "bg-slate-800 text-slate-300 border-slate-700"
      )}
    >
      {stage}
    </span>
  );
}

export function SeverityBadge({ severity }: { severity: string }) {
  const s = severity.toLowerCase();
  const styles: Record<string, string> = {
    critical: "bg-rose-500/20 text-rose-300 border-rose-500/40",
    high: "bg-orange-500/20 text-orange-300 border-orange-500/40",
    warning: "bg-amber-500/20 text-amber-300 border-amber-500/40",
    medium: "bg-yellow-500/20 text-yellow-300 border-yellow-500/40",
    low: "bg-blue-500/20 text-blue-300 border-blue-500/40",
    info: "bg-slate-700 text-slate-300 border-slate-600",
    none: "bg-emerald-500/10 text-emerald-400 border-emerald-500/20",
  };

  return (
    <span
      className={clsx(
        "inline-flex items-center px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-wider border",
        styles[s] || "bg-slate-800 text-slate-300"
      )}
    >
      {severity}
    </span>
  );
}
