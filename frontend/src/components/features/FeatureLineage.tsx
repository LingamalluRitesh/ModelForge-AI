"use client";

import React from "react";
import { Database, Layers, Cpu, Archive, Rocket, ArrowRight } from "lucide-react";

export function FeatureLineageGraph() {
  const steps = [
    { id: "s1", name: "Raw Ingestion", type: "Dataset", icon: Database, details: "50,000 Transactions (CSV)" },
    { id: "s2", name: "Data Quality Gate", type: "Validation", icon: Layers, details: "15 Automated Rules (94.6%)" },
    { id: "s3", name: "Feature Transformations", type: "Feature Store", icon: Layers, details: "StandardScaler + TargetEncoder" },
    { id: "s4", name: "Model Training", type: "Experiment", icon: Cpu, details: "XGBoost Classifier v2.1.0" },
    { id: "s5", name: "Model Registry", type: "Governance", icon: Archive, details: "Quality Gate Approved" },
    { id: "s6", name: "Real-Time Serving", type: "Deployment", icon: Rocket, details: "Canary 10% / Prod 90%" },
  ];

  return (
    <div className="bg-slate-950 border border-slate-800 rounded-xl p-6 shadow-sm overflow-x-auto">
      <div className="flex items-center justify-between min-w-[800px] gap-2">
        {steps.map((step, idx) => {
          const Icon = step.icon;
          return (
            <React.Fragment key={step.id}>
              <div className="flex flex-col items-center text-center p-3 rounded-xl bg-slate-900 border border-slate-800 w-44 hover:border-teal-500/50 transition-all group">
                <div className="p-2.5 rounded-lg bg-slate-800 text-teal-400 group-hover:bg-teal-500/20 group-hover:text-teal-300 transition-colors mb-2">
                  <Icon className="w-5 h-5" />
                </div>
                <span className="font-semibold text-xs text-slate-100">{step.name}</span>
                <span className="text-[10px] text-teal-400 uppercase font-mono mt-0.5">{step.type}</span>
                <span className="text-[10px] text-slate-400 mt-1 line-clamp-1">{step.details}</span>
              </div>
              {idx < steps.length - 1 && (
                <div className="flex items-center text-slate-600">
                  <ArrowRight className="w-4 h-4" />
                </div>
              )}
            </React.Fragment>
          );
        })}
      </div>
    </div>
  );
}
