"use client";

import React from "react";
import { VisualDAGPipelineBuilder } from "@/components/pipelines/DAGCanvas";

export default function PipelinesPage() {
  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 pb-2 border-b border-slate-800">
        <div>
          <h1 className="text-2xl font-bold text-slate-100 tracking-tight">Visual ML Pipeline Builder & DAG Scheduler</h1>
          <p className="text-xs text-slate-400 mt-1">
            Build and orchestrate multi-step pipelines: Data Ingestion &rarr; Quality Gate &rarr; Feature Engineering &rarr; Training &rarr; Evaluation &rarr; Governance &rarr; Canary Deployment.
          </p>
        </div>
      </div>

      <VisualDAGPipelineBuilder />
    </div>
  );
}
