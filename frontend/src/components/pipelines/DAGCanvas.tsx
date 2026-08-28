"use client";

import React, { useState } from "react";
import {
  Play,
  Save,
  Plus,
  Trash2,
  Settings2,
  Database,
  CheckCircle2,
  Cpu,
  Layers,
  Rocket,
  Bell,
  Archive,
  Eye,
} from "lucide-react";
import { clsx } from "clsx";

interface NodeItem {
  id: string;
  name: string;
  type: string;
  status: "idle" | "running" | "completed" | "failed";
  x: number;
  y: number;
}

const nodeTypeIcons: Record<string, any> = {
  dataset: Database,
  validation: CheckCircle2,
  feature_eng: Layers,
  train: Cpu,
  evaluate: Eye,
  register: Archive,
  deploy: Rocket,
  notify: Bell,
};

const initialNodes: NodeItem[] = [
  { id: "node-1", name: "Fraud Transactions Dataset", type: "dataset", status: "completed", x: 60, y: 120 },
  { id: "node-2", name: "Data Quality Gate (90%+)", type: "validation", status: "completed", x: 280, y: 120 },
  { id: "node-3", name: "Feature Scaling & Encoding", type: "feature_eng", status: "completed", x: 500, y: 120 },
  { id: "node-4", name: "XGBoost Classifier Training", type: "train", status: "running", x: 720, y: 120 },
  { id: "node-5", name: "Model Evaluation & SHAP", type: "evaluate", status: "idle", x: 940, y: 60 },
  { id: "node-6", name: "Staging Canary Deployment (10%)", type: "deploy", status: "idle", x: 1160, y: 60 },
];

export function VisualDAGPipelineBuilder() {
  const [nodes, setNodes] = useState<NodeItem[]>(initialNodes);
  const [selectedNode, setSelectedNode] = useState<NodeItem | null>(nodes[3]);
  const [isRunning, setIsRunning] = useState(false);

  const handleRun = () => {
    setIsRunning(true);
    setTimeout(() => {
      setNodes((prev) =>
        prev.map((n) => ({ ...n, status: "completed" }))
      );
      setIsRunning(false);
    }, 2500);
  };

  return (
    <div className="flex flex-col h-[700px] bg-slate-950 border border-slate-800 rounded-xl overflow-hidden shadow-2xl">
      {/* Top Toolbar */}
      <div className="h-14 bg-slate-900 border-b border-slate-800 px-6 flex items-center justify-between z-10">
        <div className="flex items-center gap-3">
          <span className="font-semibold text-sm text-slate-200">End-to-End MLOps Pipeline DAG</span>
          <span className="text-xs px-2 py-0.5 rounded bg-teal-500/10 text-teal-300 border border-teal-500/30">
            6 Nodes &bull; Active
          </span>
        </div>
        <div className="flex items-center gap-3">
          <button
            onClick={handleRun}
            disabled={isRunning}
            className="flex items-center gap-2 px-3.5 py-1.5 rounded-lg bg-teal-500 hover:bg-teal-400 text-slate-950 font-semibold text-xs transition-all shadow-lg shadow-teal-500/20 disabled:opacity-50"
          >
            <Play className="w-3.5 h-3.5 fill-current" />
            {isRunning ? "Executing DAG..." : "Run Pipeline"}
          </button>
          <button className="flex items-center gap-2 px-3.5 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs border border-slate-700">
            <Save className="w-3.5 h-3.5" />
            Save DAG
          </button>
        </div>
      </div>

      {/* Main Interactive Canvas Area */}
      <div className="flex-1 flex overflow-hidden relative">
        {/* Canvas Body */}
        <div className="flex-1 relative overflow-auto p-8 dag-canvas-bg">
          {/* Render Connections / SVG lines */}
          <svg className="absolute inset-0 w-full h-full pointer-events-none z-0">
            <path d="M 220 160 L 280 160" stroke="#14b8a6" strokeWidth="2.5" strokeDasharray="4 2" fill="none" />
            <path d="M 440 160 L 500 160" stroke="#14b8a6" strokeWidth="2.5" strokeDasharray="4 2" fill="none" />
            <path d="M 660 160 L 720 160" stroke="#14b8a6" strokeWidth="2.5" strokeDasharray="4 2" fill="none" />
            <path d="M 880 160 C 910 160, 910 100, 940 100" stroke="#0ea5e9" strokeWidth="2.5" fill="none" />
            <path d="M 1100 100 L 1160 100" stroke="#0ea5e9" strokeWidth="2.5" fill="none" />
          </svg>

          {/* Render Nodes */}
          {nodes.map((node) => {
            const Icon = nodeTypeIcons[node.type] || Cpu;
            const isSelected = selectedNode?.id === node.id;
            return (
              <div
                key={node.id}
                onClick={() => setSelectedNode(node)}
                className={clsx(
                  "absolute w-44 rounded-xl p-3.5 bg-slate-900 border cursor-pointer transition-all shadow-xl z-10",
                  isSelected
                    ? "border-teal-400 ring-2 ring-teal-500/30 scale-105"
                    : "border-slate-700 hover:border-slate-500",
                  node.status === "running" && "border-blue-400 animate-pulse",
                  node.status === "completed" && "border-emerald-500/60"
                )}
                style={{ left: `${node.x}px`, top: `${node.y}px` }}
              >
                <div className="flex items-center justify-between mb-2">
                  <div className="p-1.5 rounded-md bg-slate-800 border border-slate-700 text-teal-400">
                    <Icon className="w-4 h-4" />
                  </div>
                  <span
                    className={clsx(
                      "text-[9px] uppercase font-bold px-1.5 py-0.5 rounded",
                      node.status === "completed" && "bg-emerald-500/20 text-emerald-300",
                      node.status === "running" && "bg-blue-500/20 text-blue-300",
                      node.status === "idle" && "bg-slate-800 text-slate-400"
                    )}
                  >
                    {node.status}
                  </span>
                </div>
                <h4 className="font-semibold text-xs text-slate-100 leading-snug">{node.name}</h4>
                <p className="text-[10px] text-slate-400 uppercase font-mono mt-1">{node.type}</p>
              </div>
            );
          })}
        </div>

        {/* Right Configuration Inspector */}
        {selectedNode && (
          <div className="w-80 bg-slate-900 border-l border-slate-800 p-5 flex flex-col justify-between overflow-y-auto">
            <div>
              <div className="flex items-center justify-between pb-3 border-b border-slate-800">
                <h3 className="font-semibold text-sm text-slate-200 flex items-center gap-2">
                  <Settings2 className="w-4 h-4 text-teal-400" />
                  Node Configuration
                </h3>
                <span className="text-xs text-slate-500 font-mono">{selectedNode.id}</span>
              </div>

              <div className="space-y-4 mt-4 text-xs">
                <div>
                  <label className="text-slate-400 block mb-1 font-medium">Node Name</label>
                  <input
                    type="text"
                    value={selectedNode.name}
                    onChange={(e) =>
                      setNodes((prev) =>
                        prev.map((n) =>
                          n.id === selectedNode.id ? { ...n, name: e.target.value } : n
                        )
                      )
                    }
                    className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-1.5 text-slate-200 focus:outline-none focus:ring-1 focus:ring-teal-500"
                  />
                </div>

                <div>
                  <label className="text-slate-400 block mb-1 font-medium">Execution Type</label>
                  <input
                    type="text"
                    disabled
                    value={selectedNode.type}
                    className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-1.5 text-slate-400 uppercase font-mono"
                  />
                </div>

                <div>
                  <label className="text-slate-400 block mb-1 font-medium">Retries on Failure</label>
                  <select className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-1.5 text-slate-200">
                    <option value="3">3 Attempts (Exponential Backoff)</option>
                    <option value="1">1 Attempt</option>
                    <option value="0">No Retries</option>
                  </select>
                </div>

                <div>
                  <label className="text-slate-400 block mb-1 font-medium">Execution Timeout</label>
                  <input
                    type="text"
                    defaultValue="1800s"
                    className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-1.5 text-slate-200"
                  />
                </div>
              </div>
            </div>

            <div className="pt-4 border-t border-slate-800 flex gap-2">
              <button className="flex-1 py-2 rounded-lg bg-teal-500/20 text-teal-300 border border-teal-500/30 text-xs font-semibold hover:bg-teal-500/30 transition-all">
                Apply Config
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
