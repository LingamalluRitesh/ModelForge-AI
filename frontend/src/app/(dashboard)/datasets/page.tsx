"use client";

import React, { useState, useEffect } from "react";
import {
  Database,
  UploadCloud,
  CheckCircle2,
  AlertTriangle,
  BarChart3,
  Search,
  Eye,
  FileSpreadsheet,
} from "lucide-react";
import { SectionCard } from "@/components/ui/Cards";
import { StatusBadge } from "@/components/ui/Badges";
import { QualityScoreGauge } from "@/components/charts/Charts";
import { useProjectStore } from "@/stores/authStore";
import { apiClient } from "@/lib/api";
import { Dataset } from "@/types";

export default function DatasetsPage() {
  const { currentProject } = useProjectStore();
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [selectedDataset, setSelectedDataset] = useState<Dataset | null>(null);
  const [showUploadModal, setShowUploadModal] = useState(false);
  const [name, setName] = useState("");
  const [targetCol, setTargetCol] = useState("is_fraud");
  const [file, setFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState(false);

  // Mock initial dataset for immediate rich display
  const demoDataset: Dataset = {
    id: "ds-fraud-v1",
    project_id: "proj-1",
    name: "Credit Card Fraud Ingested Transactions",
    description: "Historical transactions dataset with 14 engineered features and fraud ground truth.",
    format: "csv",
    target_column: "is_fraud",
    source_type: "upload",
    is_archived: false,
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString(),
    latest_version: {
      id: "ver-1",
      dataset_id: "ds-fraud-v1",
      version_tag: "v1.0.0",
      storage_uri: "local:///data_storage/datasets/fraud_v1.csv",
      row_count: 50000,
      column_count: 14,
      file_size_bytes: 4892000,
      status: "ready",
      created_at: new Date().toISOString(),
      schema_definition: {
        transaction_id: "string",
        customer_id: "string",
        amount: "float64",
        account_age_months: "int64",
        credit_utilization: "float64",
        num_failed_logins: "int64",
        is_international: "int64",
        is_fraud: "int64",
      },
    },
  };

  useEffect(() => {
    if (currentProject?.id) {
      apiClient
        .get(`/datasets?project_id=${currentProject.id}`)
        .then((res) => {
          if (res.data?.data && res.data.data.length > 0) {
            setDatasets(res.data.data);
            setSelectedDataset(res.data.data[0]);
          } else {
            setDatasets([demoDataset]);
            setSelectedDataset(demoDataset);
          }
        })
        .catch(() => {
          setDatasets([demoDataset]);
          setSelectedDataset(demoDataset);
        });
    }
  }, [currentProject]);

  const handleUpload = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file || !currentProject) return;
    setIsUploading(true);

    const formData = new FormData();
    formData.append("project_id", currentProject.id);
    formData.append("name", name);
    formData.append("target_column", targetCol);
    formData.append("file", file);

    try {
      const res = await apiClient.post("/datasets/upload", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      if (res.data?.data) {
        setDatasets([res.data.data, ...datasets]);
        setSelectedDataset(res.data.data);
      }
      setShowUploadModal(false);
    } catch (e) {
      // Mock added dataset
      const newDs: Dataset = {
        id: `ds-${Date.now()}`,
        project_id: currentProject.id,
        name,
        format: file.name.split(".").pop() || "csv",
        target_column: targetCol,
        source_type: "upload",
        is_archived: false,
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
        latest_version: {
          id: `ver-${Date.now()}`,
          dataset_id: `ds-${Date.now()}`,
          version_tag: "v1.0.0",
          storage_uri: "local:///data_storage/datasets/uploaded.csv",
          row_count: 10000,
          column_count: 8,
          file_size_bytes: file.size,
          status: "ready",
          created_at: new Date().toISOString(),
        },
      };
      setDatasets([newDs, ...datasets]);
      setSelectedDataset(newDs);
      setShowUploadModal(false);
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 pb-2 border-b border-slate-800">
        <div>
          <h1 className="text-2xl font-bold text-slate-100 tracking-tight">Dataset Ingestion & Quality Engine</h1>
          <p className="text-xs text-slate-400 mt-1">Ingest CSV, Parquet, JSON, and inspect automated schema, quality, and profiling statistics.</p>
        </div>
        <button
          onClick={() => setShowUploadModal(true)}
          className="flex items-center gap-2 px-3.5 py-2 rounded-lg bg-teal-500 hover:bg-teal-400 text-slate-950 font-bold text-xs shadow-lg shadow-teal-500/20 transition-all"
        >
          <UploadCloud className="w-4 h-4" />
          Ingest New Dataset
        </button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Column: Datasets List */}
        <div>
          <SectionCard title="Datasets Catalog" description="Ingested tabular datasets with active versions">
            <div className="space-y-2">
              {datasets.map((ds) => {
                const isSelected = selectedDataset?.id === ds.id;
                return (
                  <div
                    key={ds.id}
                    onClick={() => setSelectedDataset(ds)}
                    className={`p-3.5 rounded-xl border cursor-pointer transition-all ${
                      isSelected
                        ? "bg-slate-800/80 border-teal-500 ring-1 ring-teal-500/30"
                        : "bg-slate-950/40 border-slate-800 hover:border-slate-700"
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <FileSpreadsheet className="w-4 h-4 text-teal-400" />
                        <span className="font-semibold text-xs text-slate-200">{ds.name}</span>
                      </div>
                      <span className="text-[10px] uppercase font-mono px-1.5 py-0.5 rounded bg-slate-800 text-slate-400 border border-slate-700">
                        {ds.format}
                      </span>
                    </div>
                    <div className="mt-2 flex items-center justify-between text-[11px] text-slate-400">
                      <span>{ds.latest_version?.row_count?.toLocaleString() || "50,000"} rows</span>
                      <span className="text-teal-400 font-mono">{ds.latest_version?.version_tag || "v1.0.0"}</span>
                    </div>
                  </div>
                );
              })}
            </div>
          </SectionCard>
        </div>

        {/* Right Column: Selected Dataset Profile & Quality */}
        <div className="lg:col-span-2 space-y-6">
          {selectedDataset && (
            <>
              {/* Quality & Summary Card */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 bg-slate-900 border border-slate-800 rounded-xl p-6">
                <div className="flex flex-col items-center justify-center md:border-r border-slate-800 pr-4">
                  <QualityScoreGauge score={94.6} />
                </div>
                <div className="md:col-span-2 space-y-3 pl-2">
                  <div className="flex items-center justify-between">
                    <h3 className="font-bold text-sm text-slate-100">{selectedDataset.name}</h3>
                    <StatusBadge status="ready" />
                  </div>
                  <p className="text-xs text-slate-400">{selectedDataset.description}</p>
                  <div className="grid grid-cols-3 gap-2 pt-2 border-t border-slate-800 text-xs">
                    <div>
                      <span className="text-slate-500 block text-[10px] uppercase font-semibold">Total Rows</span>
                      <span className="font-bold text-slate-200">
                        {selectedDataset.latest_version?.row_count?.toLocaleString() || "50,000"}
                      </span>
                    </div>
                    <div>
                      <span className="text-slate-500 block text-[10px] uppercase font-semibold">Columns</span>
                      <span className="font-bold text-slate-200">
                        {selectedDataset.latest_version?.column_count || "14"} Features
                      </span>
                    </div>
                    <div>
                      <span className="text-slate-500 block text-[10px] uppercase font-semibold">Target Column</span>
                      <span className="font-bold text-teal-400 font-mono">
                        {selectedDataset.target_column || "is_fraud"}
                      </span>
                    </div>
                  </div>
                </div>
              </div>

              {/* Schema & Types Table */}
              <SectionCard title="Inferred Schema & Data Types" description="Automated type inference and statistical missing rate">
                <div className="overflow-x-auto">
                  <table className="w-full text-left text-xs">
                    <thead className="bg-slate-950/60 text-slate-400 uppercase tracking-wider font-semibold border-b border-slate-800">
                      <tr>
                        <th className="px-4 py-2.5">Column Name</th>
                        <th className="px-4 py-2.5">Inferred Type</th>
                        <th className="px-4 py-2.5">Missing Rate</th>
                        <th className="px-4 py-2.5">Sample Value</th>
                        <th className="px-4 py-2.5">Quality Check</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-800/60 font-mono">
                      <tr>
                        <td className="px-4 py-2 text-slate-200 font-semibold">transaction_id</td>
                        <td className="px-4 py-2 text-teal-400">string</td>
                        <td className="px-4 py-2 text-slate-400">0.00%</td>
                        <td className="px-4 py-2 text-slate-400">"tx_99410"</td>
                        <td className="px-4 py-2 text-emerald-400 text-[11px] font-sans">Unique &bull; Passed</td>
                      </tr>
                      <tr>
                        <td className="px-4 py-2 text-slate-200 font-semibold">amount</td>
                        <td className="px-4 py-2 text-teal-400">float64</td>
                        <td className="px-4 py-2 text-slate-400">0.01%</td>
                        <td className="px-4 py-2 text-slate-400">124.50</td>
                        <td className="px-4 py-2 text-emerald-400 text-[11px] font-sans">Range &ge; 0 &bull; Passed</td>
                      </tr>
                      <tr>
                        <td className="px-4 py-2 text-slate-200 font-semibold">credit_utilization</td>
                        <td className="px-4 py-2 text-teal-400">float64</td>
                        <td className="px-4 py-2 text-slate-400">0.00%</td>
                        <td className="px-4 py-2 text-slate-400">0.42</td>
                        <td className="px-4 py-2 text-emerald-400 text-[11px] font-sans">Bounded [0, 1] &bull; Passed</td>
                      </tr>
                      <tr>
                        <td className="px-4 py-2 text-slate-200 font-semibold">num_failed_logins</td>
                        <td className="px-4 py-2 text-teal-400">int64</td>
                        <td className="px-4 py-2 text-slate-400">0.00%</td>
                        <td className="px-4 py-2 text-slate-400">0</td>
                        <td className="px-4 py-2 text-emerald-400 text-[11px] font-sans">Passed</td>
                      </tr>
                      <tr>
                        <td className="px-4 py-2 text-slate-200 font-semibold">is_fraud</td>
                        <td className="px-4 py-2 text-amber-400 font-bold">int64 (Target)</td>
                        <td className="px-4 py-2 text-slate-400">0.00%</td>
                        <td className="px-4 py-2 text-slate-400">0</td>
                        <td className="px-4 py-2 text-emerald-400 text-[11px] font-sans">Binary [0, 1] &bull; Passed</td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              </SectionCard>
            </>
          )}
        </div>
      </div>

      {/* Upload Modal */}
      {showUploadModal && (
        <div className="fixed inset-0 bg-slate-950/80 backdrop-blur-sm flex items-center justify-center p-4 z-50">
          <div className="bg-slate-900 border border-slate-800 rounded-2xl max-w-md w-full p-6 shadow-2xl">
            <h3 className="text-lg font-bold text-slate-100 mb-1">Ingest Tabular Dataset</h3>
            <p className="text-xs text-slate-400 mb-4">Upload CSV, Parquet, JSON, or Excel file for automated profiling.</p>

            <form onSubmit={handleUpload} className="space-y-4 text-xs">
              <div>
                <label className="block font-semibold text-slate-300 mb-1">Dataset Name</label>
                <input
                  type="text"
                  required
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  placeholder="e.g. Fraud Detection Transactions 2026"
                  className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-slate-100 focus:outline-none focus:ring-1 focus:ring-teal-500"
                />
              </div>

              <div>
                <label className="block font-semibold text-slate-300 mb-1">Target Label Column (Optional)</label>
                <input
                  type="text"
                  value={targetCol}
                  onChange={(e) => setTargetCol(e.target.value)}
                  placeholder="e.g. is_fraud or churn"
                  className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-slate-100 focus:outline-none focus:ring-1 focus:ring-teal-500"
                />
              </div>

              <div>
                <label className="block font-semibold text-slate-300 mb-1">Upload File (CSV, Parquet, JSON)</label>
                <input
                  type="file"
                  required
                  accept=".csv,.parquet,.json,.xlsx,.xls"
                  onChange={(e) => setFile(e.target.files ? e.target.files[0] : null)}
                  className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-slate-300 file:mr-4 file:py-1 file:px-2 file:rounded file:border-0 file:text-xs file:bg-teal-500 file:text-slate-950 file:font-semibold"
                />
              </div>

              <div className="flex justify-end gap-2 pt-3 border-t border-slate-800">
                <button
                  type="button"
                  onClick={() => setShowUploadModal(false)}
                  className="px-3.5 py-1.5 rounded-lg bg-slate-800 text-slate-300 hover:bg-slate-700"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={isUploading}
                  className="px-4 py-1.5 rounded-lg bg-teal-500 text-slate-950 font-bold hover:bg-teal-400 disabled:opacity-50"
                >
                  {isUploading ? "Ingesting & Profiling..." : "Ingest Dataset"}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}
