"use client";

import React, { useState, useEffect } from "react";
import { FolderGit2, Plus, Search, Layers, Cpu, Archive, ExternalLink } from "lucide-react";
import { SectionCard } from "@/components/ui/Cards";
import { StageBadge } from "@/components/ui/Badges";
import { useProjectStore } from "@/stores/authStore";
import { apiClient } from "@/lib/api";
import { Project } from "@/types";

export default function ProjectsPage() {
  const { projects, setProjects, setCurrentProject } = useProjectStore();
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [newProjectName, setNewProjectName] = useState("");
  const [problemType, setProblemType] = useState("classification");
  const [description, setDescription] = useState("");
  const [search, setSearch] = useState("");

  const handleCreate = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      const res = await apiClient.post("/projects", {
        name: newProjectName,
        problem_type: problemType,
        description,
        tags: ["production-ready"],
      });
      if (res.data?.data) {
        setProjects([...projects, res.data.data]);
        setCurrentProject(res.data.data);
      }
      setShowCreateModal(false);
      setNewProjectName("");
      setDescription("");
    } catch (e) {
      // Fallback
      const demoProj: Project = {
        id: `proj-${Date.now()}`,
        organization_id: "org-1",
        name: newProjectName,
        slug: newProjectName.toLowerCase().replace(/\s+/g, "-"),
        problem_type: problemType as any,
        description,
        is_archived: false,
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
      };
      setProjects([...projects, demoProj]);
      setShowCreateModal(false);
    }
  };

  const filteredProjects = projects.filter((p) =>
    p.name.toLowerCase().includes(search.toLowerCase())
  );

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 pb-2 border-b border-slate-800">
        <div>
          <h1 className="text-2xl font-bold text-slate-100 tracking-tight">Machine Learning Projects</h1>
          <p className="text-xs text-slate-400 mt-1">Manage isolated workspaces for ML models, datasets, and pipelines.</p>
        </div>
        <button
          onClick={() => setShowCreateModal(true)}
          className="flex items-center gap-2 px-3.5 py-2 rounded-lg bg-teal-500 hover:bg-teal-400 text-slate-950 font-bold text-xs shadow-lg shadow-teal-500/20 transition-all"
        >
          <Plus className="w-4 h-4" />
          Create New Project
        </button>
      </div>

      {/* Search Filter */}
      <div className="relative max-w-md">
        <Search className="w-4 h-4 text-slate-400 absolute left-3 top-3 pointer-events-none" />
        <input
          type="text"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="Filter projects by title..."
          className="w-full bg-slate-900 border border-slate-800 rounded-lg pl-9 pr-3 py-2 text-xs text-slate-200 focus:outline-none focus:ring-1 focus:ring-teal-500"
        />
      </div>

      {/* Projects Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {filteredProjects.map((p) => (
          <div
            key={p.id}
            onClick={() => setCurrentProject(p)}
            className="bg-slate-900 border border-slate-800 rounded-xl p-6 hover:border-teal-500/40 hover:shadow-xl hover:shadow-teal-500/5 transition-all cursor-pointer group flex flex-col justify-between"
          >
            <div>
              <div className="flex items-center justify-between mb-3">
                <div className="p-2 rounded-lg bg-slate-800 border border-slate-700 text-teal-400">
                  <FolderGit2 className="w-5 h-5" />
                </div>
                <span className="text-[10px] uppercase font-bold px-2 py-0.5 rounded bg-teal-500/15 text-teal-300 border border-teal-500/30">
                  {p.problem_type}
                </span>
              </div>
              <h3 className="font-bold text-slate-100 text-base group-hover:text-teal-300 transition-colors">
                {p.name}
              </h3>
              <p className="text-xs text-slate-400 mt-2 line-clamp-2">
                {p.description || "Production ML lifecycle workspace for model training, validation, and real-time monitoring."}
              </p>
            </div>

            <div className="pt-4 mt-6 border-t border-slate-800/80 flex items-center justify-between text-xs text-slate-500">
              <span className="font-mono text-[11px]">{new Date(p.created_at).toLocaleDateString()}</span>
              <span className="text-teal-400 font-semibold group-hover:underline flex items-center gap-1">
                Open Workspace &rarr;
              </span>
            </div>
          </div>
        ))}
      </div>

      {/* Modal */}
      {showCreateModal && (
        <div className="fixed inset-0 bg-slate-950/80 backdrop-blur-sm flex items-center justify-center p-4 z-50">
          <div className="bg-slate-900 border border-slate-800 rounded-2xl max-w-md w-full p-6 shadow-2xl">
            <h3 className="text-lg font-bold text-slate-100 mb-1">Create Machine Learning Project</h3>
            <p className="text-xs text-slate-400 mb-4">Set project workspace name and problem domain.</p>

            <form onSubmit={handleCreate} className="space-y-4 text-xs">
              <div>
                <label className="block font-semibold text-slate-300 mb-1">Project Name</label>
                <input
                  type="text"
                  required
                  value={newProjectName}
                  onChange={(e) => setNewProjectName(e.target.value)}
                  placeholder="e.g. Credit Risk Prediction"
                  className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-slate-100 focus:outline-none focus:ring-1 focus:ring-teal-500"
                />
              </div>

              <div>
                <label className="block font-semibold text-slate-300 mb-1">Problem Domain</label>
                <select
                  value={problemType}
                  onChange={(e) => setProblemType(e.target.value)}
                  className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-slate-100"
                >
                  <option value="classification">Classification (Binary & Multiclass)</option>
                  <option value="regression">Regression (Continuous Numerical)</option>
                  <option value="clustering">Clustering & Segmentation (Unsupervised)</option>
                  <option value="deep_learning">Deep Learning (Neural Network Tabular)</option>
                </select>
              </div>

              <div>
                <label className="block font-semibold text-slate-300 mb-1">Description</label>
                <textarea
                  rows={3}
                  value={description}
                  onChange={(e) => setDescription(e.target.value)}
                  placeholder="Brief summary of business objective..."
                  className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-slate-100"
                />
              </div>

              <div className="flex justify-end gap-2 pt-3 border-t border-slate-800">
                <button
                  type="button"
                  onClick={() => setShowCreateModal(false)}
                  className="px-3.5 py-1.5 rounded-lg bg-slate-800 text-slate-300 hover:bg-slate-700"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="px-4 py-1.5 rounded-lg bg-teal-500 text-slate-950 font-bold hover:bg-teal-400"
                >
                  Create Project
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}
