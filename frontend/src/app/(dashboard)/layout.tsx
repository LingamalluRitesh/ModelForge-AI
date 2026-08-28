"use client";

import React, { useEffect } from "react";
import { Sidebar } from "@/components/layout/Sidebar";
import { Header } from "@/components/layout/Header";
import { useAuthStore, useProjectStore } from "@/stores/authStore";
import { apiClient } from "@/lib/api";

export default function DashboardLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const { fetchProfile, user } = useAuthStore();
  const { setProjects, projects, currentProject, setCurrentProject } = useProjectStore();

  useEffect(() => {
    fetchProfile();
    // Load projects
    apiClient
      .get("/projects")
      .then((res) => {
        if (res.data?.data && res.data.data.length > 0) {
          setProjects(res.data.data);
        } else {
          // Set demo fallback project
          const defaultProject = {
            id: "proj-fraud-01",
            organization_id: "org-01",
            name: "Enterprise Fraud Detection",
            slug: "fraud-detection",
            problem_type: "classification" as const,
            is_archived: false,
            created_at: new Date().toISOString(),
            updated_at: new Date().toISOString(),
          };
          setProjects([defaultProject]);
          setCurrentProject(defaultProject);
        }
      })
      .catch(() => {
        const defaultProject = {
          id: "proj-fraud-01",
          organization_id: "org-01",
          name: "Enterprise Fraud Detection",
          slug: "fraud-detection",
          problem_type: "classification" as const,
          is_archived: false,
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString(),
        };
        setProjects([defaultProject]);
        setCurrentProject(defaultProject);
      });
  }, [fetchProfile, setProjects, setCurrentProject]);

  return (
    <div className="flex min-h-screen bg-slate-950 text-slate-100">
      <Sidebar />
      <div className="flex-1 flex flex-col min-w-0 overflow-hidden">
        <Header />
        <main className="flex-1 overflow-y-auto p-6 md:p-8 space-y-8 max-w-7xl w-full mx-auto">
          {children}
        </main>
      </div>
    </div>
  );
}
