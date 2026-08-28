"use client";

import React from "react";
import { useRouter } from "next/navigation";
import {
  Bell,
  Search,
  User,
  LogOut,
  ChevronDown,
  Shield,
  Layers,
  Sparkles,
} from "lucide-react";
import { useAuthStore, useProjectStore } from "@/stores/authStore";

export function Header() {
  const router = useRouter();
  const { user, logout } = useAuthStore();
  const { currentProject, projects, setCurrentProject, environment, setEnvironment } = useProjectStore();
  const [userDropdownOpen, setUserDropdownOpen] = React.useState(false);

  return (
    <header className="h-16 border-b border-slate-800 bg-slate-900/80 backdrop-blur-md px-6 flex items-center justify-between sticky top-0 z-20">
      {/* Left: Project Selector & Environment */}
      <div className="flex items-center gap-4">
        {/* Project Selector */}
        <div className="relative">
          <select
            value={currentProject?.id || ""}
            onChange={(e) => {
              const selected = projects.find((p) => p.id === e.target.value);
              if (selected) setCurrentProject(selected);
            }}
            className="bg-slate-800 border border-slate-700 text-slate-200 text-sm font-semibold rounded-lg px-3 py-1.5 focus:outline-none focus:ring-2 focus:ring-teal-500 appearance-none pr-8 cursor-pointer shadow-sm"
          >
            {projects.length === 0 && <option value="">Default Project</option>}
            {projects.map((p) => (
              <option key={p.id} value={p.id}>
                {p.name}
              </option>
            ))}
          </select>
          <ChevronDown className="w-4 h-4 text-slate-400 absolute right-2.5 top-2.5 pointer-events-none" />
        </div>

        {/* Environment Badge Selector */}
        <div className="flex items-center bg-slate-950 p-0.5 rounded-lg border border-slate-800 text-xs">
          {(["development", "staging", "production"] as const).map((env) => (
            <button
              key={env}
              onClick={() => setEnvironment(env)}
              className={`px-2.5 py-1 rounded-md capitalize font-medium transition-all ${
                environment === env
                  ? env === "production"
                    ? "bg-emerald-500/20 text-emerald-300 border border-emerald-500/30"
                    : env === "staging"
                    ? "bg-amber-500/20 text-amber-300 border border-amber-500/30"
                    : "bg-blue-500/20 text-blue-300 border border-blue-500/30"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              {env}
            </button>
          ))}
        </div>
      </div>

      {/* Right: Search, Notifications, User Menu */}
      <div className="flex items-center gap-3">
        {/* Global Search Bar */}
        <div className="relative hidden md:block">
          <Search className="w-4 h-4 text-slate-400 absolute left-3 top-2.5" />
          <input
            type="text"
            placeholder="Search models, datasets, runs..."
            className="bg-slate-800 border border-slate-700 rounded-lg pl-9 pr-4 py-1.5 text-xs text-slate-200 placeholder-slate-400 focus:outline-none focus:ring-1 focus:ring-teal-500 w-64"
          />
        </div>

        {/* Notification Bell */}
        <button
          onClick={() => router.push("/alerts")}
          className="relative p-2 rounded-lg text-slate-400 hover:text-slate-200 hover:bg-slate-800 transition-colors"
        >
          <Bell className="w-4 h-4" />
          <span className="absolute top-1.5 right-1.5 w-2 h-2 rounded-full bg-teal-400 ring-2 ring-slate-900"></span>
        </button>

        {/* User Avatar & Menu */}
        <div className="relative">
          <button
            onClick={() => setUserDropdownOpen(!userDropdownOpen)}
            className="flex items-center gap-2.5 p-1.5 rounded-lg hover:bg-slate-800 transition-colors"
          >
            <div className="w-7 h-7 rounded-full bg-gradient-to-br from-teal-400 to-indigo-600 flex items-center justify-center font-bold text-xs text-white shadow">
              {user?.first_name ? user.first_name[0] : "A"}
            </div>
            <div className="text-left hidden lg:block">
              <p className="text-xs font-medium text-slate-200 leading-tight">
                {user?.full_name || "Admin User"}
              </p>
              <p className="text-[10px] text-slate-400 leading-tight">
                {user?.roles?.[0]?.name || "ML Engineer"}
              </p>
            </div>
            <ChevronDown className="w-3.5 h-3.5 text-slate-400 hidden lg:block" />
          </button>

          {userDropdownOpen && (
            <div className="absolute right-0 mt-2 w-52 rounded-xl bg-slate-900 border border-slate-800 shadow-2xl py-1.5 text-xs z-50">
              <div className="px-3 py-2 border-b border-slate-800">
                <p className="font-semibold text-slate-200">{user?.full_name || "Administrator"}</p>
                <p className="text-slate-400 truncate">{user?.email || "admin@modelforge.ai"}</p>
              </div>
              <button
                onClick={() => {
                  setUserDropdownOpen(false);
                  router.push("/settings");
                }}
                className="w-full text-left px-3 py-2 hover:bg-slate-800 text-slate-300 flex items-center gap-2"
              >
                <Shield className="w-3.5 h-3.5 text-teal-400" />
                Organization & RBAC
              </button>
              <button
                onClick={() => {
                  logout();
                  router.push("/login");
                }}
                className="w-full text-left px-3 py-2 hover:bg-rose-500/10 text-rose-400 flex items-center gap-2"
              >
                <LogOut className="w-3.5 h-3.5" />
                Sign Out
              </button>
            </div>
          )}
        </div>
      </div>
    </header>
  );
}
