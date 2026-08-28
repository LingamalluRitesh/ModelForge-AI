"use client";

import React from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  LayoutDashboard,
  FolderGit2,
  Database,
  Layers,
  FlaskConical,
  Cpu,
  Sparkles,
  Archive,
  Rocket,
  Activity,
  GitCompare,
  TrendingDown,
  RefreshCw,
  Eye,
  GitMerge,
  Bell,
  ShieldCheck,
  Settings,
  ChevronLeft,
  ChevronRight,
  Flame,
} from "lucide-react";
import { clsx } from "clsx";

interface NavItem {
  label: string;
  href: string;
  icon: React.ElementType;
  badge?: string;
}

const navItems: NavItem[] = [
  { label: "Overview", href: "/dashboard", icon: LayoutDashboard },
  { label: "Projects", href: "/projects", icon: FolderGit2 },
  { label: "Datasets & Quality", href: "/datasets", icon: Database },
  { label: "Feature Store", href: "/features", icon: Layers },
  { label: "Experiments", href: "/experiments", icon: FlaskConical },
  { label: "Model Training", href: "/training", icon: Cpu },
  { label: "AutoML Studio", href: "/automl", icon: Sparkles, badge: "Auto" },
  { label: "Model Registry", href: "/registry", icon: Archive },
  { label: "Deployments", href: "/deployments", icon: Rocket },
  { label: "Inference & Batch", href: "/predictions", icon: Activity },
  { label: "Live Monitoring", href: "/monitoring", icon: GitCompare },
  { label: "Drift Detection", href: "/drift", icon: TrendingDown, badge: "Alert" },
  { label: "Auto Retraining", href: "/retraining", icon: RefreshCw },
  { label: "Explainability & Bias", href: "/explainability", icon: Eye },
  { label: "Visual Pipelines", href: "/pipelines", icon: GitMerge },
  { label: "Alerts & Webhooks", href: "/alerts", icon: Bell },
  { label: "Audit Logs", href: "/audit", icon: ShieldCheck },
  { label: "Settings", href: "/settings", icon: Settings },
];

export function Sidebar() {
  const pathname = usePathname();
  const [collapsed, setCollapsed] = React.useState(false);

  return (
    <aside
      className={clsx(
        "h-screen bg-slate-900 border-r border-slate-800 transition-all duration-300 flex flex-col justify-between z-30 sticky top-0",
        collapsed ? "w-16" : "w-64"
      )}
    >
      {/* Brand Header */}
      <div className="flex items-center justify-between p-4 border-b border-slate-800 h-16">
        <Link href="/dashboard" className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-lg bg-gradient-to-tr from-teal-500 to-emerald-400 flex items-center justify-center shadow-lg shadow-teal-500/20">
            <Flame className="w-5 h-5 text-slate-950 fill-current" />
          </div>
          {!collapsed && (
            <div>
              <span className="font-bold text-lg tracking-tight bg-gradient-to-r from-teal-300 via-emerald-200 to-white bg-clip-text text-transparent">
                ModelForge
              </span>
              <span className="text-xs font-semibold uppercase tracking-wider text-teal-400 ml-1.5 px-1.5 py-0.5 rounded bg-teal-950 border border-teal-800/60">
                AI
              </span>
            </div>
          )}
        </Link>
        <button
          onClick={() => setCollapsed(!collapsed)}
          className="text-slate-400 hover:text-slate-200 p-1 rounded-md hover:bg-slate-800 hidden md:block"
        >
          {collapsed ? <ChevronRight className="w-4 h-4" /> : <ChevronLeft className="w-4 h-4" />}
        </button>
      </div>

      {/* Navigation List */}
      <nav className="flex-1 overflow-y-auto px-2 py-4 space-y-1">
        {navItems.map((item) => {
          const isActive = pathname === item.href || pathname.startsWith(`${item.href}/`);
          const Icon = item.icon;
          return (
            <Link
              key={item.href}
              href={item.href}
              className={clsx(
                "flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-all group relative",
                isActive
                  ? "bg-teal-500/10 text-teal-300 border border-teal-500/30"
                  : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/60"
              )}
            >
              <Icon
                className={clsx(
                  "w-4 h-4 shrink-0 transition-colors",
                  isActive ? "text-teal-400" : "text-slate-400 group-hover:text-slate-200"
                )}
              />
              {!collapsed && <span className="truncate">{item.label}</span>}
              {!collapsed && item.badge && (
                <span className="ml-auto text-[10px] uppercase font-bold tracking-wider px-1.5 py-0.5 rounded bg-teal-500/20 text-teal-300 border border-teal-500/30">
                  {item.badge}
                </span>
              )}
            </Link>
          );
        })}
      </nav>

      {/* Footer Info */}
      {!collapsed && (
        <div className="p-3 border-t border-slate-800 m-2 rounded-lg bg-slate-950/60 border text-xs text-slate-400">
          <div className="flex items-center justify-between">
            <span className="flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
              Platform Ready
            </span>
            <span className="text-slate-500 font-mono text-[10px]">v1.0.0</span>
          </div>
        </div>
      )}
    </aside>
  );
}
