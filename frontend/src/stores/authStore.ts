/**
 * ModelForge AI - Zustand State Stores
 */

import { create } from "zustand";
import { User, Project, Organization } from "@/types";
import { apiClient } from "@/lib/api";

interface AuthState {
  user: User | null;
  organizationId: string | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  setUser: (user: User | null, orgId?: string | null) => void;
  logout: () => void;
  fetchProfile: () => Promise<void>;
}

export const useAuthStore = create<AuthState>((set) => ({
  user: null,
  organizationId: null,
  isAuthenticated: false,
  isLoading: true,

  setUser: (user, orgId) => {
    set({
      user,
      organizationId: orgId || null,
      isAuthenticated: !!user,
      isLoading: false,
    });
  },

  logout: () => {
    if (typeof window !== "undefined") {
      localStorage.removeItem("mf_access_token");
      localStorage.removeItem("mf_refresh_token");
    }
    set({ user: null, organizationId: null, isAuthenticated: false, isLoading: false });
  },

  fetchProfile: async () => {
    try {
      set({ isLoading: true });
      const res = await apiClient.get("/auth/me");
      const user = res.data.data;
      set({ user, isAuthenticated: true, isLoading: false });
    } catch (e) {
      set({ user: null, isAuthenticated: false, isLoading: false });
    }
  },
}));

interface ProjectState {
  currentProject: Project | null;
  projects: Project[];
  environment: "development" | "staging" | "production";
  setCurrentProject: (project: Project | null) => void;
  setProjects: (projects: Project[]) => void;
  setEnvironment: (env: "development" | "staging" | "production") => void;
}

export const useProjectStore = create<ProjectState>((set) => ({
  currentProject: null,
  projects: [],
  environment: "production",

  setCurrentProject: (currentProject) => set({ currentProject }),
  setProjects: (projects) => {
    set((state) => ({
      projects,
      currentProject: state.currentProject || (projects.length > 0 ? projects[0] : null),
    }));
  },
  setEnvironment: (environment) => set({ environment }),
}));

interface ThemeState {
  isDarkMode: boolean;
  toggleDarkMode: () => void;
}

export const useThemeStore = create<ThemeState>((set) => ({
  isDarkMode: true,
  toggleDarkMode: () => set((state) => ({ isDarkMode: !state.isDarkMode })),
}));
