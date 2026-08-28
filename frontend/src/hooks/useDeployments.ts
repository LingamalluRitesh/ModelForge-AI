"use client";

import { useState, useEffect, useCallback } from "react";
import { Deployment } from "@/types";
import { deploymentApi } from "@/lib/api";

export function useDeployments(projectId?: string) {
  const [deployments, setDeployments] = useState<Deployment[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchDeployments = useCallback(async () => {
    if (!projectId) return;
    setLoading(true);
    setError(null);
    try {
      const res = await deploymentApi.list(projectId);
      setDeployments(res.data.data);
    } catch (err: any) {
      setError(err.response?.data?.message || err.message || "Failed to fetch deployments");
    } finally {
      setLoading(false);
    }
  }, [projectId]);

  useEffect(() => {
    fetchDeployments();
  }, [fetchDeployments]);

  const updateTrafficSplit = async (deploymentId: string, canarySplit: number) => {
    const res = await deploymentApi.updateTrafficSplit(deploymentId, canarySplit);
    await fetchDeployments();
    return res.data.data;
  };

  const rollback = async (deploymentId: string) => {
    const res = await deploymentApi.rollback(deploymentId);
    await fetchDeployments();
    return res.data.data;
  };

  return {
    deployments,
    loading,
    error,
    refetch: fetchDeployments,
    updateTrafficSplit,
    rollback,
  };
}
