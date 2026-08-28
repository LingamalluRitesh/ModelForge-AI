"use client";

import { useState, useEffect, useCallback } from "react";
import { monitoringApi } from "@/lib/api";

export function useMonitoring(deploymentId?: string) {
  const [metrics, setMetrics] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchMetrics = useCallback(async () => {
    if (!deploymentId) return;
    setLoading(true);
    setError(null);
    try {
      const res = await monitoringApi.getDeploymentMetrics(deploymentId);
      setMetrics(res.data.data);
    } catch (err: any) {
      setError(err.response?.data?.message || err.message || "Failed to fetch monitoring metrics");
    } finally {
      setLoading(false);
    }
  }, [deploymentId]);

  useEffect(() => {
    fetchMetrics();
    const interval = setInterval(fetchMetrics, 30000); // 30s auto-refresh
    return () => clearInterval(interval);
  }, [fetchMetrics]);

  return {
    metrics,
    loading,
    error,
    refetch: fetchMetrics,
  };
}
