"use client";

import { useState, useEffect, useCallback } from "react";
import { Alert } from "@/types";
import { alertApi } from "@/lib/api";

export function useAlerts(projectId?: string) {
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchAlerts = useCallback(async () => {
    if (!projectId) return;
    setLoading(true);
    setError(null);
    try {
      const res = await alertApi.list(projectId);
      setAlerts(res.data.data);
    } catch (err: any) {
      setError(err.response?.data?.message || err.message || "Failed to fetch alerts");
    } finally {
      setLoading(false);
    }
  }, [projectId]);

  useEffect(() => {
    fetchAlerts();
  }, [fetchAlerts]);

  const acknowledgeAlert = async (alertId: string) => {
    const res = await alertApi.acknowledge(alertId);
    await fetchAlerts();
    return res.data.data;
  };

  return {
    alerts,
    loading,
    error,
    refetch: fetchAlerts,
    acknowledgeAlert,
  };
}
