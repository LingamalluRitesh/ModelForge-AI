"use client";

import { useState, useEffect, useCallback } from "react";
import { MLPipeline } from "@/types";
import { pipelineApi } from "@/lib/api";

export function usePipelines(projectId?: string) {
  const [pipelines, setPipelines] = useState<MLPipeline[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchPipelines = useCallback(async () => {
    if (!projectId) return;
    setLoading(true);
    setError(null);
    try {
      const res = await pipelineApi.list(projectId);
      setPipelines(res.data.data);
    } catch (err: any) {
      setError(err.response?.data?.message || err.message || "Failed to fetch pipelines");
    } finally {
      setLoading(false);
    }
  }, [projectId]);

  useEffect(() => {
    fetchPipelines();
  }, [fetchPipelines]);

  const triggerPipeline = async (pipelineId: string) => {
    const res = await pipelineApi.trigger(pipelineId);
    await fetchPipelines();
    return res.data.data;
  };

  return {
    pipelines,
    loading,
    error,
    refetch: fetchPipelines,
    triggerPipeline,
  };
}
