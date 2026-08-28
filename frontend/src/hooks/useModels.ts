"use client";

import { useState, useEffect, useCallback } from "react";
import { RegisteredModel } from "@/types";
import { modelRegistryApi } from "@/lib/api";

export function useModels(projectId?: string) {
  const [models, setModels] = useState<RegisteredModel[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchModels = useCallback(async () => {
    if (!projectId) return;
    setLoading(true);
    setError(null);
    try {
      const res = await modelRegistryApi.list(projectId);
      setModels(res.data.data);
    } catch (err: any) {
      setError(err.response?.data?.message || err.message || "Failed to fetch registered models");
    } finally {
      setLoading(false);
    }
  }, [projectId]);

  useEffect(() => {
    fetchModels();
  }, [fetchModels]);

  const updateStage = async (versionId: string, stage: string) => {
    const res = await modelRegistryApi.updateStage(versionId, stage);
    await fetchModels();
    return res.data.data;
  };

  return {
    models,
    loading,
    error,
    refetch: fetchModels,
    updateStage,
  };
}
