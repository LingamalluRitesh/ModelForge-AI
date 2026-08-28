"use client";

import { useState, useEffect, useCallback } from "react";
import { Dataset } from "@/types";
import { datasetApi } from "@/lib/api";

export function useDatasets(projectId?: string) {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchDatasets = useCallback(async () => {
    if (!projectId) return;
    setLoading(true);
    setError(null);
    try {
      const res = await datasetApi.list(projectId);
      setDatasets(res.data.data);
    } catch (err: any) {
      setError(err.response?.data?.message || err.message || "Failed to fetch datasets");
    } finally {
      setLoading(false);
    }
  }, [projectId]);

  useEffect(() => {
    fetchDatasets();
  }, [fetchDatasets]);

  const uploadDataset = async (formData: FormData) => {
    if (!projectId) return;
    const res = await datasetApi.upload(projectId, formData);
    await fetchDatasets();
    return res.data.data;
  };

  return {
    datasets,
    loading,
    error,
    refetch: fetchDatasets,
    uploadDataset,
  };
}
