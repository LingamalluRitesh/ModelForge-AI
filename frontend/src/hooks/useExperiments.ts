"use client";

import { useState, useEffect, useCallback } from "react";
import { Experiment, ExperimentRun } from "@/types";
import { experimentApi, trainingApi } from "@/lib/api";

export function useExperiments(projectId?: string) {
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [runs, setRuns] = useState<ExperimentRun[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchExperiments = useCallback(async () => {
    if (!projectId) return;
    setLoading(true);
    setError(null);
    try {
      const res = await experimentApi.list(projectId);
      setExperiments(res.data.data);
      const allRuns = res.data.data.flatMap((e: Experiment) => e.runs || []);
      setRuns(allRuns);
    } catch (err: any) {
      setError(err.response?.data?.message || err.message || "Failed to fetch experiments");
    } finally {
      setLoading(false);
    }
  }, [projectId]);

  useEffect(() => {
    fetchExperiments();
  }, [fetchExperiments]);

  const launchTrainingJob = async (config: any) => {
    if (!projectId) return;
    const res = await trainingApi.launchTraining(projectId, config);
    await fetchExperiments();
    return res.data.data;
  };

  return {
    experiments,
    runs,
    loading,
    error,
    refetch: fetchExperiments,
    launchTrainingJob,
  };
}
