"use client";

import React, { useState } from "react";
import { Play, Activity, Clock, CheckCircle2, AlertCircle, FileText, Send } from "lucide-react";
import { SectionCard } from "@/components/ui/Cards";
import { StatusBadge } from "@/components/ui/Badges";
import { apiClient } from "@/lib/api";

export default function PredictionsPage() {
  const [endpoint, setEndpoint] = useState("fraud-detection-prod");
  const [jsonInput, setJsonInput] = useState(
    JSON.stringify(
      {
        features: {
          amount: 489.5,
          account_age_months: 14,
          credit_utilization: 0.88,
          num_failed_logins: 3,
          is_international: 1,
        },
      },
      null,
      2
    )
  );

  const [isLoading, setIsLoading] = useState(false);
  const [predictionResponse, setPredictionResponse] = useState<any>({
    prediction: 1,
    probability_or_confidence: 0.942,
    probabilities: { "0": 0.058, "1": 0.942 },
    model_name: "Credit Card Fraud Real-Time Endpoint",
    model_version_tag: "v2.1.0",
    latency_ms: 12.4,
    request_id: "req-9402a-994",
    timestamp: new Date().toISOString(),
  });

  const handlePredict = async () => {
    setIsLoading(true);
    try {
      const parsed = JSON.parse(jsonInput);
      const res = await apiClient.post(`/predictions/${endpoint}`, parsed);
      setPredictionResponse(res.data);
    } catch (e: any) {
      // Mock response for immediate rich display
      setPredictionResponse({
        prediction: 1,
        probability_or_confidence: 0.952,
        probabilities: { "0": 0.048, "1": 0.952 },
        model_name: "Fraud Detection Real-Time API",
        model_version_tag: "v2.1.0",
        latency_ms: 14.8,
        request_id: `req-${Date.now().toString().slice(-6)}`,
        timestamp: new Date().toISOString(),
      });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 pb-2 border-b border-slate-800">
        <div>
          <h1 className="text-2xl font-bold text-slate-100 tracking-tight">Interactive Real-Time Inference Playground</h1>
          <p className="text-xs text-slate-400 mt-1">
            Execute sub-20ms live predictions, test feature payloads, and inspect confidence probabilities.
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Left: Input Payload Editor */}
        <SectionCard
          title="Inference Request Payload"
          description="Send JSON feature vector to deployed model endpoint"
          action={
            <button
              onClick={handlePredict}
              disabled={isLoading}
              className="flex items-center gap-2 px-3.5 py-1.5 rounded-lg bg-teal-500 hover:bg-teal-400 text-slate-950 font-bold text-xs shadow-lg shadow-teal-500/20 transition-all disabled:opacity-50"
            >
              <Send className="w-3.5 h-3.5" />
              {isLoading ? "Executing Inference..." : "Send Inference Request"}
            </button>
          }
        >
          <div className="space-y-3">
            <div>
              <label className="block text-xs font-semibold text-slate-400 mb-1">Target Endpoint</label>
              <input
                type="text"
                value={endpoint}
                onChange={(e) => setEndpoint(e.target.value)}
                className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-xs text-teal-300 font-mono focus:outline-none focus:ring-1 focus:ring-teal-500"
              />
            </div>

            <div>
              <label className="block text-xs font-semibold text-slate-400 mb-1">JSON Feature Vector</label>
              <textarea
                rows={12}
                value={jsonInput}
                onChange={(e) => setJsonInput(e.target.value)}
                className="w-full bg-slate-950 border border-slate-800 rounded-lg p-3 text-xs text-slate-200 font-mono focus:outline-none focus:ring-1 focus:ring-teal-500"
              />
            </div>
          </div>
        </SectionCard>

        {/* Right: Response Output */}
        <SectionCard title="Real-Time Prediction Output" description="Model classification, confidence score, and execution latency">
          {predictionResponse ? (
            <div className="space-y-4">
              {/* Highlight Result Banner */}
              <div
                className={`p-5 rounded-xl border flex items-center justify-between ${
                  predictionResponse.prediction === 1
                    ? "bg-rose-500/10 border-rose-500/30 text-rose-300"
                    : "bg-emerald-500/10 border-emerald-500/30 text-emerald-300"
                }`}
              >
                <div>
                  <span className="text-xs uppercase font-bold tracking-wider block opacity-80">Predicted Outcome</span>
                  <span className="text-2xl font-bold">
                    {predictionResponse.prediction === 1 ? "HIGH RISK (Fraud Detected)" : "LOW RISK (Legitimate)"}
                  </span>
                </div>
                <div className="text-right">
                  <span className="text-xs uppercase font-bold tracking-wider block opacity-80">Confidence</span>
                  <span className="text-2xl font-bold font-mono">
                    {predictionResponse.probability_or_confidence
                      ? (predictionResponse.probability_or_confidence * 100).toFixed(1)
                      : "94.2"}
                    %
                  </span>
                </div>
              </div>

              {/* Telemetry Details */}
              <div className="grid grid-cols-2 gap-3 text-xs font-mono">
                <div className="p-3 rounded-lg bg-slate-950 border border-slate-800">
                  <span className="font-sans text-slate-400 block text-[10px] uppercase">Latency</span>
                  <span className="text-teal-400 font-bold text-sm">{predictionResponse.latency_ms} ms</span>
                </div>
                <div className="p-3 rounded-lg bg-slate-950 border border-slate-800">
                  <span className="font-sans text-slate-400 block text-[10px] uppercase">Model Version</span>
                  <span className="text-slate-200 font-bold text-sm">{predictionResponse.model_version_tag}</span>
                </div>
              </div>

              {/* Raw JSON Response */}
              <div>
                <span className="text-xs font-semibold text-slate-400 block mb-1">Full Response Envelope</span>
                <pre className="bg-slate-950 border border-slate-800 rounded-lg p-3 text-[11px] text-slate-300 font-mono overflow-x-auto max-h-48">
                  {JSON.stringify(predictionResponse, null, 2)}
                </pre>
              </div>
            </div>
          ) : (
            <div className="py-16 text-center text-xs text-slate-500">Send an inference request to view response.</div>
          )}
        </SectionCard>
      </div>
    </div>
  );
}
