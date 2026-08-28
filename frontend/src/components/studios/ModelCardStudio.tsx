'use client';

import React from 'react';
import { Card, CardHeader, CardTitle, CardContent, CardDescription } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { FileCheck, ShieldCheck, Download, Scale, BookOpen, AlertCircle } from 'lucide-react';

export function ModelCardStudio() {
  return (
    <div className="space-y-6">
      {/* Top Banner */}
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h2 className="text-lg font-bold text-foreground flex items-center gap-2">
            <FileCheck className="h-5 w-5 text-emerald-400" />
            EU AI Act & NIST AI RMF Automated Model Card
          </h2>
          <p className="text-xs text-muted-foreground">
            Version v2.4.0 • Cryptographically Signed Model Lineage & Governance FactSheet.
          </p>
        </div>

        <Button size="sm" className="gap-1.5 text-xs">
          <Download className="h-3.5 w-3.5" />
          Export Compliance PDF
        </Button>
      </div>

      {/* Model Details Summary */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <Card className="border border-border/60 bg-card/50 backdrop-blur p-4">
          <div className="text-xs text-muted-foreground">Compliance Frameworks</div>
          <div className="mt-1 font-semibold text-foreground text-sm">EU AI Act (Annex IV) • NIST AI RMF</div>
          <div className="mt-1">
            <Badge className="bg-emerald-500/10 text-emerald-400 border-emerald-500/20 text-[10px]">
              Audit Verified
            </Badge>
          </div>
        </Card>

        <Card className="border border-border/60 bg-card/50 backdrop-blur p-4">
          <div className="text-xs text-muted-foreground">Demographic Parity Disparate Impact</div>
          <div className="mt-1 font-mono font-bold text-emerald-400 text-lg">0.912 (Passed)</div>
          <div className="text-[11px] text-muted-foreground">EEOC Four-Fifths rule threshold &ge; 0.80</div>
        </Card>

        <Card className="border border-border/60 bg-card/50 backdrop-blur p-4">
          <div className="text-xs text-muted-foreground">Human-in-the-Loop Protocol</div>
          <div className="mt-1 font-semibold text-foreground text-sm">Active Uncertainty Routing</div>
          <div className="text-[11px] text-muted-foreground">Borderline predictions flagged to human reviewer</div>
        </Card>
      </div>

      {/* Structured Sections */}
      <Card className="border border-border/60 bg-card/50 backdrop-blur">
        <CardHeader className="pb-2">
          <CardTitle className="text-base font-semibold">1. Model Intended Use & Limitations</CardTitle>
        </CardHeader>
        <CardContent className="text-xs text-muted-foreground space-y-2 leading-relaxed">
          <p>
            <strong className="text-foreground">Primary Intended Use:</strong> Real-time automated transaction scoring and risk assessment on high-throughput streaming events.
          </p>
          <p>
            <strong className="text-foreground">Out-of-Scope Uses:</strong> Fully autonomous medical diagnostics or decisions without authorized clinician validation.
          </p>
        </CardContent>
      </Card>

      <Card className="border border-border/60 bg-card/50 backdrop-blur">
        <CardHeader className="pb-2">
          <CardTitle className="text-base font-semibold">2. Quantitative Performance & Robustness Metrics</CardTitle>
        </CardHeader>
        <CardContent className="text-xs font-mono">
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
            <div>
              <span className="text-muted-foreground block text-[11px]">Holdout ROC-AUC</span>
              <span className="text-emerald-400 font-bold text-base">0.942</span>
            </div>
            <div>
              <span className="text-muted-foreground block text-[11px]">Macro F1-Score</span>
              <span className="text-primary font-bold text-base">0.884</span>
            </div>
            <div>
              <span className="text-muted-foreground block text-[11px]">Expected Calibration Error</span>
              <span className="text-sky-400 font-bold text-base">0.018</span>
            </div>
            <div>
              <span className="text-muted-foreground block text-[11px]">Certified L2 Radius</span>
              <span className="text-purple-400 font-bold text-base">0.420</span>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
