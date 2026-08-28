'use client';

import React from 'react';
import { Card, CardHeader, CardTitle, CardContent, CardDescription } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { ShieldCheck, Scale, FileText, CheckCircle2 } from 'lucide-react';

export default function ModelGovernancePage({ params }: { params: { id: string } }) {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Enterprise Model Governance & Multi-Sig Sign-Off</h1>
        <p className="text-sm text-muted-foreground">
          Four-Fifths disparate impact verification, EU AI Act conformity attestations, and cryptographic approvals.
        </p>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <Card className="border border-border/60 bg-card/50 backdrop-blur p-4">
          <div className="flex items-center justify-between">
            <span className="text-xs text-muted-foreground font-medium">Four-Fifths Fair Lending Rule</span>
            <Scale className="h-4 w-4 text-emerald-400" />
          </div>
          <div className="mt-2 text-xl font-bold text-emerald-400 font-mono">0.912 Passed</div>
          <div className="mt-1 text-[11px] text-muted-foreground">Disparate impact ratio &ge; 0.80 across protected attributes</div>
        </Card>

        <Card className="border border-border/60 bg-card/50 backdrop-blur p-4">
          <div className="flex items-center justify-between">
            <span className="text-xs text-muted-foreground font-medium">Multi-Sig Approvals</span>
            <ShieldCheck className="h-4 w-4 text-primary" />
          </div>
          <div className="mt-2 text-xl font-bold text-primary font-mono">3 / 3 Signed</div>
          <div className="mt-1 text-[11px] text-muted-foreground">Lead DS, DPO, and VP Engineering signed off</div>
        </Card>

        <Card className="border border-border/60 bg-card/50 backdrop-blur p-4">
          <div className="flex items-center justify-between">
            <span className="text-xs text-muted-foreground font-medium">Compliance Dossier</span>
            <FileText className="h-4 w-4 text-purple-400" />
          </div>
          <div className="mt-2 text-xl font-bold text-purple-400">Annex IV Validated</div>
          <div className="mt-1 text-[11px] text-muted-foreground">Cryptographic SHA-256 seal generated</div>
        </Card>
      </div>
    </div>
  );
}
