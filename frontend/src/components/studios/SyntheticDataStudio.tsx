'use client';

import React, { useState } from 'react';
import { Card, CardHeader, CardTitle, CardContent, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { Sparkles, Database, ShieldCheck, Download, RefreshCw } from 'lucide-react';

export function SyntheticDataStudio() {
  const [numRows, setNumRows] = useState('10000');
  const [method, setMethod] = useState<'CTGAN' | 'GAUSSIAN_COPULA' | 'PRIVBAYES'>('CTGAN');
  const [isGenerating, setIsGenerating] = useState(false);
  const [fidelityScore, setFidelityScore] = useState(0.948);
  const [privacyScore, setPrivacyScore] = useState(0.982);

  const handleGenerate = () => {
    setIsGenerating(true);
    setTimeout(() => {
      setIsGenerating(false);
    }, 600);
  };

  return (
    <div className="space-y-6">
      {/* Top Metrics Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <Card className="border border-border/60 bg-card/50 backdrop-blur p-4">
          <div className="flex items-center justify-between">
            <span className="text-xs text-muted-foreground font-medium">Statistical Fidelity Score</span>
            <Sparkles className="h-4 w-4 text-emerald-400" />
          </div>
          <div className="mt-2 text-2xl font-bold font-mono text-emerald-400">{(fidelityScore * 100).toFixed(1)}%</div>
          <div className="mt-1 text-[11px] text-muted-foreground">Wasserstein distance + Correlation preservation</div>
        </Card>

        <Card className="border border-border/60 bg-card/50 backdrop-blur p-4">
          <div className="flex items-center justify-between">
            <span className="text-xs text-muted-foreground font-medium">Privacy Defense Rating</span>
            <ShieldCheck className="h-4 w-4 text-sky-400" />
          </div>
          <div className="mt-2 text-2xl font-bold font-mono text-sky-400">{(privacyScore * 100).toFixed(1)}%</div>
          <div className="mt-1 text-[11px] text-muted-foreground">Zero membership inference leakage</div>
        </Card>

        <Card className="border border-border/60 bg-card/50 backdrop-blur p-4">
          <div className="flex items-center justify-between">
            <span className="text-xs text-muted-foreground font-medium">Differential Privacy Bound</span>
            <Badge className="bg-purple-500/10 text-purple-400 border-purple-500/20 text-[10px]">
              DP Guaranteed
            </Badge>
          </div>
          <div className="mt-2 text-2xl font-bold font-mono text-purple-400">&epsilon; = 1.0, &delta; = 10⁻⁵</div>
          <div className="mt-1 text-[11px] text-muted-foreground">Rigorous mathematical privacy budget</div>
        </Card>
      </div>

      {/* Generator Control Card */}
      <Card className="border border-border/60 bg-card/50 backdrop-blur">
        <CardHeader className="flex flex-row items-center justify-between pb-2">
          <div>
            <CardTitle className="text-base font-semibold">Synthetic Data Generation Engine</CardTitle>
            <CardDescription className="text-xs">
              Generate privacy-safe synthetic tabular datasets conditioned on complex joint distributions and multimodal constraints.
            </CardDescription>
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <div>
              <label className="text-xs text-muted-foreground mb-1 block">Generation Algorithm</label>
              <div className="flex gap-2">
                <Button
                  size="sm"
                  variant={method === 'CTGAN' ? 'default' : 'outline'}
                  className="text-xs flex-1"
                  onClick={() => setMethod('CTGAN')}
                >
                  CTGAN
                </Button>
                <Button
                  size="sm"
                  variant={method === 'GAUSSIAN_COPULA' ? 'default' : 'outline'}
                  className="text-xs flex-1"
                  onClick={() => setMethod('GAUSSIAN_COPULA')}
                >
                  Copula
                </Button>
                <Button
                  size="sm"
                  variant={method === 'PRIVBAYES' ? 'default' : 'outline'}
                  className="text-xs flex-1"
                  onClick={() => setMethod('PRIVBAYES')}
                >
                  PrivBayes
                </Button>
              </div>
            </div>

            <div>
              <label className="text-xs text-muted-foreground mb-1 block">Number of Synthetic Rows</label>
              <Input
                type="number"
                value={numRows}
                onChange={(e) => setNumRows(e.target.value)}
                className="h-9 text-xs font-mono"
              />
            </div>

            <div className="flex items-end">
              <Button
                className="w-full h-9 gap-1.5 text-xs"
                onClick={handleGenerate}
                disabled={isGenerating}
              >
                <RefreshCw className={`h-3.5 w-3.5 ${isGenerating ? 'animate-spin' : ''}`} />
                {isGenerating ? 'Synthesizing...' : 'Generate Synthetic Batch'}
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
