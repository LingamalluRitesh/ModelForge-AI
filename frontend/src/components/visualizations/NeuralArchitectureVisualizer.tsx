'use client';

import React from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';

interface LayerNode {
  name: string;
  type: string;
  outputShape: string;
  params: number;
  activation?: string;
  color: string;
}

export function NeuralArchitectureVisualizer() {
  const layers: LayerNode[] = [
    { name: 'InputLayer', type: 'Input', outputShape: '(None, 48)', params: 0, color: '#38bdf8' },
    { name: 'FeatureTransformer_1', type: 'GLU Dense Block', outputShape: '(None, 128)', params: 6272, activation: 'Gated Linear Unit', color: '#818cf8' },
    { name: 'GhostBatchNorm_1', type: 'Ghost BN', outputShape: '(None, 128)', params: 256, color: '#a78bfa' },
    { name: 'SparsemaxAttention', type: 'Sparsemax Mask', outputShape: '(None, 48)', params: 6192, activation: 'Sparsemax', color: '#c084fc' },
    { name: 'FeatureTransformer_2', type: 'Shared GLU Block', outputShape: '(None, 128)', params: 16512, activation: 'GLU', color: '#818cf8' },
    { name: 'DecisionHead', type: 'Linear + Softmax', outputShape: '(None, 2)', params: 258, activation: 'Softmax', color: '#34d399' },
  ];

  const totalParams = layers.reduce((acc, l) => acc + l.params, 0);

  return (
    <Card className="border border-border/60 bg-card/50 backdrop-blur">
      <CardHeader className="flex flex-row items-center justify-between pb-2">
        <div>
          <CardTitle className="text-base font-semibold">TabNet Neural Computational Graph</CardTitle>
          <p className="text-xs text-muted-foreground">Sequential Layer Tensor Flow & Sparse Attention Dimension Mapping</p>
        </div>
        <Badge variant="outline" className="font-mono text-xs border-primary/40 text-primary">
          Total Parameters: {totalParams.toLocaleString()}
        </Badge>
      </CardHeader>
      <CardContent>
        <div className="flex flex-col gap-2 py-2">
          {layers.map((layer, idx) => (
            <div key={idx} className="group relative flex items-center justify-between rounded-lg border border-border/40 bg-background/50 p-2.5 transition hover:border-primary/50 hover:bg-card">
              <div className="flex items-center gap-3">
                <div
                  className="flex h-8 w-8 items-center justify-center rounded-md font-mono text-xs font-bold text-white shadow"
                  style={{ backgroundColor: layer.color }}
                >
                  L{idx + 1}
                </div>
                <div>
                  <div className="text-xs font-semibold text-foreground">{layer.name}</div>
                  <div className="text-[10px] text-muted-foreground">{layer.type} {layer.activation && `• ${layer.activation}`}</div>
                </div>
              </div>
              <div className="flex items-center gap-4 text-xs font-mono">
                <span className="rounded bg-muted/60 px-2 py-0.5 text-[11px] text-muted-foreground">
                  {layer.outputShape}
                </span>
                <span className="text-primary font-medium">{layer.params > 0 ? `${layer.params.toLocaleString()} params` : '—'}</span>
              </div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}
