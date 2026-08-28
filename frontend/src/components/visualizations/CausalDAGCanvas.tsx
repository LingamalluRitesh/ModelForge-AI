'use client';

import React, { useState } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';

interface DAGNode {
  id: string;
  label: string;
  x: number;
  y: number;
  type: 'treatment' | 'outcome' | 'confounder' | 'mediator' | 'instrument';
}

interface DAGEdge {
  from: string;
  to: string;
  isBackdoor?: boolean;
}

export function CausalDAGCanvas() {
  const [selectedNode, setSelectedNode] = useState<string | null>('treatment');

  const nodes: DAGNode[] = [
    { id: 'confounder_age', label: 'Customer Age (Z1)', x: 180, y: 50, type: 'confounder' },
    { id: 'confounder_income', label: 'Income Bracket (Z2)', x: 420, y: 50, type: 'confounder' },
    { id: 'instrument_promo', label: 'Promo Code (IV)', x: 50, y: 180, type: 'instrument' },
    { id: 'treatment', label: 'Discount Offered (T)', x: 220, y: 180, type: 'treatment' },
    { id: 'mediator_engagement', label: 'App Session Time (M)', x: 380, y: 180, type: 'mediator' },
    { id: 'outcome', label: '30-Day Churn (Y)', x: 550, y: 180, type: 'outcome' },
  ];

  const edges: DAGEdge[] = [
    { from: 'confounder_age', to: 'treatment', isBackdoor: true },
    { from: 'confounder_age', to: 'outcome', isBackdoor: true },
    { from: 'confounder_income', to: 'treatment', isBackdoor: true },
    { from: 'confounder_income', to: 'outcome', isBackdoor: true },
    { from: 'instrument_promo', to: 'treatment' },
    { from: 'treatment', to: 'mediator_engagement' },
    { from: 'mediator_engagement', to: 'outcome' },
    { from: 'treatment', to: 'outcome' },
  ];

  const getNodeColor = (type: DAGNode['type']) => {
    switch (type) {
      case 'treatment':
        return '#38bdf8'; // sky
      case 'outcome':
        return '#34d399'; // emerald
      case 'confounder':
        return '#f59e0b'; // amber
      case 'mediator':
        return '#c084fc'; // purple
      case 'instrument':
        return '#ec4899'; // pink
    }
  };

  return (
    <Card className="border border-border/60 bg-card/50 backdrop-blur">
      <CardHeader className="flex flex-row items-center justify-between pb-2">
        <div>
          <CardTitle className="text-base font-semibold">Causal Directed Acyclic Graph (DAG)</CardTitle>
          <p className="text-xs text-muted-foreground">Pearl's Structural Causal Model & Backdoor Adjustment Set</p>
        </div>
        <div className="flex flex-wrap gap-2 text-xs">
          <Badge variant="outline" className="border-sky-500/40 text-sky-400">Treatment (T)</Badge>
          <Badge variant="outline" className="border-emerald-500/40 text-emerald-400">Outcome (Y)</Badge>
          <Badge variant="outline" className="border-amber-500/40 text-amber-400">Confounder (Z)</Badge>
          <Badge variant="outline" className="border-purple-500/40 text-purple-400">Mediator (M)</Badge>
        </div>
      </CardHeader>
      <CardContent>
        <div className="relative h-[280px] w-full rounded-lg border border-border/40 bg-background/50 p-2 overflow-hidden">
          <svg className="h-full w-full">
            <defs>
              <marker id="arrowhead" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
                <polygon points="0 0, 8 3, 0 6" fill="#64748b" />
              </marker>
              <marker id="arrowhead-backdoor" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
                <polygon points="0 0, 8 3, 0 6" fill="#f59e0b" />
              </marker>
            </defs>

            {/* Edges */}
            {edges.map((e, idx) => {
              const src = nodes.find((n) => n.id === e.from);
              const dst = nodes.find((n) => n.id === e.to);
              if (!src || !dst) return null;

              return (
                <line
                  key={idx}
                  x1={src.x}
                  y1={src.y}
                  x2={dst.x}
                  y2={dst.y}
                  stroke={e.isBackdoor ? '#f59e0b' : '#64748b'}
                  strokeWidth={e.isBackdoor ? 1.5 : 1.5}
                  strokeDasharray={e.isBackdoor ? '4 3' : undefined}
                  markerEnd={e.isBackdoor ? 'url(#arrowhead-backdoor)' : 'url(#arrowhead)'}
                  opacity={0.8}
                />
              );
            })}

            {/* Nodes */}
            {nodes.map((n) => {
              const isSelected = selectedNode === n.id;
              const color = getNodeColor(n.type);

              return (
                <g
                  key={n.id}
                  transform={`translate(${n.x}, ${n.y})`}
                  className="cursor-pointer transition-all duration-200"
                  onClick={() => setSelectedNode(n.id)}
                >
                  <circle
                    r={isSelected ? 20 : 16}
                    fill={color}
                    fillOpacity={0.2}
                    stroke={color}
                    strokeWidth={isSelected ? 2.5 : 1.5}
                  />
                  <text
                    textAnchor="middle"
                    dy="30"
                    fill="#e2e8f0"
                    fontSize="11"
                    fontFamily="monospace"
                    className="select-none font-medium"
                  >
                    {n.label}
                  </text>
                </g>
              );
            })}
          </svg>
        </div>

        <div className="mt-3 flex items-center justify-between rounded border bg-background/60 p-2 text-xs">
          <span className="text-muted-foreground">
            Backdoor Criterion Satisfied: <strong className="text-emerald-400">Yes (Conditioning on Z1, Z2 blocks all confounding paths)</strong>
          </span>
          <span className="font-mono text-primary font-medium">Estimated ATE: -14.2% Churn</span>
        </div>
      </CardContent>
    </Card>
  );
}
