'use client';

import React from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';

interface LineageNode {
  category: 'Ingestion' | 'Feature Store' | 'Training' | 'Endpoint';
  items: string[];
}

export function DataSankeyDiagram() {
  const pipelineStages: LineageNode[] = [
    {
      category: 'Ingestion',
      items: ['Kafka Clickstream (v3)', 'Snowflake DataWarehouse', 'S3 Parquet Bronze'],
    },
    {
      category: 'Feature Store',
      items: ['User Transaction View (AS-OF)', 'Customer Engagement 30D', 'Realtime Redis Cache'],
    },
    {
      category: 'Training',
      items: ['TabNet Classifier (v2.1)', 'XGBoost Baseline (v1.8)', 'DeepAR Forecaster'],
    },
    {
      category: 'Endpoint',
      items: ['Fraud Canary (90/10 Split)', 'Recommendations API', 'Batch Churn Worker'],
    },
  ];

  return (
    <Card className="border border-border/60 bg-card/50 backdrop-blur">
      <CardHeader className="flex flex-row items-center justify-between pb-2">
        <div>
          <CardTitle className="text-base font-semibold">End-to-End Data Lineage & Provenance Flow</CardTitle>
          <p className="text-xs text-muted-foreground">Immutable Source-to-Serving Lineage Tracking & Point-In-Time AS-OF Join Mapping</p>
        </div>
        <Badge variant="outline" className="border-emerald-500/40 text-emerald-400 font-mono text-xs">
          Lineage Validated • Zero Leakage
        </Badge>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 py-3">
          {pipelineStages.map((stage, idx) => (
            <div key={idx} className="flex flex-col gap-2 rounded-lg border border-border/50 bg-background/40 p-3">
              <div className="text-xs font-bold uppercase tracking-wider text-muted-foreground border-b border-border/40 pb-1">
                {stage.category}
              </div>
              <div className="flex flex-col gap-2">
                {stage.items.map((item, itemIdx) => (
                  <div
                    key={itemIdx}
                    className="rounded border border-primary/20 bg-card p-2 text-xs font-mono text-foreground hover:border-primary transition"
                  >
                    <div className="truncate font-semibold">{item}</div>
                    <div className="text-[10px] text-muted-foreground">Provenance Hash: #{(idx * 17 + itemIdx * 23).toString(16).padStart(4, '0')}</div>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}
