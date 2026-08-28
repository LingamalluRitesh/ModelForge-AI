'use client';

import React from 'react';
import { Card, CardHeader, CardTitle, CardContent, CardDescription } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Database, Zap, Clock, ShieldCheck } from 'lucide-react';

export default function FeatureStorePage({ params }: { params: { id: string } }) {
  const featureViews = [
    { name: 'customer_online_features', entities: 'user_id', count: 42, latency: '0.8 ms', ttl: '24h', storage: 'Redis + Delta' },
    { name: 'merchant_risk_aggregations', entities: 'merchant_id', count: 18, latency: '1.2 ms', ttl: '7d', storage: 'Redis + BigQuery' },
    { name: 'device_trust_graph_embeddings', entities: 'device_hash', count: 128, latency: '2.4 ms', ttl: '30d', storage: 'pgvector (HNSW)' },
  ];

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Enterprise Feature Store & Online Cache</h1>
        <p className="text-sm text-muted-foreground">
          Sub-millisecond online serving, bi-temporal AS-OF joins, and automated entity aggregations.
        </p>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-4 gap-4">
        <Card className="border border-border/60 bg-card/50 backdrop-blur p-4">
          <div className="flex items-center justify-between">
            <span className="text-xs text-muted-foreground font-medium">Registered Feature Views</span>
            <Database className="h-4 w-4 text-primary" />
          </div>
          <div className="mt-2 text-2xl font-bold font-mono text-primary">18 Views</div>
          <div className="mt-1 text-[11px] text-muted-foreground">480 materialized feature columns</div>
        </Card>

        <Card className="border border-border/60 bg-card/50 backdrop-blur p-4">
          <div className="flex items-center justify-between">
            <span className="text-xs text-muted-foreground font-medium">Online P99 Lookup Latency</span>
            <Zap className="h-4 w-4 text-emerald-400" />
          </div>
          <div className="mt-2 text-2xl font-bold font-mono text-emerald-400">0.82 ms</div>
          <div className="mt-1 text-[11px] text-muted-foreground">Sub-millisecond Redis cluster read</div>
        </Card>

        <Card className="border border-border/60 bg-card/50 backdrop-blur p-4">
          <div className="flex items-center justify-between">
            <span className="text-xs text-muted-foreground font-medium">Point-in-Time Join Integrity</span>
            <Clock className="h-4 w-4 text-sky-400" />
          </div>
          <div className="mt-2 text-2xl font-bold font-mono text-sky-400">100% Verified</div>
          <div className="mt-1 text-[11px] text-muted-foreground">Zero future target lookahead leakage</div>
        </Card>

        <Card className="border border-border/60 bg-card/50 backdrop-blur p-4">
          <div className="flex items-center justify-between">
            <span className="text-xs text-muted-foreground font-medium">Storage Redundancy</span>
            <ShieldCheck className="h-4 w-4 text-purple-400" />
          </div>
          <div className="mt-2 text-2xl font-bold font-mono text-purple-400">Dual Sync</div>
          <div className="mt-1 text-[11px] text-muted-foreground">Online Redis & Offline Delta Lake</div>
        </Card>
      </div>

      {/* Feature Views Table */}
      <Card className="border border-border/60 bg-card/50 backdrop-blur">
        <CardHeader className="pb-2">
          <CardTitle className="text-base font-semibold">Active Enterprise Feature Views</CardTitle>
          <CardDescription className="text-xs">
            Materialized feature views serving synchronous online inference and training set point-in-time joins.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs font-mono">
              <thead>
                <tr className="border-b border-border/60 text-muted-foreground font-sans">
                  <th className="pb-2 font-medium">Feature View Name</th>
                  <th className="pb-2 font-medium">Entity Keys</th>
                  <th className="pb-2 font-medium">Feature Count</th>
                  <th className="pb-2 font-medium">Online P99 Latency</th>
                  <th className="pb-2 font-medium">TTL Window</th>
                  <th className="pb-2 font-medium">Storage Tier</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border/30">
                {featureViews.map((fv, idx) => (
                  <tr key={idx} className="hover:bg-muted/30">
                    <td className="py-2.5 font-sans font-semibold text-foreground">{fv.name}</td>
                    <td className="py-2.5 text-primary">{fv.entities}</td>
                    <td className="py-2.5 text-foreground">{fv.count} cols</td>
                    <td className="py-2.5 text-emerald-400">{fv.latency}</td>
                    <td className="py-2.5 text-muted-foreground">{fv.ttl}</td>
                    <td className="py-2.5">
                      <Badge variant="outline" className="text-[10px] border-primary/30">
                        {fv.storage}
                      </Badge>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
