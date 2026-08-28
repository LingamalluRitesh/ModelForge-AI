'use client';

import React from 'react';
import { Card, CardHeader, CardTitle, CardContent, CardDescription } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { ShieldCheck, Lock, Eye, AlertTriangle, Key } from 'lucide-react';

export function SecurityAuditStudio() {
  const auditLogs = [
    { timestamp: '2026-08-29 02:45:10', user: 'admin@modelforge.ai', action: 'PROMOTE_MODEL_VERSION', resource: 'Risk-XGBoost-v2.4', ip: '10.100.1.45', status: 'ALLOWED' },
    { timestamp: '2026-08-29 02:40:02', user: 'mlops-bot', action: 'TRIGGER_CANARY_ROLLOUT', resource: 'Endpoint /v1/predict/credit', ip: '10.100.10.12', status: 'ALLOWED' },
    { timestamp: '2026-08-29 02:30:15', user: 'data_eng@modelforge.ai', action: 'INGEST_FEATURE_VIEW', resource: 'user_realtime_features', ip: '10.100.2.88', status: 'ALLOWED' },
    { timestamp: '2026-08-29 02:15:44', user: 'anonymous', action: 'ACCESS_REGISTRY_UNAUTHORIZED', resource: 'ModelRegistry', ip: '198.51.100.24', status: 'BLOCKED' },
  ];

  return (
    <div className="space-y-6">
      {/* Top Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-4 gap-4">
        <Card className="border border-border/60 bg-card/50 backdrop-blur p-4">
          <div className="flex items-center justify-between">
            <span className="text-xs text-muted-foreground font-medium">RBAC Security Posture</span>
            <ShieldCheck className="h-4 w-4 text-emerald-400" />
          </div>
          <div className="mt-2 text-2xl font-bold font-mono text-emerald-400">100% SECURE</div>
          <div className="mt-1 text-[11px] text-muted-foreground">Zero unauthorized data access breaches</div>
        </Card>

        <Card className="border border-border/60 bg-card/50 backdrop-blur p-4">
          <div className="flex items-center justify-between">
            <span className="text-xs text-muted-foreground font-medium">Data Encryption at Rest</span>
            <Lock className="h-4 w-4 text-sky-400" />
          </div>
          <div className="mt-2 text-2xl font-bold font-mono text-sky-400">AES-256-GCM</div>
          <div className="mt-1 text-[11px] text-muted-foreground">KMS managed keys rotated every 90 days</div>
        </Card>

        <Card className="border border-border/60 bg-card/50 backdrop-blur p-4">
          <div className="flex items-center justify-between">
            <span className="text-xs text-muted-foreground font-medium">In-Transit Encryption</span>
            <Key className="h-4 w-4 text-purple-400" />
          </div>
          <div className="mt-2 text-2xl font-bold font-mono text-purple-400">TLS 1.3 / mTLS</div>
          <div className="mt-1 text-[11px] text-muted-foreground">Istio Service Mesh sidecar mutual TLS</div>
        </Card>

        <Card className="border border-border/60 bg-card/50 backdrop-blur p-4">
          <div className="flex items-center justify-between">
            <span className="text-xs text-muted-foreground font-medium">Membership Inference Defense</span>
            <Eye className="h-4 w-4 text-amber-400" />
          </div>
          <div className="mt-2 text-2xl font-bold font-mono text-amber-400">Passed (0.012)</div>
          <div className="mt-1 text-[11px] text-muted-foreground">Empirical privacy risk score &le; 0.05</div>
        </Card>
      </div>

      {/* Security Audit Table */}
      <Card className="border border-border/60 bg-card/50 backdrop-blur">
        <CardHeader className="flex flex-row items-center justify-between pb-2">
          <div>
            <CardTitle className="text-base font-semibold">Immutable Enterprise Security Audit Trail</CardTitle>
            <CardDescription className="text-xs">
              Cryptographically signed access logs recording all model registry actions, role modifications, and deployments.
            </CardDescription>
          </div>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs font-mono">
              <thead>
                <tr className="border-b border-border/60 text-muted-foreground font-sans">
                  <th className="pb-2 font-medium">Timestamp (UTC)</th>
                  <th className="pb-2 font-medium">Principal User / Service</th>
                  <th className="pb-2 font-medium">Action Event</th>
                  <th className="pb-2 font-medium">Target Resource</th>
                  <th className="pb-2 font-medium">Client IP Address</th>
                  <th className="pb-2 font-medium">Decision</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border/30">
                {auditLogs.map((log, idx) => (
                  <tr key={idx} className="hover:bg-muted/30">
                    <td className="py-2.5 text-muted-foreground">{log.timestamp}</td>
                    <td className="py-2.5 font-sans font-semibold text-foreground">{log.user}</td>
                    <td className="py-2.5 text-primary">{log.action}</td>
                    <td className="py-2.5">{log.resource}</td>
                    <td className="py-2.5 text-muted-foreground">{log.ip}</td>
                    <td className="py-2.5">
                      {log.status === 'ALLOWED' ? (
                        <Badge className="bg-emerald-500/10 text-emerald-400 border-emerald-500/20 text-[10px]">
                          Allowed
                        </Badge>
                      ) : (
                        <Badge className="bg-rose-500/10 text-rose-400 border-rose-500/20 text-[10px]">
                          Blocked
                        </Badge>
                      )}
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
