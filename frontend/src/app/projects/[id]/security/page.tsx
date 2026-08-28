'use client';

import React from 'react';
import { SecurityAuditStudio } from '@/components/studios/SecurityAuditStudio';

export default function SecurityAuditPage({ params }: { params: { id: string } }) {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Security Posture & Compliance Audit Trail</h1>
        <p className="text-sm text-muted-foreground">
          Cryptographically signed immutable access logs, membership inference defenses, and RBAC governance.
        </p>
      </div>

      <SecurityAuditStudio />
    </div>
  );
}
