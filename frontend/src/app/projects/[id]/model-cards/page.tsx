'use client';

import React from 'react';
import { ModelCardStudio } from '@/components/studios/ModelCardStudio';

export default function ModelCardsPage({ params }: { params: { id: string } }) {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">EU AI Act & NIST Model Cards Studio</h1>
        <p className="text-sm text-muted-foreground">
          Automated regulatory compliance documentation, fairness attestations, and audit sign-offs.
        </p>
      </div>

      <ModelCardStudio />
    </div>
  );
}
