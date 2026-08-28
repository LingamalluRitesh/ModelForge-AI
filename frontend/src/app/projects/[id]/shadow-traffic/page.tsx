'use client';

import React from 'react';
import { ShadowTrafficStudio } from '@/components/studios/ShadowTrafficStudio';

export default function ShadowTrafficPage({ params }: { params: { id: string } }) {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Dark Shadow Traffic Mirroring Studio</h1>
        <p className="text-sm text-muted-foreground">
          Zero-impact live production traffic mirroring for safe candidate model validation.
        </p>
      </div>

      <ShadowTrafficStudio />
    </div>
  );
}
