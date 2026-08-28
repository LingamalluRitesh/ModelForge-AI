'use client';

import React from 'react';
import { FairnessGovernanceStudio } from '@/components/studios/FairnessGovernanceStudio';

export default function FairnessPage({ params }: { params: { id: string } }) {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Fairness, Bias Mitigation & Ethical AI Studio</h1>
        <p className="text-sm text-muted-foreground">
          Four-Fifths Rule compliance, Equalized Odds post-processing, and Adversarial Gradient Reversal debiasing.
        </p>
      </div>

      <FairnessGovernanceStudio />
    </div>
  );
}
