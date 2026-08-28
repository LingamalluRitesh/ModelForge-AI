'use client';

import React from 'react';
import { KnowledgeBaseRAGStudio } from '@/components/studios/KnowledgeBaseRAGStudio';

export default function KnowledgeBasesPage({ params }: { params: { id: string } }) {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Enterprise Vector Knowledge Bases (RAG)</h1>
        <p className="text-sm text-muted-foreground">
          Sub-millisecond dense pgvector retrieval, semantic document search, and HNSW graph indexing.
        </p>
      </div>

      <KnowledgeBaseRAGStudio />
    </div>
  );
}
