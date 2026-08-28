'use client';

import React, { useState } from 'react';
import { Card, CardHeader, CardTitle, CardContent, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { Search, Database, FileText, CheckCircle2, Zap } from 'lucide-react';

export function KnowledgeBaseRAGStudio() {
  const [query, setQuery] = useState('How does ModelForge AI enforce zero lookahead leakage in feature joins?');
  const [isSearching, setIsSearching] = useState(false);
  const [results, setResults] = useState([
    {
      document: 'feature_store_architecture_spec.pdf',
      chunkIndex: 4,
      cosineSimilarity: 0.924,
      snippet: 'ModelForge Feature Store uses strict AS-OF temporal joins. Feature timestamps must strictly precede the observation timestamp (t_feature <= t_observation), preventing any future data leakage into the training matrix.',
    },
    {
      document: 'governance_quality_gates_handbook.md',
      chunkIndex: 12,
      cosineSimilarity: 0.865,
      snippet: 'Automated Quality Gate validation evaluates out-of-fold F1 scores, demographic parity disparate impact ratios (>= 0.80), and PSI population stability drift (< 0.20) prior to production Canary deployment.',
    },
  ]);

  const handleSearch = () => {
    setIsSearching(true);
    setTimeout(() => {
      setIsSearching(false);
    }, 400);
  };

  return (
    <div className="space-y-6">
      {/* Search Header */}
      <Card className="border border-border/60 bg-card/50 backdrop-blur">
        <CardHeader className="pb-2">
          <CardTitle className="text-base font-semibold flex items-center gap-2">
            <Database className="h-4 w-4 text-primary" />
            HNSW Vector Retrieval & RAG Semantic Search
          </CardTitle>
          <CardDescription className="text-xs">
            Query dense pgvector 1536-dimensional embeddings with sub-millisecond approximate nearest neighbor search.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex gap-2">
            <Input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              className="text-xs font-mono h-9"
              placeholder="Search enterprise knowledge base..."
            />
            <Button size="sm" onClick={handleSearch} disabled={isSearching} className="gap-1.5 text-xs">
              <Search className="h-3.5 w-3.5" />
              {isSearching ? 'Searching...' : 'Search Vector Index'}
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Semantic Results */}
      <div className="space-y-3">
        {results.map((res, idx) => (
          <Card key={idx} className="border border-border/50 bg-background/50 p-4">
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                <FileText className="h-4 w-4 text-primary" />
                <span className="font-semibold text-xs text-foreground">{res.document} (Chunk #{res.chunkIndex})</span>
              </div>
              <Badge variant="outline" className="border-emerald-500/40 text-emerald-400 font-mono text-[10px]">
                Cosine Similarity: {(res.cosineSimilarity * 100).toFixed(1)}%
              </Badge>
            </div>
            <p className="text-xs text-muted-foreground leading-relaxed font-sans">{res.snippet}</p>
          </Card>
        ))}
      </div>
    </div>
  );
}
