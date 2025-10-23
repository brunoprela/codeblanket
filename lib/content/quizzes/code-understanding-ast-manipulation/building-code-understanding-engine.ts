/**
 * Quiz questions for Building a Code Understanding Engine section
 */

export const buildingcodeunderstandingengineQuiz = [
  {
    id: 'cuam-buildingcodeunderstandingengine-q-1',
    question:
      'Why is a semantic index critical for code understanding engines? What would be the limitations of relying only on AST traversal for every query?',
    hint: 'Think about query speed, cross-file references, and scalability.',
    sampleAnswer:
      "AST traversal for every query is **too slow** for large codebases. **Problems without index**: 1) **Slow queries** - finding all callers of function requires parsing/traversing entire codebase (seconds/minutes), 2) **No cross-file links** - each file's AST is isolated, can't find references in other files without scanning everything, 3) **Doesn't scale** - million-line codebase = million lines to parse on every query, 4) **Complex queries impossible** - 'find all implementations of interface' requires type analysis across all files. **Semantic Index solution**: Pre-build database of symbols, references, types, relationships. Query is database lookup, not code parsing. Benefits: 1) **Fast** - millisecond queries even on huge codebases, 2) **Cross-file** - tracks references across entire project, 3) **Scalable** - query time independent of codebase size, 4) **Rich queries** - 'find all callers', 'type hierarchy', 'unused exports' all instant. For Cursor: When you jump to definition/find references, it's querying the semantic index, not re-parsing all files. This is how modern IDEs feel instant - the hard work (indexing) is done incrementally in the background, queries just look up pre-computed information.",
    keyPoints: [
      'Pre-builds database of symbols and relationships',
      'Enables fast queries (ms vs seconds)',
      'Supports cross-file analysis',
      'Essential for large codebase scalability',
    ],
  },
  {
    id: 'cuam-buildingcodeunderstandingengine-q-2',
    question:
      'Explain the trade-offs between real-time indexing vs. batch indexing for a code understanding engine. When would you choose each approach?',
    hint: 'Consider accuracy, performance, resource usage, and user experience.',
    sampleAnswer:
      "**Real-time indexing**: Update index immediately on every edit. **Pros**: Index always accurate, no stale data, instant feature updates. **Cons**: Can slow down editing on large files, consumes CPU continuously, complex to implement (race conditions). **Batch indexing**: Update index periodically or on-demand (save, commit). **Pros**: Doesn't impact editing performance, simpler implementation, can optimize batch processing. **Cons**: Index can be stale, features show outdated info, delays before new code indexed. **Hybrid approach** (best practice): 1) **Real-time for current file** - user's edits reflected immediately, 2) **Incremental for project** - index changed files in background with low priority, 3) **Full re-index** - periodic or on-demand (rebuild entire index). **Choose real-time when**: Editor features (autocomplete, diagnostics) - must be accurate. **Choose batch when**: Background analysis (code metrics, dead code detection) - can tolerate delay. For Cursor: Uses hybrid - your edits update immediately (real-time), project-wide changes indexed incrementally in background. This balances responsiveness (no lag while typing) with accuracy (features stay up-to-date).",
    keyPoints: [
      'Real-time: accurate but resource-intensive',
      'Batch: efficient but can be stale',
      'Hybrid approach often best',
      'Balance accuracy vs. performance',
    ],
  },
  {
    id: 'cuam-buildingcodeunderstandingengine-q-3',
    question:
      'What are the key components needed to build a production-ready code understanding engine? Describe how they interact.',
    hint: 'Think about parsing, indexing, querying, and maintaining consistency.',
    sampleAnswer:
      '**Core components**: 1) **Parser layer** - converts code to AST (tree-sitter for multi-language), 2) **Semantic analyzer** - resolves types, symbols, builds scope graphs, 3) **Indexer** - builds searchable database of symbols, references, relationships, 4) **Query engine** - fast lookups (find definition, references, callers), 5) **Watch service** - monitors file changes, triggers re-indexing, 6) **Cache layer** - stores parsed trees, analysis results for performance, 7) **API layer** - exposes features to editor/tools. **Interaction flow**: User edits file → Watch service detects change → Parser generates AST → Semantic analyzer extracts symbols/types → Indexer updates database → Query engine uses updated index → Features (autocomplete, diagnostics) reflect changes. **Key challenges**: 1) **Consistency** - ensure index matches code state, 2) **Performance** - sub-100ms for interactive features, 3) **Incremental updates** - only re-analyze what changed, 4) **Multi-language** - support different languages uniformly. For production: Need robust error handling (partial index on parse errors), horizontal scaling (large mono-repos), plugin system (language-specific rules). This is what powers Cursor, rust-analyzer, pyright - comprehensive engines, not simple parsers.',
    keyPoints: [
      'Parser, analyzer, indexer, query engine',
      'Watch service for file monitoring',
      'Cache layer for performance',
      'Incremental updates for efficiency',
    ],
  },
];
