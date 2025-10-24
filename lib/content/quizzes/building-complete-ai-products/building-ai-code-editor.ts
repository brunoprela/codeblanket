export const buildingAICodeEditorQuiz = [
  {
    id: 'bcap-bace-q-1',
    question:
      'Design the codebase indexing system for an AI code editor. How would you parse a 100k-line codebase, extract semantic chunks, generate embeddings, and build a queryable index? What happens when files change? How do you keep embeddings fresh without re-indexing everything? Include: parsing strategy, chunking heuristics, incremental updates, and query optimization.',
    sampleAnswer:
      'Parsing: Use Tree-sitter for language-agnostic AST parsing. Parse entire codebase on first load, extract: functions, classes, imports, type definitions. Chunking: Semantic chunks (complete functions/classes, not arbitrary text windows), include: function signature + docstring + body, surrounding context (imports, parent class). Chunk size: target 500-1500 tokens. Embeddings: Use Voyage Code for code-specific embeddings, batch process (50 chunks at once). Index: Store in Qdrant with metadata (file path, language, chunk type, dependencies). Incremental updates: On file save, parse only changed file, compare AST diff, re-embed only modified chunks, delete removed chunks. Optimization: (1) Cache embeddings in SQLite, keyed by file hash. (2) Lazy loading - index frequently edited files first. (3) Background worker for full reindexing. (4) Query: Embed user prompt, vector search top-k=10, re-rank by recency + relevance. Cost: 100k lines = ~3k functions = $1.50 for initial embedding (Voyage $0.0005/1k tokens), incremental updates pennies per day.',
    keyPoints: [
      'Tree-sitter for AST parsing (language-agnostic, fast, reliable)',
      'Semantic chunking: complete functions/classes, not arbitrary windows',
      'Incremental updates: AST diff, re-embed only changes',
      'Cache embeddings by file hash, avoid redundant API calls',
      'Query optimization: vector search + re-ranking by relevance/recency',
    ],
  },
  {
    id: 'bcap-bace-q-2',
    question:
      'Your AI code editor needs to generate a complex refactoring (rename variable across 50 files, update imports, fix references). How would you: (1) Plan the refactoring (which files to change, what edits), (2) Generate diffs reliably, (3) Handle conflicts, (4) Provide preview before applying, (5) Support undo? Compare LLM-based vs traditional AST-based approaches.',
    sampleAnswer:
      'Hybrid approach (AST + LLM): (1) Planning: Use AST analysis (Tree-sitter) to find all references to variable across codebase (fast, 100% accurate). Build dependency graph. (2) LLM for edge cases: Ask Claude to identify semantic references (comments, string literals, similar variables). Combine AST + LLM results. (3) Diff generation: For each file, use AST to locate exact edit positions, generate unified diff format. (4) Conflict detection: Check if files modified since analysis, re-analyze if needed. (5) Preview: Show diffs in UI, allow user to deselect changes. (6) Apply: Use git-like patch application, transaction-based (all or nothing). (7) Undo: Store original file contents + diffs, reverse apply. Why hybrid: Pure LLM is slow (must process 50 files), expensive ($2-5), and unreliable (might miss references). Pure AST misses semantic references. Hybrid gets 99% accuracy from AST (fast, free), uses LLM only for 1% edge cases. Cost: $0.10 vs $5 for pure LLM. Time: 2s vs 30s.',
    keyPoints: [
      'Hybrid approach: AST for structural analysis (fast, accurate), LLM for semantic understanding',
      'AST finds 99% of references reliably, LLM handles edge cases',
      'Transaction-based application: all changes or none (atomicity)',
      'Preview UI critical for user trust',
      'Cost/time optimization: AST is free and instant, use LLM sparingly',
    ],
  },
  {
    id: 'bcap-bace-q-3',
    question:
      'Design the context window management for code completions. You have 200k context limit but codebase is 500k tokens. For each completion request, how do you select: which files to include, how much of each file, imports vs implementations? How do you handle: cursor position, recent edits, project structure, user intent? What if user explicitly mentions a file in a comment?',
    sampleAnswer:
      'Multi-stage context building: (1) Always include: Current file (full), open tabs (truncate to key functions), recent edits (last 10 files touched). (2) Semantic search: Embed completion prompt, vector search codebase for relevant functions/classes (top-k=10). (3) Dependency analysis: If current file imports X, include X\'s public API. (4) User intent: Parse comment above cursor for explicit references ("// Use the LoginService"), fetch mentioned files. (5) Project structure: Include relevant config files (package.json, tsconfig.json). Priority order: (1) Current file [50k tokens], (2) Open tabs [30k], (3) Dependencies [20k], (4) Vector search results [30k], (5) Recent edits [20k], (6) Config [10k] = 160k total. Optimization: Truncate large files to: imports, type definitions, function signatures (omit bodies). Cache: Vector search results for 5min (avoid re-embedding same prompt). Dynamic adjustment: If completion is simple (autocomplete variable), use minimal context. If complex (generate new function), use full context. Token counting: Estimate with tiktoken, stay under 180k to leave room for response.',
    keyPoints: [
      'Prioritize: current file > open tabs > dependencies > semantic search',
      'Truncate strategically: keep signatures/types, omit function bodies',
      'Parse user intent from comments for explicit file references',
      'Dynamic context: simple completions need less, complex need more',
      'Cache semantic search results, avoid redundant vector queries',
    ],
  },
];
