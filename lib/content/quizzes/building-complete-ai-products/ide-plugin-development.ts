export const idePluginDevelopmentQuiz = [
  {
    id: 'bcap-ipd-q-1',
    question:
      'Compare developing an AI code assistant as: (1) VS Code extension, (2) JetBrains plugin, (3) LSP server, (4) Standalone editor (Cursor fork). Discuss: development complexity, distribution, user adoption, feature capabilities, and maintenance burden. Which would you choose for: (a) MVP/early stage, (b) Scale (100k+ users)?',
    sampleAnswer:
      "Comparison: (1) VS Code extension: Easy (TypeScript/JavaScript), webview UI, 50M+ potential users, limited to VS Code, moderate capabilities (can't modify core editor deeply). (2) JetBrains plugin: Harder (Java/Kotlin), 10M users, expensive IDEs ($200/yr), powerful APIs, complex UI. (3) LSP server: Universal (works with any editor), focuses on language features, limited UI, best for pure language features (autocomplete, diagnostics). (4) Standalone fork: Full control, can modify everything, distribution burden (updates, installers), fragmentation risk. For MVP: VS Code extension. Reasoning: Fastest to market (1-2 weeks), JavaScript (easy hiring), large user base, VS Code Marketplace distribution. Limitations: Can't deeply integrate (limited inline completions, no core editor mods). For scale: Standalone editor (Cursor approach). Reasoning: Full control enables best UX (e.g., Cursor's Tab prediction, cmd+K), avoids platform lock-in, can optimize performance, own update channel. Trade-off: 3-6 months development, must maintain editor fork, harder distribution. LSP server good middle ground: works everywhere, but UI-limited.",
    keyPoints: [
      'VS Code extension: fastest MVP, 50M users, limited deep integration',
      'Standalone editor: full control, best UX, high maintenance',
      'LSP server: universal compatibility, limited to language features',
      'Choose based on stage: extension for MVP, standalone for scale',
      'Trade-off: speed to market vs control vs maintenance burden',
    ],
  },
  {
    id: 'bcap-ipd-q-2',
    question:
      'Design the communication architecture between IDE plugin (frontend) and AI backend. Requirements: support streaming completions, handle network failures gracefully, work offline, rate limit users, track usage. Should you use: REST API, WebSocket, gRPC, or embedded local model? How do you handle: authentication, caching, retries, and fallbacks?',
    sampleAnswer:
      'Hybrid architecture: (1) Primary: SSE (Server-Sent Events) for streaming completions. Unidirectional, simpler than WebSocket, HTTP-based (easier auth/proxies). (2) REST API for non-streaming (search, analytics). (3) Local model for offline/latency-sensitive (small Qwen/Phi model via ONNX). (4) Protocol Buffers for binary data (codebase sync). Flow: Plugin → REST API (create session) → SSE endpoint (stream completions) → Local model (if offline). Authentication: JWT tokens in Authorization header, refresh tokens for long sessions. Caching: Plugin caches embeddings locally (IndexedDB), semantic cache for completions (Redis backend). Rate limiting: Backend tracks requests per user/tier, returns 429 with Retry-After header. Plugin queues requests, implements exponential backoff. Network failures: (1) Timeout: 30s for completion, retry 3x with exponential backoff. (2) Offline: Detect connection, switch to local model, show degraded mode indicator. (3) Resume: On reconnect, resume from last state. Fallback chain: Primary provider → Fallback provider → Local model → Show error. Cost: Track tokens client-side, sync to backend every 10 requests.',
    keyPoints: [
      'SSE for streaming (simpler than WebSocket, better than polling)',
      'Local model fallback for offline/latency-sensitive operations',
      'Exponential backoff + retry for network resilience',
      'Client-side caching (embeddings, completions) reduces API calls',
      'Fallback chain: primary → backup provider → local → error',
    ],
  },
  {
    id: 'bcap-ipd-q-3',
    question:
      "Your IDE plugin shows inline completions (ghost text). Users complain: (1) Completions appear too slowly, (2) They're often irrelevant, (3) They interrupt typing flow. How would you optimize: triggering logic, latency, relevance, and UX? When should you NOT show a completion? How do you measure success?",
    sampleAnswer:
      "Optimization: (1) Triggering: Debounce 300ms after typing stops (not every keystroke). Smart triggers: after newline, opening brace, function declaration. DON'T trigger: mid-word, inside strings, rapid typing. (2) Latency: Speculative completion (predict user will stop typing, start generating early). Show after 500ms max. If >1s, don't show (too late). Cache: Keep last 5 completions for common patterns (for loop, function template). (3) Relevance: Multi-model approach - use fast model (Haiku) for simple completions (autocomplete), slow model (Sonnet) for complex (generate function). Filter: If completion is <5 tokens or >200 tokens, likely bad. If confidence <0.7 (from model), don't show. (4) UX: Ghost text (low opacity), Tab to accept, Esc to dismiss. Show accept rate (% of completions accepted) to user. DON'T show when: (1) User typing rapidly (typing speed >200 WPM), (2) Inside string/comment unless relevant, (3) Cursor in middle of word, (4) Recently rejected completion. Metrics: (1) Accept rate (target: >25%), (2) Time-to-accept (<2s), (3) Characters saved (completion length / user typing), (4) User retention. A/B test: latency vs relevance trade-off.",
    keyPoints: [
      'Debounce 300ms, smart triggers (newline, braces), avoid mid-word',
      'Speculative completion: predict typing stop, generate early',
      'Multi-model: fast for simple, slow for complex',
      'Filter low-confidence completions (acceptance rate matters more than quantity)',
      'Measure: accept rate >25%, time-to-accept <2s, characters saved',
    ],
  },
];
