/**
 * Quiz questions for Context Management & Truncation section
 */

export const contextmanagementtruncationQuiz = [
    {
        id: 'peo-context-q-1',
        question:
            'Explain different truncation strategies (beginning, end, middle, smart). When would you use each strategy?',
        hint: 'Consider content type, where important information typically is, and task requirements.',
        sampleAnswer:
            'Truncation strategies: 1) KEEP BEGINNING: Use for documentation, tutorials, where introduction/setup important - preserves context; 2) KEEP END: Use for conversations, logs, where recent events most relevant - chronological systems; 3) KEEP MIDDLE: Rare, use when know relevant content in middle section - specific targeted analysis; 4) SMART (beginning + end): Best for code files, documents where structure matters - preserves headers/imports + conclusions/main logic, shows "..." for truncated middle. WHEN TO USE: Code files → smart (imports at top, main logic at bottom); Conversations → end (recent context); Documentation → beginning (overview first); Error logs → end (recent errors); Legal documents → smart (definitions + conclusions). Cursor uses smart truncation for large code files. IMPLEMENTATION: Token count start/end sections, calculate budget split, decode with truncation marker. Test each strategy on your domain to find optimal.',
        keyPoints: [
            'Beginning: preserves introduction and context',
            'End: keeps most recent, relevant for chronological',
            'Smart (beginning + end): best for structured content',
            'Code files: smart truncation preserves imports + logic',
            'Conversations: end truncation for recency',
            'Test strategies on your specific content type',
        ],
    },
    {
        id: 'peo-context-q-2',
        question:
            'How do you manage context for multi-document scenarios? Design a hierarchical context management system with priorities.',
        hint: 'Think about priority levels, token budgets, relevance ranking, and dynamic selection.',
        sampleAnswer:
            'Hierarchical context system: 1) PRIORITY LEVELS: System (1.0) - instructions, always included; Current (0.9) - active file/selection; Recent (0.7) - recently viewed; Relevant (0.5) - related files; Background (0.3) - documentation; 2) BUDGET ALLOCATION: Reserve tokens for response (500), allocate by priority: System 10%, Current 40%, Recent 20%, Relevant 20%, Background 10%; 3) DOCUMENT SCORING: Recency score, relevance score (embedding similarity), user interaction score, combine with priority; 4) SELECTION ALGORITHM: Sort items by (priority × score), add to context until budget exhausted, truncate last item if needed; 5) DYNAMIC UPDATES: Re-rank on each query, adapt to user focus. EXAMPLE: 8K token budget: System 800, Current 3200, Recent 1600, Relevant 1600, Background 800, Response 500. Cursor implements similar: system prompt + current file + related files + user query. This ensures most important context always included while respecting limits.',
        keyPoints: [
            'Assign priorities: System > Current > Recent > Relevant > Background',
            'Allocate token budget by priority percentages',
            'Score documents by relevance and recency',
            'Select highest priority×score items until budget full',
            'Truncate intelligently if item doesn\'t fit fully',
            'Re-rank dynamically based on user actions',
        ],
    },
    {
        id: 'peo-context-q-3',
        question:
            'What is sliding window processing for long documents? When is it preferable to single-shot summarization?',
        hint: 'Consider very long documents, information density, and aggregation strategies.',
        sampleAnswer:
            'Sliding window: Process long document in overlapping chunks, aggregate results. PROCESS: 1) Split document into windows (e.g., 2000 tokens); 2) Add overlap (25%) to maintain continuity; 3) Process each window independently; 4) Aggregate results (combine summaries, merge entities, etc.). WHEN PREFERABLE: Documents exceeding context limit (>128K tokens); Dense information requiring detailed analysis (can\'t compress losslessly); Parallel processing possible (multiple windows simultaneously); Need comprehensive coverage without information loss. SINGLE-SHOT BETTER WHEN: Document fits in context window; Holistic understanding needed; Relationships between parts critical; Summary or high-level analysis sufficient. TRADE-OFFS: Sliding window → slower (multiple API calls), more expensive, but handles any length and preserves detail. Single-shot → faster, cheaper, but limited by context window and may miss details. IMPLEMENTATION: For 10K token document with 8K limit → 2 windows with 25% overlap → process both → combine. Used for analyzing long codebases, legal documents, research papers.',
        keyPoints: [
            'Process long docs in overlapping chunks',
            'Overlap (25%) prevents information loss',
            'Use for docs exceeding context limits',
            'More expensive but handles any length',
            'Aggregate results from all windows',
            'Better than single-shot when detail matters',
        ],
    },
];

