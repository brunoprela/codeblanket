/**
 * Quiz questions for Tokens, Context Windows & Limitations section
 */

export const tokenscontextwindowsQuiz = [
    {
        id: 'q1',
        question:
            'Your application processes legal documents that are often 50,000+ tokens long, but your model has a 16K context window. Compare the trade-offs between chunking with overlap, hierarchical summarization, and RAG (retrieval) approaches. Which would you recommend and why?',
        sampleAnswer:
            'For legal documents, I recommend RAG (Retrieval-Augmented Generation) as the primary approach. Here is why: Chunking with overlap works for sequential reading but fails when answers require information from multiple distant sections (e.g., a clause in Section 2 that references definitions in Section 15). You would need to process multiple chunks and synthesize, which is expensive and error-prone. Hierarchical summarization risks losing critical legal details - in legal contexts, specific wording matters enormously, and summarization might drop important qualifiers like "except where", "notwithstanding", or specific date ranges. Even minor omissions can completely change legal meaning. RAG is optimal because: (1) It retrieves only relevant sections based on the question, keeping all original detailed wording, (2) Legal documents have clear structure (sections, clauses) making chunking and indexing straightforward, (3) Most legal questions reference specific topics, so semantic search finds them reliably, (4) It scales to unlimited document length without quality loss, and (5) You can cite exact sections in responses for verifiability. Implementation: Chunk by logical sections (clauses, paragraphs), create embeddings for each chunk, store in vector database with metadata (section number, title), for each question, retrieve top 5-10 most relevant chunks (fitting in context), and present with section references. The key is preserving original text while selecting relevant portions - perfect for legal use cases where precision matters more than synthesis.',
        keyPoints: [
            'Legal documents require precision - summarization loses details',
            'Chunking alone fails for multi-section questions',
            'RAG retrieves relevant sections without losing information',
            'Legal structure (sections/clauses) maps well to chunks',
            'Verifiability through section citations is critical'
        ]
    },
    {
        id: 'q2',
        question:
            'Explain the relationship between tokens and cost in LLM applications. If you had a 10,000-character document to process, how would you minimize tokens while preserving essential information?',
        sampleAnswer:
            'Token management directly impacts cost because pricing is per-token (~$0.50-$30 per 1M tokens depending on model). A 10,000-character document is roughly 2,500 tokens at 4 chars/token. At $0.50/1M (GPT-3.5), that is $0.00125 per input, but if you process this document 10,000 times/day, it is $12.50/day or $375/month just for input tokens. To minimize tokens: (1) Remove unnecessary whitespace and formatting (markdown, extra line breaks, indentation) - can save 10-20% tokens without information loss. (2) Use abbreviations consistently (e.g., "United States" → "US", "for example" → "e.g.") - natural language is verbose. (3) Remove redundancy - documents often repeat information or have boilerplate that is not needed for the task. (4) Extract only relevant sections - if you are analyzing a contract for payment terms, do not send the entire document, extract just the relevant sections using simple parsing first. (5) Use summarization for background context - if some sections provide context but are not directly analyzed, summarize them first. (6) For repeated processing, compute it once and cache - if you are extracting structured data from the same document multiple times, extract once and reuse. Real example: A 10,000-char legal document processed for multiple queries: Instead of sending full document each time (2,500 tokens × 100 queries = 250K tokens), extract relevant sections per query (500 tokens × 100 queries = 50K tokens) - 80% token reduction, same quality for focused queries.',
        keyPoints: [
            'Tokens directly determine API costs',
            'Remove whitespace and formatting without info loss',
            'Extract relevant sections instead of full documents',
            'Abbreviate and deduplicate where possible',
            'Cache processed results for reuse'
        ]
    },
    {
        id: 'q3',
        question:
            'Your application uses both GPT-3.5 (16K context) and Claude 3 (200K context). Describe a strategy for intelligently routing requests to the appropriate model based on context requirements.',
        sampleAnswer:
            'My routing strategy would be: (1) Measure context size: Count tokens using tiktoken for each request before routing. (2) Define thresholds: 0-8K tokens (50% of GPT-3.5 limit) → route to GPT-3.5 (cheaper at $0.50/1M input vs Claude at $3/1M), 8K-16K tokens → still GPT-3.5 but monitor truncation risk, 16K-100K tokens → route to Claude (required, GPT-3.5 cannot handle), 100K+ tokens → summarize first or use RAG, even Claude 200K can struggle with quality at extreme lengths. (3) Task-based exceptions: For simple extraction/classification, prefer GPT-3.5 even with larger context by using extraction instead of full document. For complex analysis requiring full document understanding, prefer Claude even for smaller documents to leverage better instruction following. (4) Cost optimization: Track actual token usage patterns and refine thresholds monthly based on: average costs per request, quality metrics (do longer documents actually need Claude?), and user satisfaction. (5) Fallback strategy: If GPT-3.5 fails due to context (shouldn\'t happen with proper routing but can occur with unexpected input), automatically retry with Claude. (6) Implementation: Create ModelRouter class with route(messages, context_length, task_type) method, track routing decisions and outcomes for analysis, and expose metrics dashboard showing distribution of requests across models.Real impact: If 70 % of requests are under 8K tokens, routing them to GPT - 3.5 saves 6x on costs compared to always using Claude, while ensuring the 30 % that need Claude get proper handling.',
        keyPoints: [
            'Route based on measured token count, not guesses',
            'Use cheaper models for requests within their limits',
            'Define clear thresholds with safety margins',
            'Consider task complexity, not just token count',
            'Track routing decisions to optimize over time'
        ]
    }
];

