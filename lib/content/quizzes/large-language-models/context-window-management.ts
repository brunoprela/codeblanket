export const contextWindowManagementQuiz = {
  title: 'Context Window Management Discussion',
  id: 'context-window-management-quiz',
  sectionId: 'context-window-management',
  questions: [
    {
      id: 1,
      question:
        'Models like Claude 3 support 200k token context windows, yet longer context doesn\'t always improve performance—models struggle with "lost in the middle" where they miss information in the center of long contexts. Explain this phenomenon and discuss strategies to address it: context compression, reranking, hierarchical summarization, and retrieval. When should you use long context vs retrieval?',
      expectedAnswer:
        'Should cover: attention dilution across long contexts, position bias toward beginning/end, empirical studies showing degraded performance with context length, context compression via summarization, reranking to put relevant info at ends, hierarchical processing (summarize sections then combine), RAG as alternative to long context, cost comparison (long context vs retrieval), latency considerations, when documents must be processed as whole (legal contracts) vs when retrieval suffices (QA over docs), and hybrid approaches combining both.',
    },
    {
      id: 2,
      question:
        'Discuss the different document chunking strategies: fixed-size, semantic, and hierarchical. How does chunking strategy affect RAG performance and context management? What are the tradeoffs in chunk size: small chunks vs large chunks? How do overlap and metadata factor into effective chunking?',
      expectedAnswer:
        'Should analyze: fixed-size simplicity but boundary problems, semantic chunking preserving meaning (paragraph/section boundaries), hierarchical enabling multiple granularities, small chunks (256 tokens) giving precise retrieval but missing context, large chunks (1024 tokens) providing context but less precise matching, overlap (10-20%) addressing boundary issues, metadata (headings, source info) improving filtering and attribution, token counting importance for context limits, and optimal chunk size being task and domain dependent (technical docs vs narrative text).',
    },
    {
      id: 3,
      question:
        'Managing conversation history in chatbots requires balancing context coherence with token limits and cost. Compare strategies: sliding window (keep last N messages), summarization (compress old messages), and importance-based selection. How do you maintain long-term memory while controlling context size? Discuss the impact on conversation quality and cost.',
      expectedAnswer:
        'Should cover: sliding window simplicity but losing long-term context, summarization preserving key information but losing nuance, importance scoring keeping relevant messages, hybrid approaches (summary + recent messages), entity tracking across conversation, cost implications of long conversations, user experience with memory loss, techniques for explicit memory (databases of facts), when to start fresh conversations, conversation branching and context management, and monitoring conversation quality metrics.',
    },
  ],
};
