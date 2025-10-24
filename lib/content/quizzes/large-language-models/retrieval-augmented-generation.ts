export const retrievalAugmentedGenerationQuiz = {
  title: 'Retrieval-Augmented Generation Discussion',
  id: 'retrieval-augmented-generation-quiz',
  sectionId: 'retrieval-augmented-generation',
  questions: [
    {
      id: 1,
      question:
        'RAG helps address hallucination and knowledge freshness issues in LLMs. However, retrieval introduces its own challenges: irrelevant results polluting context, retrieval latency, and context window limits. Discuss strategies to optimize the retrieval-generation pipeline: chunking, reranking, query expansion, and hybrid search. How do you measure RAG system quality end-to-end?',
      expectedAnswer:
        'Should cover: chunking strategies balancing context preservation and retrieval granularity, two-stage retrieval (fast vector search + slow reranking), query rewriting to improve retrieval, hybrid search combining keywords and vectors, filtering by metadata, relevance thresholding to exclude poor matches, context compression to fit more relevant info, evaluation metrics (retrieval recall, answer accuracy, citation accuracy), human evaluation importance, and iterative improvement cycles.',
    },
    {
      id: 2,
      question:
        'Compare different approaches to maintaining up-to-date knowledge in LLM systems: frequent retraining, continual learning, fine-tuning, and RAG. What are the tradeoffs in cost, freshness, accuracy, and engineering complexity? When is each approach appropriate?',
      expectedAnswer:
        "Should discuss: retraining costs and infrequency (months-years), continual learning's technical challenges and catastrophic forgetting, fine- tuning for domain knowledge but not real- time updates, RAG enabling instant updates without retraining, cost comparison (RAG's inference cost vs training cost), hybrid approaches (fine-tuned model + RAG), knowledge factuality vs reasoning capability, editable neural memory research, and practical recommendations by use case (news vs specialized domain vs general knowledge).",
    },
    {
      id: 3,
      question:
        'Analyze the failure modes of RAG systems: retrieval failures, context window overflow, conflicting information, and attribution issues. For each failure mode, discuss detection strategies and mitigation techniques. How do you build robust RAG systems that gracefully handle edge cases?',
      expectedAnswer:
        'Should cover: empty retrieval results requiring fallback behavior, irrelevant retrieval diluting context, contradictory retrieved documents and resolution strategies, context overflow with long documents requiring summarization, attribution and citation generation for transparency, confidence scoring for retrieval quality, fallback to base model when retrieval fails, user feedback loops for quality improvement, monitoring retrieval and generation metrics separately, and A/B testing RAG variations.',
    },
  ],
};
