export const retrievalAugmentedGenerationMC = {
  title: 'Retrieval-Augmented Generation Quiz',
  id: 'retrieval-augmented-generation-mc',
  sectionId: 'retrieval-augmented-generation',
  questions: [
    {
      id: 1,
      question:
        'What is the primary benefit of RAG (Retrieval-Augmented Generation) over fine-tuning?',
      options: [
        'Better model performance',
        'Up-to-date knowledge without retraining',
        'Lower inference cost',
        'Faster response times',
      ],
      correctAnswer: 1,
      explanation:
        'RAG allows updating knowledge by simply updating the retrieval corpus, without expensive retraining. Fine-tuned models have static knowledge from training time. RAG can instantly incorporate new information added to the vector database.',
    },
    {
      id: 2,
      question: 'In a RAG pipeline, what is "reranking"?',
      options: [
        'Sorting the final generated responses',
        'Using a second model to re-score and reorder retrieved documents for relevance',
        'Generating multiple responses and picking the best',
        'Updating the vector index ordering',
      ],
      correctAnswer: 1,
      explanation:
        'Reranking uses a cross-encoder model to rescore retrieved documents after initial vector search. Vector search is fast but approximate; reranking provides higher quality relevance scoring for the top results, often improving final answer quality significantly.',
    },
    {
      id: 3,
      question:
        'What is a common failure mode where retrieved information actually hurts RAG performance?',
      options: [
        'When retrieval is too fast',
        'When irrelevant documents dilute the context',
        'When documents are too short',
        'When using too few retrievals',
      ],
      correctAnswer: 1,
      explanation:
        "Irrelevant retrieved documents can pollute the context, confusing the LLM and leading to worse answers than using no retrieval. It's critical to set relevance thresholds and use reranking to ensure only truly relevant information is included.",
    },
    {
      id: 4,
      question:
        'What chunk size typically provides the best balance for RAG systems?',
      options: [
        '50-100 tokens (sentence-level)',
        '256-512 tokens (paragraph-level)',
        '2000-4000 tokens (section-level)',
        '10000+ tokens (document-level)',
      ],
      correctAnswer: 1,
      explanation:
        'Paragraph-level chunks (256-512 tokens) typically work best—large enough to contain coherent ideas but small enough for precise retrieval. Too small loses context; too large reduces retrieval precision. The optimal size is task and domain-dependent.',
    },
    {
      id: 5,
      question: 'How should you evaluate RAG system quality?',
      options: [
        'Only measure retrieval recall',
        'Only measure final answer accuracy',
        'Measure both retrieval quality (recall/precision) and end-to-end answer quality',
        'Only measure response latency',
      ],
      correctAnswer: 2,
      explanation:
        'RAG evaluation requires both retrieval metrics (did we find relevant docs?) and generation metrics (is the final answer correct?). A system can have perfect retrieval but poor generation, or vice versa. End-to-end metrics plus intermediate metrics enable effective debugging.',
    },
  ],
};
