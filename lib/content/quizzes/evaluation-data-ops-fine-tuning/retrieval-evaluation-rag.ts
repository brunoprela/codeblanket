/**
 * Discussion questions for Retrieval Evaluation (RAG) section
 */

export const retrievalEvaluationRAGQuiz = [
  {
    id: 'retrieval-eval-rag-q-1',
    question:
      "Your RAG system retrieves top-5 documents for each query. You measure: Retrieval Recall@5 = 78%, but end-to-end answer accuracy = 62%. The gap suggests the LLM isn't effectively using retrieved documents. Design a comprehensive evaluation strategy to diagnose: (1) What types of retrieval failures occur, (2) Why the LLM fails to use good retrievals, (3) How to measure and improve both components.",
    hint: 'Consider breaking down into retrieval metrics (recall, precision, MRR) and generation metrics (faithfulness, relevance). Think about failure modes at each stage.',
    sampleAnswer:
      'This is a comprehensive answer that would be displayed in the UI.',
    keyPoints: [
      '16% gap between retrieval (78%) and end-to-end (62%) indicates generation problems',
      'Decompose evaluation: Retrieval metrics (recall, precision, MRR, NDCG) + generation metrics (faithfulness, context usage)',
      'Failure categories: Retrieval failed (22%), context not used (6%), hallucination (4%), poor ranking (3%), other (2%)',
      'Faithfulness check: Use NLI model to verify answer is entailed by context',
      'Context usage: Check n-gram overlap between answer and retrieved docs',
      'Improvement targets: Better prompting (context usage), NLI verification (hallucination), reranking (ranking), fine-tuning',
      'Continuous monitoring: Log all requests, track user feedback, weekly analysis, trigger re-evaluation if degraded',
    ],
  },
  {
    id: 'retrieval-eval-rag-q-2',
    question:
      'You\'re evaluating two retrieval methods for RAG: (A) Dense retrieval (embedding similarity) with Recall@5 = 75%, (B) BM25 (keyword matching) with Recall@5 = 68%. However, when you measure end-to-end answer accuracy, BM25 achieves 72% while dense retrieval only gets 67%. Why might the "worse" retrieval method produce better final answers? How would you design a hybrid system that gets the best of both?',
    hint: "Consider that recall@k doesn't measure ranking quality or diversity. BM25 might retrieve fewer but more relevant/diverse docs. Think about combining strengths.",
    sampleAnswer:
      'This is a comprehensive answer that would be displayed in the UI.',
    keyPoints: [
      'Paradox: BM25 lower recall (68%) but higher accuracy (72%) vs dense (75% recall, 67% accuracy)',
      'Ranking matters: BM25 puts relevant docs at position 1-2 (MRR 0.68) vs dense at 3-5 (MRR 0.52)',
      'Diversity: BM25 retrieves diverse docs (0.68) vs dense near-duplicates (0.42), helps LLM synthesis',
      'Exact matching: BM25 guarantees keyword overlap (4.1 words) vs dense semantic drift (2.3 words)',
      'Query types: BM25 excels at factual/entity (78%), dense better at conceptual (74%)',
      'Hybrid - RRF: Simple fusion, 82% recall + 76% accuracy (easy to implement)',
      'Hybrid - Reranker: Cross-encoder reranking, 85% recall + 79% accuracy (best performance)',
    ],
  },
  {
    id: 'retrieval-eval-rag-q-3',
    question:
      'Your RAG system works great on your test set (88% accuracy) but degrades in production over time (month 1: 85%, month 3: 78%, month 6: 71%). What are likely causes of this degradation, and how would you design an automated monitoring and alerting system to detect and diagnose issues before they impact users?',
    hint: 'Consider document corpus drift, query distribution shift, model staleness, and infrastructure issues. Think about metrics to track and thresholds to alert on.',
    sampleAnswer:
      'This is a comprehensive answer that would be displayed in the UI.',
    keyPoints: [
      'Degradation causes: Corpus drift (new docs not indexed), query shift (new topics), embedding staleness (OOV terms), infrastructure (slow retrieval)',
      'Corpus drift detection: Monitor indexed vs total docs, find stale embeddings (>90 days), alert if >10% missing',
      'Query shift detection: Compare recent vs baseline using JS divergence, identify novel query types',
      'Embedding staleness: Detect OOV terms, consider updating model if many new vocabulary',
      'Infrastructure monitoring: Track retrieval latency (p95), vector store health, index size growth',
      'Automated alerts: Retrieval score drop >10%, user feedback drop >15%, index staleness >10%, unindexed >5%',
      'Auto-remediation: Trigger re-indexing (incremental daily, full weekly), run diagnostics, escalate critical',
    ],
  },
];
