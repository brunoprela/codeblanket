/**
 * Multiple choice questions for Retrieval Evaluation (RAG) section
 */

export const retrievalEvaluationRAGMultipleChoice = [
  {
    id: 'retrieval-eval-rag-mc-1',
    question:
      'Your RAG system has Recall@5 = 80% but end-to-end answer accuracy = 65%. What is the MOST likely bottleneck?',
    options: [
      'Retrieval is failing (need better embeddings)',
      'Retrieved docs are not ranked well (need reranking)',
      'LLM is not effectively using retrieved context (generation problem)',
      'Not enough documents in corpus',
    ],
    correctAnswer: 2,
    explanation:
      "Option C (generation problem) is most likely. 80% retrieval recall means the right document IS in top-5. If it were a retrieval problem (Option A), recall would be low. If ranking were the issue (Option B), you'd see high recall but low MRR. The 15% gap (80% retrieval vs 65% accuracy) indicates the LLM isn't using the retrieved documents effectively. Common causes: LLM ignores context and uses pre-trained knowledge, context is too long and LLM loses focus, prompt doesn't emphasize using context, LLM hallucinates despite having correct info. Solutions: Improve prompt (\"Answer ONLY using context below\"), add faithfulness verification, rerank to put best doc first, fine-tune LLM on RAG examples. Option D is unlikely to cause this specific gap pattern.",
  },
  {
    id: 'retrieval-eval-rag-mc-2',
    question:
      'You measure RAG retrieval with these metrics: Recall@5 = 75%, Precision@5 = 40%, MRR = 0.5. What does this tell you?',
    options: [
      'Retrieval is working perfectly',
      'You find relevant docs, but too many irrelevant ones are mixed in',
      "You're missing most relevant documents",
      'The metrics are contradictory and invalid',
    ],
    correctAnswer: 1,
    explanation:
      'Option B is correct. Interpretation: Recall@5 = 75% means in 75% of cases, at least ONE relevant doc is in top-5 → Retrieval is finding relevant docs. Precision@5 = 40% means only 40% of retrieved docs are relevant → 2 out of 5 docs are relevant, 3 are noise. MRR = 0.5 means relevant doc averages position 2 (1/0.5) → Not at the top, but not buried. Problem: Too many irrelevant docs dilute the signal. LLM gets confused by noise. Solution: Better filtering, reranking, or stricter similarity threshold. Option A is wrong (precision is too low). Option C is wrong (recall is decent). Option D is wrong (metrics are consistent and make sense together).',
  },
  {
    id: 'retrieval-eval-rag-mc-3',
    question:
      'Your RAG system works well on test set (85% accuracy) but degrades in production after 3 months (70% accuracy). What is MOST likely the cause?',
    options: [
      'LLM model was updated by the API provider',
      "Document corpus has grown/changed but embeddings weren't updated",
      'Users are asking harder questions',
      'Network latency increased',
    ],
    correctAnswer: 1,
    explanation:
      'Option B (corpus drift without reindexing) is most common cause of time-based degradation. What happens: New documents added to corpus, documents updated with new information, but embeddings not regenerated. New content not in vector index → retrieval misses recent information. Users ask about new topics → system fails. This is classic "index staleness". Solution: Schedule regular reindexing (daily for updated docs, weekly full), monitor indexed vs total doc count, set up alerts when >5% docs are unindexed. Option A is possible but less common (API providers maintain compatibility). Option C doesn\'t explain why test set still works. Option D wouldn\'t cause accuracy drop, just slower responses. Key lesson: RAG systems need continuous maintenance as corpus evolves.',
  },
  {
    id: 'retrieval-eval-rag-mc-4',
    question:
      'You want to evaluate if your RAG system is "faithful" (answers grounded in retrieved docs). What is the BEST automated method?',
    options: [
      'Check if answer text appears verbatim in retrieved docs',
      'Measure BLEU score between answer and retrieved docs',
      'Use NLI model to check if context entails answer',
      'Measure cosine similarity between answer and doc embeddings',
    ],
    correctAnswer: 2,
    explanation:
      'Option C (NLI - Natural Language Inference) is best for faithfulness. How it works: NLI models (e.g., DeBERTa-MNLI) predict if premise entails hypothesis. Input: "Context: [retrieved docs] Hypothesis: [generated answer]". Output: ENTAILMENT (answer supported by context), NEUTRAL (can\'t determine), CONTRADICTION (answer contradicts context). Faithful = ENTAILMENT. Why other options fail: Option A (verbatim match) is too strict—faithful paraphrasing would fail. Option B (BLEU) measures word overlap, not semantic entailment. Option D (cosine sim) measures similarity, not logical entailment (similar ≠ supported). Example: Context: "Paris is capital of France." Answer: "France\'s capital city is Paris." → Verbatim: ❌, BLEU: Medium, Cosine: High, NLI: ✅ ENTAILMENT. NLI is the gold standard for faithfulness evaluation.',
  },
  {
    id: 'retrieval-eval-rag-mc-5',
    question:
      'Dense retrieval (embeddings) gets 72% accuracy, BM25 (keywords) gets 68% accuracy. You combine them with RRF (Reciprocal Rank Fusion) and get 79% accuracy. Why did hybrid work so well?',
    options: [
      'RRF always improves any two systems',
      'Dense and BM25 capture different signals, errors are uncorrelated',
      'The math of RRF is optimal',
      "It's just luck from this specific test set",
    ],
    correctAnswer: 1,
    explanation:
      'Option B (complementary signals, uncorrelated errors) is correct. Why hybrid works: Dense captures semantic similarity ("car" matches "vehicle"), BM25 captures exact keywords ("model T" matches exactly). They fail on different examples: Dense fails on rare entities (not in embedding training), BM25 fails on paraphrases (no keyword overlap). Hybrid catches what each misses: Query: "automobile invented by Ford" → BM25 finds "Ford" docs, Dense finds "car invention" docs. Union + reranking = best of both. RRF combines rankings: If both rank doc high → high score. If only one ranks high → medium score. Result: Docs both methods like rise to top, unique finds from each method fill rest. Option A is wrong (hybrid doesn\'t always help if errors are correlated). Option C is wrong (RRF is simple heuristic, not optimal). Option D is wrong (consistent improvement across datasets). Key principle: Ensemble diverse retrievers for best results.',
  },
];
