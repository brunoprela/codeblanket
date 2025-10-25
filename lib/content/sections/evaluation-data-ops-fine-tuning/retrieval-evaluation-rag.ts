/**
 * Retrieval Evaluation (RAG) Section
 * Module 16: Evaluation, Data Operations & Fine-Tuning
 */

export const retrievalEvaluationRAG = {
  id: 'retrieval-evaluation-rag',
  title: 'Retrieval Evaluation (RAG)',
  content: `# Retrieval Evaluation (RAG)

Master evaluating RAG systems with retrieval and generation metrics.

## Overview: RAG Evaluation Challenges

RAG systems have two components to evaluate:
1. **Retrieval**: Did we find the right documents?
2. **Generation**: Did we use them correctly?

**Both must work for RAG to succeed.**

\`\`\`python
class RAGMetrics:
    """
    Key metrics for RAG evaluation:
    
    Retrieval:
    - Precision@K: % of retrieved docs that are relevant
    - Recall@K: % of relevant docs that were retrieved  
    - MRR: Mean reciprocal rank of first relevant doc
    - NDCG: Normalized discounted cumulative gain
    
    Generation:
    - Faithfulness: Output is grounded in retrieved docs
    - Answer Relevance: Output answers the question
    - Context Relevance: Retrieved docs are relevant
    """
    pass
\`\`\`

## Retrieval Metrics

\`\`\`python
from typing import List, Set, Dict, Any
import numpy as np

class RetrievalEvaluator:
    """Evaluate retrieval quality."""
    
    def precision_at_k(
        self,
        retrieved: List[str],
        relevant: Set[str],
        k: int = 5
    ) -> float:
        """
        Precision@K: What % of top K results are relevant?
        
        High precision = few false positives
        """
        top_k = retrieved[:k]
        relevant_retrieved = sum(1 for doc in top_k if doc in relevant)
        return relevant_retrieved / k if k > 0 else 0.0
    
    def recall_at_k(
        self,
        retrieved: List[str],
        relevant: Set[str],
        k: int = 5
    ) -> float:
        """
        Recall@K: What % of all relevant docs did we retrieve in top K?
        
        High recall = few false negatives
        """
        if not relevant:
            return 0.0
        
        top_k = retrieved[:k]
        relevant_retrieved = sum(1 for doc in top_k if doc in relevant)
        return relevant_retrieved / len (relevant)
    
    def mean_reciprocal_rank(
        self,
        retrieved: List[str],
        relevant: Set[str]
    ) -> float:
        """
        MRR: 1 / rank of first relevant document
        
        MRR=1.0: First result is relevant
        MRR=0.5: Second result is relevant
        MRR=0.0: No relevant results
        """
        for i, doc in enumerate (retrieved):
            if doc in relevant:
                return 1.0 / (i + 1)
        return 0.0
    
    def ndcg_at_k(
        self,
        retrieved: List[str],
        relevance_scores: Dict[str, float],
        k: int = 5
    ) -> float:
        """
        NDCG@K: Normalized Discounted Cumulative Gain
        
        Accounts for:
        - Ranking position (earlier is better)
        - Relevance scores (not just binary)
        """
        # DCG: Discounted cumulative gain
        dcg = 0.0
        for i, doc in enumerate (retrieved[:k]):
            relevance = relevance_scores.get (doc, 0.0)
            dcg += relevance / np.log2(i + 2)  # i+2 because i starts at 0
        
        # IDCG: Ideal DCG (perfect ranking)
        ideal_relevances = sorted (relevance_scores.values(), reverse=True)[:k]
        idcg = sum(
            rel / np.log2(i + 2)
            for i, rel in enumerate (ideal_relevances)
        )
        
        # Normalize
        return dcg / idcg if idcg > 0 else 0.0
    
    def evaluate_query(
        self,
        query: str,
        retrieved_docs: List[str],
        relevant_docs: Set[str],
        relevance_scores: Dict[str, float] = None
    ) -> Dict[str, float]:
        """Evaluate single query."""
        
        metrics = {
            'precision@5': self.precision_at_k (retrieved_docs, relevant_docs, k=5),
            'recall@5': self.recall_at_k (retrieved_docs, relevant_docs, k=5),
            'mrr': self.mean_reciprocal_rank (retrieved_docs, relevant_docs)
        }
        
        if relevance_scores:
            metrics['ndcg@5'] = self.ndcg_at_k (retrieved_docs, relevance_scores, k=5)
        
        return metrics

# Usage
evaluator = RetrievalEvaluator()

# Test query
query = "What is machine learning?"
retrieved = ["doc1", "doc5", "doc3", "doc2", "doc9"]  # Retrieved in order
relevant = {"doc1", "doc2", "doc3"}  # Ground truth relevant docs

metrics = evaluator.evaluate_query (query, retrieved, relevant)

print(f"Precision@5: {metrics['precision@5']:.2%}")  # 60% (3/5 relevant)
print(f"Recall@5: {metrics['recall@5']:.2%}")  # 100% (found all 3)
print(f"MRR: {metrics['mrr']:.3f}")  # 1.0 (first result relevant)
\`\`\`

## Generation Metrics for RAG

\`\`\`python
class RAGGenerationEvaluator:
    """Evaluate generation quality in RAG."""
    
    async def faithfulness(
        self,
        question: str,
        answer: str,
        context: str
    ) -> float:
        """
        Faithfulness: Is answer grounded in context?
        
        Uses NLI to check if context entails answer.
        """
        from transformers import pipeline
        
        nli = pipeline("text-classification", model="microsoft/deberta-large-mnli")
        
        # Extract claims from answer
        claims = self._extract_claims (answer)
        
        # Check each claim against context
        supported = 0
        for claim in claims:
            result = nli (f"{context} [SEP] {claim}")
            if result[0]['label'] == 'ENTAILMENT':
                supported += 1
        
        return supported / len (claims) if claims else 0.0
    
    async def answer_relevance(
        self,
        question: str,
        answer: str
    ) -> float:
        """
        Answer Relevance: Does answer address the question?
        
        Uses semantic similarity.
        """
        from sentence_transformers import SentenceTransformer
        from scipy.spatial.distance import cosine
        
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        q_emb = model.encode (question)
        a_emb = model.encode (answer)
        
        similarity = 1 - cosine (q_emb, a_emb)
        
        return similarity
    
    async def context_relevance(
        self,
        question: str,
        contexts: List[str]
    ) -> float:
        """
        Context Relevance: Are retrieved docs relevant to question?
        """
        from sentence_transformers import SentenceTransformer
        from scipy.spatial.distance import cosine
        
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        q_emb = model.encode (question)
        
        relevances = []
        for context in contexts:
            c_emb = model.encode (context)
            sim = 1 - cosine (q_emb, c_emb)
            relevances.append (sim)
        
        return sum (relevances) / len (relevances) if relevances else 0.0
    
    def _extract_claims (self, text: str) -> List[str]:
        """Extract factual claims from text."""
        # Simple: split into sentences
        import re
        sentences = re.split (r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]

# Usage
rag_eval = RAGGenerationEvaluator()

question = "What is the capital of France?"
answer = "The capital of France is Paris, a major European city."
context = "Paris is the capital and largest city of France."

faithfulness = await rag_eval.faithfulness (question, answer, context)
answer_rel = await rag_eval.answer_relevance (question, answer)
context_rel = await rag_eval.context_relevance (question, [context])

print(f"Faithfulness: {faithfulness:.2%}")  # ~100% (grounded in context)
print(f"Answer Relevance: {answer_rel:.2%}")  # ~90% (answers question)
print(f"Context Relevance: {context_rel:.2%}")  # ~95% (context relevant)
\`\`\`

## RAGAS Framework

\`\`\`python
# RAGAS: RAG Assessment framework
# pip install ragas

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_relevancy,
    context_recall,
    context_precision
)
from datasets import Dataset

class RAGASEvaluator:
    """Use RAGAS for comprehensive RAG evaluation."""
    
    def prepare_dataset(
        self,
        test_cases: List[Dict]
    ) -> Dataset:
        """
        Prepare dataset for RAGAS.
        
        Each test case needs:
        - question: The query
        - answer: Generated answer
        - contexts: Retrieved documents
        - ground_truths: Reference answers (optional)
        """
        data = {
            'question': [tc['question'] for tc in test_cases],
            'answer': [tc['answer'] for tc in test_cases],
            'contexts': [tc['contexts'] for tc in test_cases],
            'ground_truths': [tc.get('ground_truth', ') for tc in test_cases]
        }
        
        return Dataset.from_dict (data)
    
    async def evaluate_rag_system(
        self,
        test_cases: List[Dict]
    ) -> Dict[str, float]:
        """Evaluate RAG system with RAGAS metrics."""
        
        dataset = self.prepare_dataset (test_cases)
        
        # Evaluate with all metrics
        result = evaluate(
            dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_relevancy,
                context_recall,
                context_precision
            ]
        )
        
        return dict (result)

# Usage
ragas_eval = RAGASEvaluator()

test_cases = [
    {
        'question': 'What is the capital of France?',
        'answer': 'Paris is the capital of France.',
        'contexts': ['Paris is the capital and largest city of France.'],
        'ground_truth': 'Paris'
    },
    # ... more test cases
]

scores = await ragas_eval.evaluate_rag_system (test_cases)

print("RAG System Evaluation:")
print(f"  Faithfulness: {scores['faithfulness']:.2%}")
print(f"  Answer Relevancy: {scores['answer_relevancy']:.2%}")
print(f"  Context Relevancy: {scores['context_relevancy']:.2%}")
print(f"  Context Recall: {scores['context_recall']:.2%}")
print(f"  Context Precision: {scores['context_precision']:.2%}")
\`\`\`

## End-to-End RAG Evaluation

\`\`\`python
class ComprehensiveRAGEvaluator:
    """Complete RAG evaluation pipeline."""
    
    def __init__(self):
        self.retrieval_eval = RetrievalEvaluator()
        self.generation_eval = RAGGenerationEvaluator()
    
    async def evaluate_rag_pipeline(
        self,
        test_dataset: List[Dict]
    ) -> Dict[str, Any]:
        """
        Evaluate complete RAG pipeline.
        
        test_dataset: [
            {
                'question': str,
                'relevant_doc_ids': Set[str],
                'ground_truth_answer': str
            }
        ]
        """
        
        all_metrics = []
        
        for test_case in test_dataset:
            question = test_case['question']
            
            # 1. Retrieval
            retrieved_docs = await self._retrieve (question)
            retrieved_ids = [doc['id'] for doc in retrieved_docs]
            relevant_ids = test_case['relevant_doc_ids']
            
            retrieval_metrics = self.retrieval_eval.evaluate_query(
                question,
                retrieved_ids,
                relevant_ids
            )
            
            # 2. Generation
            context = "\\n".join([doc['text'] for doc in retrieved_docs])
            answer = await self._generate (question, context)
            
            faithfulness = await self.generation_eval.faithfulness(
                question,
                answer,
                context
            )
            
            answer_rel = await self.generation_eval.answer_relevance(
                question,
                answer
            )
            
            # 3. End-to-end quality
            # Compare generated answer to ground truth
            ground_truth = test_case['ground_truth_answer']
            exact_match = self._exact_match (answer, ground_truth)
            f1 = self._f1_score (answer, ground_truth)
            
            all_metrics.append({
                **retrieval_metrics,
                'faithfulness': faithfulness,
                'answer_relevance': answer_rel,
                'exact_match': exact_match,
                'f1_score': f1
            })
        
        # Aggregate
        aggregated = {}
        for key in all_metrics[0].keys():
            values = [m[key] for m in all_metrics]
            aggregated[key] = sum (values) / len (values)
        
        return aggregated
    
    async def _retrieve (self, query: str) -> List[Dict]:
        """Your retrieval implementation."""
        # Placeholder
        return [
            {'id': 'doc1', 'text': 'Retrieved document 1'},
            {'id': 'doc2', 'text': 'Retrieved document 2'}
        ]
    
    async def _generate (self, query: str, context: str) -> str:
        """Your generation implementation."""
        # Placeholder
        return "Generated answer using context"
    
    def _exact_match (self, pred: str, truth: str) -> float:
        """Exact match scoring."""
        return 1.0 if pred.strip().lower() == truth.strip().lower() else 0.0
    
    def _f1_score (self, pred: str, truth: str) -> float:
        """Token-level F1."""
        pred_tokens = set (pred.lower().split())
        truth_tokens = set (truth.lower().split())
        
        if not pred_tokens or not truth_tokens:
            return 0.0
        
        common = pred_tokens & truth_tokens
        if not common:
            return 0.0
        
        precision = len (common) / len (pred_tokens)
        recall = len (common) / len (truth_tokens)
        
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1

# Usage
comprehensive_eval = ComprehensiveRAGEvaluator()

results = await comprehensive_eval.evaluate_rag_pipeline (test_dataset)

print("RAG Pipeline Evaluation:")
print(f"\\nRetrieval:")
print(f"  Precision@5: {results['precision@5']:.2%}")
print(f"  Recall@5: {results['recall@5']:.2%}")
print(f"  MRR: {results['mrr']:.3f}")

print(f"\\nGeneration:")
print(f"  Faithfulness: {results['faithfulness']:.2%}")
print(f"  Answer Relevance: {results['answer_relevance']:.2%}")

print(f"\\nEnd-to-End:")
print(f"  Exact Match: {results['exact_match']:.2%}")
print(f"  F1 Score: {results['f1_score']:.2%}")
\`\`\`

## Common RAG Failure Modes

\`\`\`python
class RAGDiagnostics:
    """Diagnose common RAG issues."""
    
    def diagnose(
        self,
        retrieval_precision: float,
        retrieval_recall: float,
        faithfulness: float,
        answer_relevance: float
    ) -> List[str]:
        """Identify likely issues."""
        
        issues = []
        
        # Low precision: Retrieving irrelevant docs
        if retrieval_precision < 0.5:
            issues.append("Retrieval precision low - getting irrelevant documents")
            issues.append("  Fix: Improve embedding model, better chunking, add reranker")
        
        # Low recall: Missing relevant docs
        if retrieval_recall < 0.7:
            issues.append("Retrieval recall low - missing relevant documents")
            issues.append("  Fix: Increase k, improve indexing, check chunking")
        
        # Low faithfulness: Hallucinating
        if faithfulness < 0.8:
            issues.append("Low faithfulness - model hallucinating beyond context")
            issues.append("  Fix: Stronger prompt, fine-tune for faithfulness, add citation requirement")
        
        # Low answer relevance: Not answering question
        if answer_relevance < 0.7:
            issues.append("Answer not relevant to question")
            issues.append("  Fix: Check if context contains answer, improve generation prompt")
        
        # Both retrieval and generation good but still fails
        if (retrieval_precision > 0.8 and retrieval_recall > 0.8 and 
            faithfulness > 0.8 but answer_relevance < 0.7):
            issues.append("Retrieval good but answer quality poor")
            issues.append("  Fix: Improve generation model, better context formatting")
        
        return issues if issues else ["No major issues detected!"]

# Usage
diagnostics = RAGDiagnostics()

issues = diagnostics.diagnose(
    retrieval_precision=0.4,  # Low!
    retrieval_recall=0.9,     # Good
    faithfulness=0.9,         # Good
    answer_relevance=0.6      # Low
)

print("Diagnosed Issues:")
for issue in issues:
    print(f"  {issue}")
\`\`\`

## Production Checklist

✅ **Retrieval Evaluation**
- [ ] Precision@K measured
- [ ] Recall@K measured
- [ ] MRR/NDCG tracked
- [ ] Test set with ground truth relevance

✅ **Generation Evaluation**
- [ ] Faithfulness checked
- [ ] Answer relevance verified
- [ ] Context relevance validated
- [ ] Hallucination monitored

✅ **End-to-End**
- [ ] Complete pipeline tested
- [ ] Failure modes diagnosed
- [ ] Improvements prioritized
- [ ] Continuous monitoring enabled

## Next Steps

You now understand RAG evaluation. Next, learn:
- Multi-modal evaluation
- Continuous evaluation & monitoring
- Building complete evaluation platforms
`,
};
