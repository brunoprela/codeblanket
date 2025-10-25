export const ragEvaluation = {
  title: 'RAG Evaluation',
  content: `
# RAG Evaluation

## Introduction

"You can't improve what you don't measure." Evaluating RAG systems is critical for understanding performance, identifying problems, and driving improvements. Unlike traditional software, RAG systems have complex failure modes that require sophisticated evaluation strategies.

In this comprehensive section, we'll explore retrieval metrics, generation quality assessment, the RAGAS framework, end-to-end evaluation, and building production evaluation pipelines.

## Why RAG Evaluation is Challenging

RAG evaluation is complex because it involves multiple stages:

\`\`\`python
# RAG Pipeline has multiple evaluation points:
query = "What is machine learning?"

# 1. Retrieval Quality
retrieved_docs = retriever.search (query)  # Are these relevant?

# 2. Context Quality  
context = format_context (retrieved_docs)  # Is context well-formed?

# 3. Generation Quality
answer = llm.generate (context, query)  # Is answer accurate?

# 4. End-to-End Quality
# Does the complete system meet user needs?
\`\`\`

### Evaluation Dimensions

1. **Retrieval Quality**: Are retrieved documents relevant?
2. **Answer Relevance**: Does answer address the question?
3. **Faithfulness**: Is answer grounded in retrieved context?
4. **Context Precision**: Are retrieved docs ranked properly?
5. **Answer Correctness**: Is answer factually correct?

## Retrieval Metrics

Measure how well the retrieval system finds relevant documents:

\`\`\`python
from typing import List, Set
import numpy as np

class RetrievalMetrics:
    """
    Calculate retrieval quality metrics.
    """
    
    @staticmethod
    def precision_at_k(
        retrieved: List[str],
        relevant: Set[str],
        k: int = 5
    ) -> float:
        """
        Calculate Precision@K.
        
        Precision@K = (# relevant docs in top K) / K
        
        Args:
            retrieved: List of retrieved document IDs (in rank order)
            relevant: Set of relevant document IDs
            k: Number of top results to consider
        
        Returns:
            Precision score (0-1)
        """
        top_k = retrieved[:k]
        relevant_in_k = sum(1 for doc_id in top_k if doc_id in relevant)
        return relevant_in_k / k if k > 0 else 0.0
    
    @staticmethod
    def recall_at_k(
        retrieved: List[str],
        relevant: Set[str],
        k: int = 5
    ) -> float:
        """
        Calculate Recall@K.
        
        Recall@K = (# relevant docs in top K) / (# total relevant docs)
        
        Args:
            retrieved: List of retrieved document IDs
            relevant: Set of relevant document IDs
            k: Number of top results to consider
        
        Returns:
            Recall score (0-1)
        """
        if len (relevant) == 0:
            return 0.0
        
        top_k = retrieved[:k]
        relevant_in_k = sum(1 for doc_id in top_k if doc_id in relevant)
        return relevant_in_k / len (relevant)
    
    @staticmethod
    def mean_reciprocal_rank(
        retrieved: List[str],
        relevant: Set[str]
    ) -> float:
        """
        Calculate Mean Reciprocal Rank (MRR).
        
        MRR = 1 / (rank of first relevant document)
        
        Args:
            retrieved: List of retrieved document IDs
            relevant: Set of relevant document IDs
        
        Returns:
            MRR score
        """
        for rank, doc_id in enumerate (retrieved, start=1):
            if doc_id in relevant:
                return 1.0 / rank
        return 0.0
    
    @staticmethod
    def ndcg_at_k(
        retrieved: List[str],
        relevance_scores: dict,
        k: int = 5
    ) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain (NDCG@K).
        
        Args:
            retrieved: List of retrieved document IDs
            relevance_scores: Dict of doc_id -> relevance score (0-3)
            k: Number of top results
        
        Returns:
            NDCG score (0-1)
        """
        def dcg (scores):
            """Calculate DCG."""
            return sum(
                score / np.log2(rank + 2)
                for rank, score in enumerate (scores)
            )
        
        # Get relevance scores for retrieved docs
        retrieved_scores = [
            relevance_scores.get (doc_id, 0)
            for doc_id in retrieved[:k]
        ]
        
        # Calculate DCG
        dcg_score = dcg (retrieved_scores)
        
        # Calculate ideal DCG (best possible ranking)
        ideal_scores = sorted (relevance_scores.values(), reverse=True)[:k]
        idcg_score = dcg (ideal_scores)
        
        # Normalize
        if idcg_score == 0:
            return 0.0
        
        return dcg_score / idcg_score


# Example usage
metrics = RetrievalMetrics()

# Simulated retrieval results
retrieved = ["doc1", "doc3", "doc5", "doc2", "doc4"]
relevant = {"doc1", "doc2", "doc4", "doc7"}

# Calculate metrics
precision = metrics.precision_at_k (retrieved, relevant, k=5)
recall = metrics.recall_at_k (retrieved, relevant, k=5)
mrr = metrics.mean_reciprocal_rank (retrieved, relevant)

print(f"Precision@5: {precision:.3f}")
print(f"Recall@5: {recall:.3f}")
print(f"MRR: {mrr:.3f}")

# NDCG example
relevance_scores = {
    "doc1": 3, "doc2": 2, "doc3": 1,
    "doc4": 2, "doc5": 0, "doc7": 3
}
ndcg = metrics.ndcg_at_k (retrieved, relevance_scores, k=5)
print(f"NDCG@5: {ndcg:.3f}")
\`\`\`

## RAGAS Framework

RAGAS (Retrieval Augmented Generation Assessment) provides comprehensive RAG evaluation:

\`\`\`python
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_precision,
    context_recall,
)
from datasets import Dataset

class RAGASEvaluator:
    """
    Evaluate RAG systems using RAGAS framework.
    """
    
    def __init__(self):
        self.metrics = [
            answer_relevancy,
            faithfulness,
            context_precision,
            context_recall,
        ]
    
    def evaluate(
        self,
        questions: List[str],
        answers: List[str],
        contexts: List[List[str]],
        ground_truths: List[str] = None
    ) -> dict:
        """
        Evaluate RAG system using RAGAS.
        
        Args:
            questions: List of questions
            answers: List of generated answers
            contexts: List of retrieved contexts (list of docs per question)
            ground_truths: Optional ground truth answers
        
        Returns:
            Evaluation results with scores
        """
        # Prepare data in RAGAS format
        data = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
        }
        
        if ground_truths:
            data["ground_truths"] = ground_truths
        
        # Create dataset
        dataset = Dataset.from_dict (data)
        
        # Evaluate
        result = evaluate(
            dataset,
            metrics=self.metrics,
        )
        
        return result


# Example usage
evaluator = RAGASEvaluator()

# Sample data
questions = [
    "What is machine learning?",
    "How does gradient descent work?"
]

answers = [
    "Machine learning is a branch of AI that enables computers to learn from data.",
    "Gradient descent is an optimization algorithm that iteratively adjusts parameters."
]

contexts = [
    [
        "Machine learning is a subset of artificial intelligence...",
        "ML algorithms learn patterns from data..."
    ],
    [
        "Gradient descent minimizes loss functions...",
        "The algorithm uses derivatives to find minima..."
    ]
]

ground_truths = [
    "Machine learning is AI that learns from data",
    "Gradient descent optimizes using gradients"
]

# Evaluate
results = evaluator.evaluate (questions, answers, contexts, ground_truths)
print(results)
\`\`\`

### RAGAS Metrics Explained

**1. Answer Relevancy**
- Measures if answer addresses the question
- Uses LLM to generate questions from answer, compares to original

**2. Faithfulness**
- Measures if answer is grounded in retrieved context
- Checks if claims in answer can be verified from context

**3. Context Precision**
- Measures if relevant docs are ranked higher
- Assesses ranking quality

**4. Context Recall**
- Measures if all necessary info was retrieved
- Checks if ground truth can be derived from context

## LLM-as-Judge Evaluation

Use LLMs to evaluate RAG outputs:

\`\`\`python
from openai import OpenAI

client = OpenAI()

class LLMJudge:
    """
    Use LLM to evaluate RAG system outputs.
    """
    
    def __init__(self, model: str = "gpt-4"):
        self.model = model
        self.client = OpenAI()
    
    def evaluate_answer_relevance(
        self,
        question: str,
        answer: str
    ) -> dict:
        """
        Evaluate if answer is relevant to question.
        
        Args:
            question: User question
            answer: Generated answer
        
        Returns:
            Score and reasoning
        """
        prompt = f"""Evaluate if the answer is relevant to the question.

Question: {question}

Answer: {answer}

Rate the relevance on a scale of 1-5:
1 - Completely irrelevant
2 - Slightly relevant
3 - Moderately relevant
4 - Relevant
5 - Highly relevant and comprehensive

Provide your rating and brief reasoning."""
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an expert evaluator of question-answering systems."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        
        content = response.choices[0].message.content
        
        # Parse score (simplified)
        score = self._extract_score (content)
        
        return {
            "score": score,
            "reasoning": content,
            "max_score": 5
        }
    
    def evaluate_faithfulness(
        self,
        answer: str,
        context: str
    ) -> dict:
        """
        Evaluate if answer is faithful to context.
        
        Args:
            answer: Generated answer
            context: Retrieved context
        
        Returns:
            Faithfulness score and analysis
        """
        prompt = f"""Evaluate if the answer is faithful to the provided context.

Context:
{context}

Answer:
{answer}

Rate faithfulness on a scale of 1-5:
1 - Contains significant hallucinations
2 - Mostly accurate but some unsupported claims
3 - Moderately faithful
4 - Faithful with minor issues
5 - Completely faithful, all claims supported

Provide your rating and identify any unsupported claims."""
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an expert at detecting hallucinations and verifying factual accuracy."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        
        content = response.choices[0].message.content
        score = self._extract_score (content)
        
        return {
            "score": score,
            "analysis": content,
            "max_score": 5
        }
    
    def _extract_score (self, text: str) -> float:
        """Extract numeric score from LLM response."""
        import re
        
        # Look for patterns like "Score: 4" or "Rating: 4/5"
        patterns = [
            r'(?:score|rating):\s*(\d+(?:\.\d+)?)',
            r'(\d+)\s*(?:/\s*5|out of 5)',
            r'^(\d+(?:\.\d+)?)\s*[-:]'
        ]
        
        for pattern in patterns:
            match = re.search (pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                return float (match.group(1))
        
        return 0.0


# Example usage
judge = LLMJudge()

question = "What is the capital of France?"
answer = "The capital of France is Paris, which is known for the Eiffel Tower."
context = "Paris is the capital and largest city of France. It is located on the Seine River."

# Evaluate relevance
relevance = judge.evaluate_answer_relevance (question, answer)
print(f"Relevance Score: {relevance['score']}/5")
print(f"Reasoning: {relevance['reasoning'][:100]}...\n")

# Evaluate faithfulness
faithfulness = judge.evaluate_faithfulness (answer, context)
print(f"Faithfulness Score: {faithfulness['score']}/5")
print(f"Analysis: {faithfulness['analysis'][:100]}...")
\`\`\`

## End-to-End Evaluation Pipeline

Complete evaluation system for production:

\`\`\`python
from typing import List, Dict
import time

class RAGEvaluationPipeline:
    """
    Complete RAG evaluation pipeline.
    """
    
    def __init__(self):
        self.retrieval_metrics = RetrievalMetrics()
        self.llm_judge = LLMJudge()
    
    def evaluate_system(
        self,
        test_cases: List[Dict]
    ) -> Dict:
        """
        Evaluate complete RAG system.
        
        Args:
            test_cases: List of test cases with:
                - question: User question
                - retrieved_docs: Retrieved document IDs
                - relevant_docs: Ground truth relevant docs
                - answer: Generated answer
                - context: Retrieved context text
                - ground_truth_answer: Expected answer (optional)
        
        Returns:
            Comprehensive evaluation results
        """
        results = {
            "retrieval": {},
            "generation": {},
            "end_to_end": {},
            "per_query": []
        }
        
        precision_scores = []
        recall_scores = []
        mrr_scores = []
        relevance_scores = []
        faithfulness_scores = []
        
        for i, test_case in enumerate (test_cases):
            print(f"Evaluating test case {i+1}/{len (test_cases)}...")
            
            # Evaluate retrieval
            precision = self.retrieval_metrics.precision_at_k(
                test_case["retrieved_docs"],
                set (test_case["relevant_docs"]),
                k=5
            )
            
            recall = self.retrieval_metrics.recall_at_k(
                test_case["retrieved_docs"],
                set (test_case["relevant_docs"]),
                k=5
            )
            
            mrr = self.retrieval_metrics.mean_reciprocal_rank(
                test_case["retrieved_docs"],
                set (test_case["relevant_docs"])
            )
            
            # Evaluate generation
            relevance = self.llm_judge.evaluate_answer_relevance(
                test_case["question"],
                test_case["answer"]
            )
            
            faithfulness = self.llm_judge.evaluate_faithfulness(
                test_case["answer"],
                test_case["context"]
            )
            
            # Store per-query results
            query_results = {
                "question": test_case["question"],
                "precision@5": precision,
                "recall@5": recall,
                "mrr": mrr,
                "answer_relevance": relevance["score"] / 5,
                "faithfulness": faithfulness["score"] / 5,
            }
            
            results["per_query"].append (query_results)
            
            # Accumulate for averages
            precision_scores.append (precision)
            recall_scores.append (recall)
            mrr_scores.append (mrr)
            relevance_scores.append (relevance["score"] / 5)
            faithfulness_scores.append (faithfulness["score"] / 5)
        
        # Calculate aggregate metrics
        results["retrieval"] = {
            "avg_precision@5": np.mean (precision_scores),
            "avg_recall@5": np.mean (recall_scores),
            "avg_mrr": np.mean (mrr_scores),
        }
        
        results["generation"] = {
            "avg_answer_relevance": np.mean (relevance_scores),
            "avg_faithfulness": np.mean (faithfulness_scores),
        }
        
        # Overall score
        results["end_to_end"] = {
            "overall_score": np.mean([
                results["retrieval"]["avg_precision@5"],
                results["retrieval"]["avg_recall@5"],
                results["generation"]["avg_answer_relevance"],
                results["generation"]["avg_faithfulness"],
            ])
        }
        
        return results
    
    def print_report (self, results: Dict):
        """Print evaluation report."""
        print("\\n" + "="*60)
        print("RAG SYSTEM EVALUATION REPORT")
        print("="*60)
        
        print("\\n📊 RETRIEVAL METRICS")
        print(f"  Precision@5: {results['retrieval']['avg_precision@5']:.3f}")
        print(f"  Recall@5: {results['retrieval']['avg_recall@5']:.3f}")
        print(f"  MRR: {results['retrieval']['avg_mrr']:.3f}")
        
        print("\\n💬 GENERATION METRICS")
        print(f"  Answer Relevance: {results['generation']['avg_answer_relevance']:.3f}")
        print(f"  Faithfulness: {results['generation']['avg_faithfulness']:.3f}")
        
        print("\\n🎯 OVERALL")
        print(f"  Overall Score: {results['end_to_end']['overall_score']:.3f}")
        print("="*60)


# Example usage
pipeline = RAGEvaluationPipeline()

# Sample test cases
test_cases = [
    {
        "question": "What is machine learning?",
        "retrieved_docs": ["doc1", "doc3", "doc5"],
        "relevant_docs": ["doc1", "doc3"],
        "answer": "Machine learning is a branch of AI.",
        "context": "Machine learning is a field of artificial intelligence...",
    },
    # Add more test cases...
]

# Run evaluation
results = pipeline.evaluate_system (test_cases)
pipeline.print_report (results)
\`\`\`

## A/B Testing RAG Systems

Compare different RAG configurations:

\`\`\`python
from scipy import stats

class RAGABTest:
    """
    A/B test different RAG configurations.
    """
    
    def __init__(self):
        self.evaluation_pipeline = RAGEvaluationPipeline()
    
    def compare_systems(
        self,
        system_a_results: Dict,
        system_b_results: Dict,
        metric: str = "overall_score"
    ) -> Dict:
        """
        Compare two RAG systems statistically.
        
        Args:
            system_a_results: Results from system A
            system_b_results: Results from system B
            metric: Metric to compare
        
        Returns:
            Comparison results with statistical significance
        """
        # Extract scores for the metric
        scores_a = [
            q[metric] for q in system_a_results["per_query"]
            if metric in q
        ]
        
        scores_b = [
            q[metric] for q in system_b_results["per_query"]
            if metric in q
        ]
        
        # Calculate statistics
        mean_a = np.mean (scores_a)
        mean_b = np.mean (scores_b)
        std_a = np.std (scores_a)
        std_b = np.std (scores_b)
        
        # Perform t-test
        t_statistic, p_value = stats.ttest_ind (scores_a, scores_b)
        
        # Determine winner
        if p_value < 0.05:  # Statistically significant
            winner = "System A" if mean_a > mean_b else "System B"
            significant = True
        else:
            winner = "No significant difference"
            significant = False
        
        return {
            "metric": metric,
            "system_a_mean": mean_a,
            "system_b_mean": mean_b,
            "system_a_std": std_a,
            "system_b_std": std_b,
            "difference": mean_b - mean_a,
            "percent_improvement": ((mean_b - mean_a) / mean_a * 100) if mean_a > 0 else 0,
            "p_value": p_value,
            "significant": significant,
            "winner": winner
        }
\`\`\`

## Creating Evaluation Datasets

Build high-quality evaluation datasets:

\`\`\`python
class EvaluationDatasetBuilder:
    """
    Build evaluation datasets for RAG systems.
    """
    
    def __init__(self):
        self.client = OpenAI()
    
    def generate_questions_from_docs(
        self,
        documents: List[str],
        num_questions: int = 5
    ) -> List[Dict]:
        """
        Generate questions from documents.
        
        Args:
            documents: Source documents
            num_questions: Questions per document
        
        Returns:
            List of generated questions with metadata
        """
        questions = []
        
        for doc_id, doc in enumerate (documents):
            prompt = f"""Generate {num_questions} diverse questions that can be answered using this document:

Document:
{doc[:500]}...

Generate questions of varying difficulty that test understanding of the content."""
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )
            
            content = response.choices[0].message.content
            
            # Parse questions (simplified)
            doc_questions = [
                line.strip()
                for line in content.split('\\n')
                if line.strip() and any (line.strip().endswith (c) for c in '?.')
            ]
            
            for q in doc_questions[:num_questions]:
                questions.append({
                    "question": q,
                    "source_doc_id": doc_id,
                    "relevant_docs": [doc_id]
                })
        
        return questions
\`\`\`

## Best Practices

### Evaluation Checklist

✅ **Retrieval Evaluation**
- Measure precision and recall
- Track MRR and NDCG
- Monitor coverage (% queries with results)

✅ **Generation Evaluation**
- Answer relevance
- Faithfulness/groundedness
- Answer completeness

✅ **End-to-End Evaluation**
- User satisfaction (if possible)
- Task completion rate
- Response time

### Continuous Evaluation

\`\`\`python
# Monitor RAG quality in production
class ContinuousEvaluator:
    """
    Continuously evaluate RAG system in production.
    """
    
    def log_query (self, query_data: Dict):
        """Log query for later evaluation."""
        # Store: query, results, user feedback
        pass
    
    def sample_for_evaluation (self, sample_rate: float = 0.1):
        """Sample queries for detailed evaluation."""
        pass
    
    def generate_weekly_report (self):
        """Generate evaluation report."""
        pass
\`\`\`

## Summary

RAG evaluation requires multi-dimensional assessment:

- **Retrieval Metrics**: Precision, Recall, MRR, NDCG
- **RAGAS Framework**: Comprehensive RAG-specific evaluation
- **LLM-as-Judge**: Flexible quality assessment
- **End-to-End Pipeline**: Complete system evaluation
- **A/B Testing**: Compare configurations
- **Continuous Monitoring**: Track quality over time

**Key Takeaway:** Build evaluation into your RAG system from day one. You can't improve what you don't measure.

**Production Pattern:**
1. Create evaluation dataset early
2. Measure baseline performance
3. Iterate and re-evaluate
4. A/B test improvements
5. Monitor continuously in production
`,
};
