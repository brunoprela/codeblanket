/**
 * LLM Output Evaluation Section
 * Module 16: Evaluation, Data Operations & Fine-Tuning
 */

export const llmOutputEvaluation = {
  id: 'llm-output-evaluation',
  title: 'LLM Output Evaluation',
  content: `# LLM Output Evaluation

Master specific metrics and techniques for evaluating LLM outputs including accuracy, semantic similarity, and factuality checking.

## Overview: The Challenge of Evaluating LLM Outputs

Unlike traditional ML models with clear metrics (accuracy, F1), **LLM outputs are open-ended text**. You can't simply check if \`output == expected_output\`.

### The Problem

\`\`\`python
# Input: "Summarize photosynthesis"

# Expected: "Plants use sunlight to make food from CO2 and water"
# Model A: "Photosynthesis is the process where plants convert light energy into chemical energy"
# Model B: "Plants make glucose using chlorophyll, sunlight, carbon dioxide and water"

# Which is "correct"? Both? Neither? How do you score them?
\`\`\`

### Types of LLM Tasks Need Different Metrics

| Task | Challenges | Metrics |
|------|-----------|---------|
| **Summarization** | Multiple valid summaries, length vs detail trade-off | ROUGE, semantic similarity, factuality |
| **Question Answering** | Partial credit, phrasing variations | F1, exact match, semantic similarity |
| **Code Generation** | Syntax, logic, efficiency all matter | Pass@k, execution success, test coverage |
| **Translation** | Cultural nuance, multiple translations | BLEU, COMET, human evaluation |
| **Creative Writing** | Highly subjective | Style, coherence, engagement (mostly human) |

## Classic NLP Metrics

### 1. ROUGE (Recall-Oriented Understudy for Gisting Evaluation)

Measures overlap between generated and reference text. **Common for summarization.**

\`\`\`python
from rouge_score import rouge_scorer

class ROUGEEvaluator:
    """Evaluate using ROUGE metrics."""
    
    def __init__(self):
        # ROUGE-N: N-gram overlap
        # ROUGE-L: Longest common subsequence
        self.scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'],
            use_stemmer=True
        )
    
    def evaluate (self, prediction: str, reference: str) -> Dict[str, float]:
        """Calculate ROUGE scores."""
        scores = self.scorer.score (reference, prediction)
        
        # Extract F1 scores
        return {
            'rouge1': scores['rouge1'].fmeasure,  # Unigram overlap
            'rouge2': scores['rouge2'].fmeasure,  # Bigram overlap
            'rougeL': scores['rougeL'].fmeasure,  # Longest common subsequence
        }
    
    def evaluate_multi_reference(
        self,
        prediction: str,
        references: List[str]
    ) -> Dict[str, float]:
        """Evaluate against multiple references (take max)."""
        all_scores = [self.evaluate (prediction, ref) for ref in references]
        
        # Take maximum score for each metric
        return {
            metric: max (scores[metric] for scores in all_scores)
            for metric in ['rouge1', 'rouge2', 'rougeL']
        }

# Usage
evaluator = ROUGEEvaluator()

reference = "The cat sat on the mat and looked out the window"
prediction = "A cat was sitting on a mat while looking through the window"

scores = evaluator.evaluate (prediction, reference)
print(f"ROUGE-1: {scores['rouge1']:.2%}")  # ~60% unigram overlap
print(f"ROUGE-2: {scores['rouge2']:.2%}")  # ~30% bigram overlap
print(f"ROUGE-L: {scores['rougeL']:.2%}")  # ~50% longest subsequence
\`\`\`

**Interpretation:**
- ROUGE-1: Measures unigram (single word) overlap
- ROUGE-2: Measures bigram (two-word phrase) overlap
- ROUGE-L: Longest common subsequence (captures sentence structure)

**Limitations:**
- Doesn't understand semantics ("big" vs "large" = no overlap)
- Rewards copying reference text (not always good)
- Doesn't check factuality

### 2. BLEU (Bilingual Evaluation Understudy)

Measures precision of n-grams. **Common for translation.**

\`\`\`python
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

class BLEUEvaluator:
    """Evaluate using BLEU metric."""
    
    def __init__(self):
        self.smoothing = SmoothingFunction().method1
    
    def evaluate(
        self,
        prediction: str,
        reference: str,
        max_n: int = 4
    ) -> float:
        """Calculate BLEU score."""
        
        # Tokenize
        reference_tokens = reference.lower().split()
        prediction_tokens = prediction.lower().split()
        
        # BLEU expects reference as list of token lists
        references = [reference_tokens]
        
        # Calculate BLEU score
        score = sentence_bleu(
            references,
            prediction_tokens,
            smoothing_function=self.smoothing
        )
        
        return score
    
    def evaluate_corpus(
        self,
        predictions: List[str],
        references: List[str]
    ) -> float:
        """Evaluate entire corpus."""
        scores = [
            self.evaluate (pred, ref)
            for pred, ref in zip (predictions, references)
        ]
        return sum (scores) / len (scores)

# Usage
evaluator = BLEUEvaluator()

reference = "The quick brown fox jumps over the lazy dog"
prediction = "A fast brown fox jumped over a lazy dog"

score = evaluator.evaluate (prediction, reference)
print(f"BLEU: {score:.2%}")  # ~40%
\`\`\`

**Interpretation:**
- Score 0-1 (higher is better)
- Considers n-gram precision (1-4 grams)
- Has brevity penalty (penalizes very short outputs)

**Limitations:**
- Precision-focused (doesn't check if all reference content covered)
- Poor for single sentence evaluation
- Doesn't understand meaning

## Semantic Similarity Metrics

### Embedding-Based Similarity

Compare semantic meaning using embeddings:

\`\`\`python
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
import numpy as np

class SemanticSimilarity:
    """Evaluate semantic similarity using embeddings."""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Popular models:
        - all-MiniLM-L6-v2: Fast, good quality
        - all-mpnet-base-v2: Higher quality, slower
        - paraphrase-multilingual: For non-English
        """
        self.model = SentenceTransformer (model_name)
    
    def cosine_similarity (self, text1: str, text2: str) -> float:
        """Cosine similarity between two texts."""
        embeddings = self.model.encode([text1, text2])
        
        # Cosine similarity
        similarity = 1 - cosine (embeddings[0], embeddings[1])
        
        return similarity
    
    def evaluate_batch(
        self,
        predictions: List[str],
        references: List[str]
    ) -> List[float]:
        """Batch evaluate for efficiency."""
        # Encode all at once (faster)
        pred_embeddings = self.model.encode (predictions)
        ref_embeddings = self.model.encode (references)
        
        # Compute similarities
        similarities = []
        for pred_emb, ref_emb in zip (pred_embeddings, ref_embeddings):
            sim = 1 - cosine (pred_emb, ref_emb)
            similarities.append (sim)
        
        return similarities
    
    def multi_reference_similarity(
        self,
        prediction: str,
        references: List[str]
    ) -> float:
        """Max similarity with any reference."""
        similarities = [
            self.cosine_similarity (prediction, ref)
            for ref in references
        ]
        return max (similarities)

# Usage
sem_sim = SemanticSimilarity()

ref = "The movie was excellent and entertaining"
pred = "The film was great and very enjoyable"  # Similar meaning, different words

score = sem_sim.cosine_similarity (pred, ref)
print(f"Semantic Similarity: {score:.2%}")  # ~85% (high despite different words)

# Compare with ROUGE (would be low due to different words)
rouge_eval = ROUGEEvaluator()
rouge_score = rouge_eval.evaluate (pred, ref)
print(f"ROUGE-1: {rouge_score['rouge1']:.2%}")  # ~20% (low - doesn't understand synonyms)
\`\`\`

**Advantages:**
- Understands semantic meaning
- Handles paraphrasing
- Language-agnostic (with multilingual models)

**Limitations:**
- Doesn't check factual accuracy
- Can be fooled by fluent but wrong text

### BERTScore

Contextual embeddings for token-level similarity:

\`\`\`python
from bert_score import score as bert_score

class BERTScoreEvaluator:
    """Evaluate using BERTScore."""
    
    def evaluate(
        self,
        predictions: List[str],
        references: List[str],
        lang: str = 'en'
    ) -> Dict[str, List[float]]:
        """
        BERTScore: Token-level contextual matching.
        Returns precision, recall, F1 for each example.
        """
        
        P, R, F1 = bert_score(
            predictions,
            references,
            lang=lang,
            verbose=False,
            model_type='microsoft/deberta-large-mnli'  # High quality
        )
        
        return {
            'precision': P.tolist(),
            'recall': R.tolist(),
            'f1': F1.tolist()
        }
    
    def evaluate_single (self, prediction: str, reference: str) -> float:
        """Single example evaluation."""
        result = self.evaluate([prediction], [reference])
        return result['f1'][0]

# Usage
bert_eval = BERTScoreEvaluator()

ref = "The patient was diagnosed with pneumonia"
pred = "The individual was found to have lung infection"  # Medical paraphrase

score = bert_eval.evaluate_single (pred, ref)
print(f"BERTScore: {score:.2%}")  # ~75% (understands medical synonyms)
\`\`\`

**Advantages:**
- Token-level matching with context
- Handles synonyms and paraphrasing better than ROUGE/BLEU
- Correlates better with human judgment

**Limitations:**
- Slower than n-gram metrics
- Requires GPU for efficiency
- Still doesn't verify facts

## Factuality and Hallucination Detection

### Fact Verification

Check if claims in output are supported by reference:

\`\`\`python
import spacy
from typing import List, Tuple

class FactualityChecker:
    """Verify factual consistency."""
    
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')
        
        # For production, use NLI model
        from transformers import pipeline
        self.nli = pipeline('text-classification', model='microsoft/deberta-large-mnli')
    
    def extract_claims (self, text: str) -> List[str]:
        """Extract factual claims from text."""
        doc = self.nlp (text)
        
        # Simple: extract sentences
        claims = [sent.text.strip() for sent in doc.sents]
        
        return claims
    
    def verify_claim (self, claim: str, reference: str) -> Dict[str, Any]:
        """Verify if claim is supported by reference using NLI."""
        
        # NLI: Natural Language Inference
        # premise: reference text
        # hypothesis: claim to verify
        
        result = self.nli (f"{reference} [SEP] {claim}")
        
        # Result: entailment, neutral, or contradiction
        label = result[0]['label']
        score = result[0]['score']
        
        is_supported = label == 'ENTAILMENT' and score > 0.7
        
        return {
            'claim': claim,
            'supported': is_supported,
            'label': label,
            'confidence': score
        }
    
    def evaluate_factuality(
        self,
        prediction: str,
        reference: str
    ) -> Dict[str, Any]:
        """Evaluate overall factuality."""
        
        # Extract claims from prediction
        claims = self.extract_claims (prediction)
        
        # Verify each claim
        verifications = [
            self.verify_claim (claim, reference)
            for claim in claims
        ]
        
        # Calculate metrics
        total_claims = len (claims)
        supported_claims = sum(1 for v in verifications if v['supported'])
        
        factuality_score = supported_claims / total_claims if total_claims > 0 else 0.0
        
        return {
            'factuality_score': factuality_score,
            'total_claims': total_claims,
            'supported_claims': supported_claims,
            'unsupported_claims': total_claims - supported_claims,
            'details': verifications
        }

# Usage
fact_checker = FactualityChecker()

reference = """
The Apollo 11 mission launched on July 16, 1969.
Neil Armstrong and Buzz Aldrin walked on the moon.
They returned to Earth on July 24, 1969.
"""

prediction = """
Apollo 11 launched in July 1969. Neil Armstrong was the first person on the moon.
The mission lasted about 8 days total. Armstrong planted a flag on Mars.
"""

result = fact_checker.evaluate_factuality (prediction, reference)
print(f"Factuality Score: {result['factuality_score']:.2%}")
print(f"Supported: {result['supported_claims']}/{result['total_claims']}")

for detail in result['details']:
    status = "✓" if detail['supported'] else "✗"
    print(f"{status} {detail['claim'][:60]}... ({detail['label']})")
# ✓ Apollo 11 launched in July 1969 (entailment)
# ✓ Neil Armstrong was the first person on the moon (entailment)
# ✗ Armstrong planted a flag on Mars (contradiction - it was the moon!)
\`\`\`

### Hallucination Detection

\`\`\`python
class HallucinationDetector:
    """Detect if model generated information not in input."""
    
    def __init__(self):
        self.nli = pipeline('zero-shot-classification')
    
    def detect_hallucination(
        self,
        input_context: str,
        model_output: str
    ) -> Dict[str, Any]:
        """
        Detect if output contains information not in input.
        """
        
        # Extract facts from output
        output_claims = self.extract_claims (model_output)
        
        hallucinations = []
        grounded_claims = []
        
        for claim in output_claims:
            # Check if claim can be inferred from input
            result = self.nli(
                claim,
                candidate_labels=['supported by context', 'not in context'],
                hypothesis_template='This claim is {}'
            )
            
            if result['labels'][0] == 'not in context' and result['scores'][0] > 0.7:
                hallucinations.append({
                    'claim': claim,
                    'confidence': result['scores'][0]
                })
            else:
                grounded_claims.append (claim)
        
        hallucination_rate = len (hallucinations) / len (output_claims) if output_claims else 0
        
        return {
            'hallucination_rate': hallucination_rate,
            'total_claims': len (output_claims),
            'hallucinated_claims': hallucinations,
            'grounded_claims': grounded_claims
        }

# Usage
detector = HallucinationDetector()

context = "Apple announced a new iPhone model with improved camera features."

output = """
Apple\'s new iPhone features an advanced camera system with 48MP sensors.
The device also includes a revolutionary holographic display.
It will be available in 12 new colors including rainbow gradient.
"""

result = detector.detect_hallucination (context, output)
print(f"Hallucination Rate: {result['hallucination_rate']:.2%}")
print("\\nHallucinated claims:")
for h in result['hallucinated_claims']:
    print(f"  - {h['claim']}")
# - The device also includes a revolutionary holographic display (not mentioned)
# - It will be available in 12 new colors (specific number not in context)
\`\`\`

## Task-Specific Evaluation

### Question Answering

\`\`\`python
class QAEvaluator:
    """Evaluate question answering outputs."""
    
    def exact_match (self, prediction: str, reference: str) -> float:
        """Exact string match (common for extractive QA)."""
        pred_normalized = prediction.strip().lower()
        ref_normalized = reference.strip().lower()
        return 1.0 if pred_normalized == ref_normalized else 0.0
    
    def f1_score (self, prediction: str, reference: str) -> float:
        """Token-level F1 score."""
        pred_tokens = set (prediction.lower().split())
        ref_tokens = set (reference.lower().split())
        
        if not pred_tokens or not ref_tokens:
            return 0.0
        
        common = pred_tokens & ref_tokens
        
        if not common:
            return 0.0
        
        precision = len (common) / len (pred_tokens)
        recall = len (common) / len (ref_tokens)
        
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1
    
    def evaluate (self, prediction: str, reference: str) -> Dict[str, float]:
        """Comprehensive QA evaluation."""
        return {
            'exact_match': self.exact_match (prediction, reference),
            'f1_score': self.f1_score (prediction, reference),
            'semantic_similarity': SemanticSimilarity().cosine_similarity (prediction, reference)
        }

# Usage
qa_eval = QAEvaluator()

question = "When was Python created?"
reference = "1991"
prediction = "Python was created in 1991"

scores = qa_eval.evaluate (prediction, reference)
print(f"Exact Match: {scores['exact_match']:.2%}")  # 0% (not exact)
print(f"F1 Score: {scores['f1_score']:.2%}")  # ~33% (contains "1991")
print(f"Semantic Sim: {scores['semantic_similarity']:.2%}")  # ~70% (same meaning)
\`\`\`

### Code Generation

\`\`\`python
class CodeEvaluator:
    """Evaluate generated code."""
    
    def __init__(self, test_cases: List[Dict]):
        self.test_cases = test_cases
    
    def execute_code (self, code: str, test_input: Any) -> Tuple[bool, Any, str]:
        """
        Safely execute code with test input.
        Returns (success, output, error_message)
        """
        import subprocess
        import json
        
        # Create temporary Python file
        with open('/tmp/test_code.py', 'w') as f:
            f.write (code)
            f.write (f"\\n\\nif __name__ == '__main__':\\n")
            f.write (f"    result = solution({json.dumps (test_input)})\\n")
            f.write (f"    print(json.dumps (result))\\n")
        
        try:
            # Execute with timeout
            result = subprocess.run(
                ['python', '/tmp/test_code.py'],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                output = json.loads (result.stdout.strip())
                return True, output, ""
            else:
                return False, None, result.stderr
                
        except subprocess.TimeoutExpired:
            return False, None, "Execution timeout"
        except Exception as e:
            return False, None, str (e)
    
    def evaluate (self, code: str) -> Dict[str, Any]:
        """Evaluate code on test cases."""
        
        passed = 0
        failed = 0
        errors = []
        
        for i, test in enumerate (self.test_cases):
            success, output, error = self.execute_code (code, test['input'])
            
            if success and output == test['expected_output']:
                passed += 1
            else:
                failed += 1
                errors.append({
                    'test_case': i,
                    'input': test['input'],
                    'expected': test['expected_output'],
                    'actual': output,
                    'error': error
                })
        
        pass_rate = passed / len (self.test_cases)
        
        return {
            'pass_rate': pass_rate,
            'passed': passed,
            'failed': failed,
            'total': len (self.test_cases),
            'errors': errors
        }

# Usage
test_cases = [
    {'input': [1, 2, 3], 'expected_output': 6},
    {'input': [0], 'expected_output': 0},
    {'input': [-1, 1], 'expected_output': 0},
]

code_eval = CodeEvaluator (test_cases)

generated_code = """
def solution (numbers):
    return sum (numbers)
"""

results = code_eval.evaluate (generated_code)
print(f"Pass Rate: {results['pass_rate']:.2%}")
print(f"Passed: {results['passed']}/{results['total']}")
\`\`\`

## LLM-as-Judge for Complex Evaluation

\`\`\`python
class LLMJudge:
    """Use GPT-4 to evaluate quality dimensions."""
    
    async def evaluate_dimensions(
        self,
        input: str,
        output: str,
        dimensions: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate output on multiple dimensions.
        
        dimensions: ['accuracy', 'helpfulness', 'clarity', 'safety', ...]
        """
        
        results = {}
        
        for dimension in dimensions:
            result = await self._evaluate_dimension (input, output, dimension)
            results[dimension] = result
        
        return results
    
    async def _evaluate_dimension(
        self,
        input: str,
        output: str,
        dimension: str
    ) -> Dict[str, Any]:
        """Evaluate single dimension."""
        
        prompts = {
            'accuracy': 'Is the information accurate and factually correct?',
            'helpfulness': 'Is this response helpful and relevant to the query?',
            'clarity': 'Is the response clear and easy to understand?',
            'safety': 'Is the content safe and appropriate?',
            'coherence': 'Is the response coherent and well-structured?',
            'completeness': 'Does the response fully address the query?'
        }
        
        prompt = f"""Evaluate this AI output on {dimension.upper()}.

Question: {prompts.get (dimension, f'Evaluate {dimension}')}

User Input:
{input}

AI Output:
{output}

Provide your evaluation in this JSON format:
{{
    "score": <number 1-10>,
    "reasoning": "<detailed explanation>",
    "issues": ["<specific issue 1>", "<issue 2>", ...],
    "strengths": ["<strength 1>", "<strength 2>", ...]
}}"""
        
        response = await openai.ChatCompletion.acreate(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert evaluator. Be critical but fair."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        
        result = json.loads (response.choices[0].message.content)
        result['normalized_score'] = result['score'] / 10.0
        
        return result

# Usage
judge = LLMJudge()

evaluation = await judge.evaluate_dimensions(
    input="Explain how vaccines work",
    output="Vaccines work by training your immune system...",
    dimensions=['accuracy', 'clarity', 'completeness']
)

for dimension, result in evaluation.items():
    print(f"\\n{dimension.upper()}: {result['score']}/10")
    print(f"  Reasoning: {result['reasoning']}")
    if result['issues']:
        print(f"  Issues: {', '.join (result['issues'])}")
\`\`\`

## Production Checklist

✅ **Metrics Selection**
- [ ] Multiple complementary metrics (n-gram + semantic + task-specific)
- [ ] Automatic metrics for speed
- [ ] LLM-as-judge for nuanced quality
- [ ] Human evaluation for critical cases

✅ **Factuality**
- [ ] Hallucination detection implemented
- [ ] Fact verification for critical claims
- [ ] Citation/source tracking

✅ **Efficiency**
- [ ] Batch processing for embeddings
- [ ] Caching of embeddings
- [ ] Parallel evaluation where possible

✅ **Analysis**
- [ ] Aggregate scores across test set
- [ ] Per-category breakdown
- [ ] Failure case analysis
- [ ] Correlation between metrics

## Next Steps

You now understand LLM output evaluation. Next, learn:
- A/B testing frameworks for comparing prompts/models
- Creating high-quality evaluation datasets
- Human evaluation workflows
- Continuous evaluation in production
`,
};
