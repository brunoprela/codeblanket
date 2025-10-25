export const llmEvaluationSafety = {
  title: 'LLM Evaluation & Safety',
  id: 'llm-evaluation-safety',
  content: `
# LLM Evaluation & Safety

## Evaluation Metrics

**Perplexity**: How surprised model is by test data (lower = better)
**BLEU/ROUGE**: N-gram overlap with reference (translation, summarization)
**Exact Match**: Percentage of exactly correct answers
**F1 Score**: Balance of precision and recall

## Benchmarks

**MMLU**: Multi-task understanding across 57 subjects
**HellaSwag**: Commonsense reasoning
**HumanEval**: Code generation (pass@k)
**TruthfulQA**: Factual accuracy

\`\`\`python
"""Evaluation example"""
from datasets import load_dataset

def evaluate_model (model, benchmark="mmlu"):
    dataset = load_dataset (benchmark)
    correct = 0
    total = 0
    
    for example in dataset['test']:
        prediction = model.generate (example['question'])
        if prediction.strip() == example['answer'].strip():
            correct += 1
        total += 1
    
    accuracy = correct / total
    return accuracy
\`\`\`

## Safety Measures

**Content Filtering**: Detect harmful content before/after generation
**PII Detection**: Remove personally identifiable information
**Bias Evaluation**: Test for demographic biases
**Hallucination Detection**: Verify factual claims

\`\`\`python
"""Safety layer"""
def safe_generation (prompt):
    # Pre-check
    if contains_harmful_content (prompt):
        return "I cannot respond to that request."
    
    # Generate
    response = model.generate (prompt)
    
    # Post-check
    if contains_pii (response):
        response = redact_pii (response)
    
    return response
\`\`\`

## Red Teaming

Adversarial testing to find model vulnerabilities:
- Jailbreak attempts
- Bias elicitation
- Harmful content generation
- Privacy violations

## Key Insights

- Multiple metrics needed (no single perfect metric)
- Human evaluation remains gold standard
- Safety is ongoing process, not one-time
- Balance capability with safety
`,
};
