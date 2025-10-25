/**
 * Model Fine-Tuning Fundamentals Section
 * Module 16: Evaluation, Data Operations & Fine-Tuning
 */

export const modelFineTuningFundamentals = {
  id: 'model-fine-tuning-fundamentals',
  title: 'Model Fine-Tuning Fundamentals',
  content: `# Model Fine-Tuning Fundamentals

Master when and how to fine-tune LLMs for your specific use case.

## Overview: Fine-Tuning vs Prompting

**When to prompt:** Quick, no data needed, flexible
**When to fine-tune:** Need consistent behavior, specific domain, high volume

\`\`\`python
def should_fine_tune (scenario: Dict) -> bool:
    """Decide if fine-tuning is worth it."""
    
    score = 0
    
    # Strong indicators FOR fine-tuning
    if scenario.get('consistent_output_format'):
        score += 2  # Fine-tuning excels at format consistency
    
    if scenario.get('domain_specific') and scenario.get('have_domain_data'):
        score += 2  # Learn domain knowledge
    
    if scenario.get('high_volume_requests'):
        score += 2  # Cost savings from smaller model
    
    if scenario.get('latency_critical'):
        score += 1  # Fine-tuned smaller model faster
    
    # Indicators AGAINST fine-tuning
    if scenario.get('task_changes_frequently'):
        score -= 2  # Prompting more flexible
    
    if scenario.get('limited_training_data') and scenario['training_examples'] < 100:
        score -= 2  # Need sufficient data
    
    if scenario.get('need_latest_knowledge'):
        score -= 1  # Prompting with RAG better
    
    # Decision threshold
    return score >= 3

# Examples
print(should_fine_tune({
    'consistent_output_format': True,
    'domain_specific': True,
    'have_domain_data': True,
    'training_examples': 5000,
    'high_volume_requests': True
}))  # True - perfect fine-tuning use case

print(should_fine_tune({
    'task_changes_frequently': True,
    'limited_training_data': True,
    'training_examples': 50,
    'need_latest_knowledge': True
}))  # False - stick with prompting
\`\`\`

## Fine-Tuning Methods

### 1. Full Fine-Tuning

Update all model weights.

**Pros:** Maximum adaptation
**Cons:** Expensive, needs lots of data, slow

**Use when:** Have large dataset (10K+ examples), significant distribution shift

### 2. LoRA (Low-Rank Adaptation)

Train small adapter layers, freeze base model.

**Pros:** 10-100x fewer parameters, fast, efficient
**Cons:** Slightly lower quality than full fine-tune

**Use when:** Limited compute, quick iteration needed

\`\`\`python
# LoRA conceptual example
class LoRALayer:
    """
    Instead of updating W (d x k), we add:
    ΔW = A @ B where A is (d x r), B is (r x k)
    r << d, k (e.g., r=8, d=4096, k=4096)
    
    Parameters: r (d+k) instead of d*k
    Example: 8(4096+4096) = 65K vs 16M = 250x reduction!
    """
    pass
\`\`\`

### 3. Prompt Tuning

Learn soft prompts (embedding vectors).

**Pros:** Minimal parameters (just prompt), very fast
**Cons:** Limited capability, works best for specific tasks

## Preparing Training Data

\`\`\`python
from typing import List, Dict, Any

class FineTuningDataPrep:
    """Prepare data for fine-tuning."""
    
    def format_for_openai(
        self,
        examples: List[Dict[str, str]]
    ) -> List[Dict]:
        """Format for OpenAI fine-tuning (JSONL)."""
        
        formatted = []
        
        for ex in examples:
            formatted.append({
                "messages": [
                    {"role": "system", "content": ex.get('system', 'You are a helpful assistant.')},
                    {"role": "user", "content": ex['input']},
                    {"role": "assistant", "content": ex['output']}
                ]
            })
        
        return formatted
    
    def validate_dataset(
        self,
        dataset: List[Dict]
    ) -> Dict[str, Any]:
        """Validate training dataset quality."""
        
        issues = []
        
        # Check size
        if len (dataset) < 50:
            issues.append("Dataset too small (<50 examples). Need 100+ for good results.")
        
        # Check for duplicates
        inputs = [ex['messages'][1]['content'] for ex in dataset]
        if len (inputs) != len (set (inputs)):
            issues.append("Duplicate inputs found")
        
        # Check output diversity
        outputs = [ex['messages'][2]['content'] for ex in dataset]
        if len (set (outputs)) < len (outputs) * 0.5:
            issues.append("Low output diversity - may overfit")
        
        # Check length distribution
        lengths = [len (ex['messages'][2]['content'].split()) for ex in dataset]
        avg_length = sum (lengths) / len (lengths)
        
        return {
            'num_examples': len (dataset),
            'avg_output_length': avg_length,
            'issues': issues,
            'quality_score': 1.0 if not issues else 0.5
        }

# Usage
data_prep = FineTuningDataPrep()

training_data = [
    {
        'system': 'You are a customer service bot',
        'input': 'How do I return an item?',
        'output': 'To return an item, please visit...'
    },
    # ... more examples
]

formatted = data_prep.format_for_openai (training_data)
validation = data_prep.validate_dataset (formatted)

if validation['issues']:
    print("⚠️  Data quality issues:")
    for issue in validation['issues']:
        print(f"  - {issue}")
else:
    print(f"✅ Dataset ready: {validation['num_examples']} examples")
\`\`\`

## Training Hyperparameters

\`\`\`python
class FineTuningConfig:
    """Fine-tuning hyperparameters."""
    
    def __init__(
        self,
        epochs: int = 3,
        learning_rate: float = 1e-5,
        batch_size: int = 8,
        warmup_steps: int = 100
    ):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
    
    @classmethod
    def get_recommended (cls, dataset_size: int) -> 'FineTuningConfig':
        """Get recommended config based on dataset size."""
        
        if dataset_size < 500:
            # Small dataset: more epochs, careful learning rate
            return cls(
                epochs=5,
                learning_rate=5e-6,
                batch_size=4,
                warmup_steps=50
            )
        elif dataset_size < 5000:
            # Medium dataset: balanced
            return cls(
                epochs=3,
                learning_rate=1e-5,
                batch_size=8,
                warmup_steps=100
            )
        else:
            # Large dataset: fewer epochs, larger batch
            return cls(
                epochs=2,
                learning_rate=2e-5,
                batch_size=16,
                warmup_steps=200
            )

# Usage
config = FineTuningConfig.get_recommended (dataset_size=2000)
print(f"Recommended: {config.epochs} epochs, LR={config.learning_rate}")
\`\`\`

## Evaluating Fine-Tuned Models

\`\`\`python
class FineTuneEvaluator:
    """Evaluate fine-tuned model vs base."""
    
    async def compare_models(
        self,
        base_model: callable,
        finetuned_model: callable,
        test_set: List[Dict]
    ) -> Dict[str, Any]:
        """Compare base vs fine-tuned performance."""
        
        base_scores = []
        ft_scores = []
        
        for example in test_set:
            # Base model
            base_output = await base_model (example['input'])
            base_score = self._score_output (base_output, example['expected'])
            base_scores.append (base_score)
            
            # Fine-tuned model
            ft_output = await finetuned_model (example['input'])
            ft_score = self._score_output (ft_output, example['expected'])
            ft_scores.append (ft_score)
        
        avg_base = sum (base_scores) / len (base_scores)
        avg_ft = sum (ft_scores) / len (ft_scores)
        
        improvement = ((avg_ft - avg_base) / avg_base) * 100
        
        return {
            'base_model_score': avg_base,
            'finetuned_model_score': avg_ft,
            'improvement_percent': improvement,
            'worth_it': improvement > 10  # >10% improvement threshold
        }
    
    def _score_output (self, output: str, expected: str) -> float:
        """Score output quality (0-1)."""
        # Implement your scoring logic
        return 0.85  # Placeholder

# Usage
evaluator = FineTuneEvaluator()

comparison = await evaluator.compare_models(
    base_model=gpt35_function,
    finetuned_model=my_finetuned_model,
    test_set=held_out_test_set
)

print(f"Base: {comparison['base_model_score']:.2%}")
print(f"Fine-tuned: {comparison['finetuned_model_score']:.2%}")
print(f"Improvement: {comparison['improvement_percent']:.1f}%")

if comparison['worth_it']:
    print("✅ Fine-tuning was worth it!")
else:
    print("⚠️  Marginal improvement - consider staying with base model + better prompts")
\`\`\`

## Common Pitfalls

### 1. Overfitting

**Problem:** Model memorizes training data, doesn't generalize.

**Solution:**
- Use validation set to monitor
- Early stopping
- More diverse training data
- Data augmentation

### 2. Catastrophic Forgetting

**Problem:** Model forgets general knowledge after fine-tuning.

**Solution:**
- Mix in general examples with domain-specific
- Use LoRA (preserves base model)
- Lower learning rate

### 3. Insufficient Data

**Problem:** <100 examples, model doesn't learn patterns.

**Solution:**
- Generate synthetic data
- Use data augmentation
- Try prompt engineering first
- Collect more data before fine-tuning

## Production Checklist

✅ **Decision**
- [ ] Validated fine-tuning is necessary
- [ ] Prompting + RAG explored first
- [ ] Clear success metrics defined

✅ **Data**
- [ ] 100+ high-quality examples
- [ ] Validation set held out
- [ ] Data diversity ensured
- [ ] Format validated

✅ **Training**
- [ ] Hyperparameters chosen appropriately
- [ ] Monitoring set up
- [ ] Checkpoints saved

✅ **Evaluation**
- [ ] Compared to base model
- [ ] Tested on held-out data
- [ ] Checked for overfitting
- [ ] Validated improvement worthwhile

## Next Steps

Now understand fine-tuning fundamentals. Next, learn:
- Fine-tuning OpenAI models
- Fine-tuning open-source models
- RAG evaluation
- Production deployment
`,
};
