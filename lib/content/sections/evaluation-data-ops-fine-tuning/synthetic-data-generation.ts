/**
 * Synthetic Data Generation Section
 * Module 16: Evaluation, Data Operations & Fine-Tuning
 */

export const syntheticDataGeneration = {
  id: 'synthetic-data-generation',
  title: 'Synthetic Data Generation',
  content: `# Synthetic Data Generation

Master using LLMs to generate high-quality synthetic training and evaluation data.

## Overview: Why Synthetic Data?

Real labeled data is expensive and time-consuming. Synthetic data offers:

✅ **Scale**: Generate thousands of examples in hours
✅ **Cost**: 10-100x cheaper than human labeling
✅ **Control**: Target specific edge cases and scenarios
✅ **Privacy**: No real user data needed
✅ **Speed**: Rapid iteration and experimentation

⚠️ **Challenges**:
- Quality varies
- May not capture real distribution
- Potential for model biases
- Needs validation

## Synthetic Data Generation Strategies

### 1. LLM-Generated Examples

\`\`\`python
import openai
from typing import List, Dict, Any

class SyntheticDataGenerator:
    """Generate synthetic training data using LLMs."""
    
    def __init__(self, model: str = "gpt-4"):
        self.model = model
    
    async def generate_examples(
        self,
        task_description: str,
        num_examples: int = 100,
        difficulty_levels: List[str] = None
    ) -> List[Dict[str, Any]]:
        """Generate synthetic examples for a task."""
        
        if difficulty_levels is None:
            difficulty_levels = ["easy", "medium", "hard"]
        
        examples = []
        examples_per_level = num_examples // len (difficulty_levels)
        
        for difficulty in difficulty_levels:
            prompt = f"""Generate {examples_per_level} {difficulty} difficulty examples for this task:

Task: {task_description}

For each example, provide:
1. Input (the prompt/question)
2. Output (the expected response)
3. Category (what type of example)

Format as JSON array:
[
  {{"input": "...", "output": "...", "category": "..."}},
  ...
]

Make examples diverse and realistic."""

            response = await openai.ChatCompletion.acreate(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.8  # Higher for diversity
            )
            
            generated = json.loads (response.choices[0].message.content)
            
            # Add metadata
            for ex in generated:
                ex['difficulty'] = difficulty
                ex['source'] = 'synthetic'
                examples.append (ex)
        
        return examples

# Usage
generator = SyntheticDataGenerator()

examples = await generator.generate_examples(
    task_description="Summarize news articles",
    num_examples=300
)

print(f"Generated {len (examples)} synthetic examples")
# Can now use for training or evaluation!
\`\`\`

### 2. Few-Shot Bootstrapping

\`\`\`python
class FewShotBootstrapper:
    """Bootstrap from few real examples to many synthetic."""
    
    def __init__(self, seed_examples: List[Dict]):
        self.seed_examples = seed_examples
    
    async def bootstrap(
        self,
        target_count: int = 1000,
        model: str = "gpt-4"
    ) -> List[Dict]:
        """Generate many examples from few seeds."""
        
        synthetic = []
        
        while len (synthetic) < target_count:
            # Sample seed examples
            import random
            seeds = random.sample (self.seed_examples, min(3, len (self.seed_examples)))
            
            prompt = f"""Here are some examples of our task:

{self._format_seeds (seeds)}

Generate 10 NEW examples following the same pattern but with different content.
Make them diverse and realistic.

Format as JSON array."""

            response = await openai.ChatCompletion.acreate(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.9
            )
            
            new_examples = json.loads (response.choices[0].message.content)
            synthetic.extend (new_examples)
            
            print(f"Generated {len (synthetic)}/{target_count}...")
        
        return synthetic[:target_count]
    
    def _format_seeds (self, seeds: List[Dict]) -> str:
        """Format seed examples for prompt."""
        formatted = []
        for i, seed in enumerate (seeds):
            formatted.append (f"Example {i+1}:")
            formatted.append (f"Input: {seed['input']}")
            formatted.append (f"Output: {seed['output']}")
            formatted.append("")
        return "\\n".join (formatted)

# Usage
bootstrapper = FewShotBootstrapper (seed_examples=[
    {"input": "What is AI?", "output": "Artificial Intelligence is..."},
    {"input": "Explain ML", "output": "Machine Learning is..."},
    # Just 5-10 seed examples!
])

synthetic_data = await bootstrapper.bootstrap (target_count=1000)
print(f"Bootstrapped to {len (synthetic_data)} examples from {len (bootstrapper.seed_examples)} seeds!")
\`\`\`

### 3. Data Augmentation

\`\`\`python
class DataAugmentor:
    """Augment existing data with variations."""
    
    async def paraphrase (self, text: str) -> List[str]:
        """Generate paraphrases of text."""
        prompt = f"""Generate 5 different paraphrases of this text:

"{text}"

Requirements:
- Keep the same meaning
- Use different words and structure
- Vary complexity

Return as JSON array of strings."""

        response = await openai.ChatCompletion.acreate(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8
        )
        
        paraphrases = json.loads (response.choices[0].message.content)
        return paraphrases
    
    async def add_noise (self, example: Dict) -> List[Dict]:
        """Add realistic noise/errors to example."""
        variations = []
        
        # Typos
        variations.append({
            **example,
            'input': self._add_typos (example['input']),
            'augmentation': 'typos'
        })
        
        # Informal language
        informal = await self._make_informal (example['input'])
        variations.append({
            **example,
            'input': informal,
            'augmentation': 'informal'
        })
        
        return variations
    
    def _add_typos (self, text: str) -> str:
        """Add realistic typos."""
        import random
        words = text.split()
        if random.random() < 0.3:  # 30% chance
            idx = random.randint(0, len (words) - 1)
            words[idx] = self._typo (words[idx])
        return " ".join (words)
    
    def _typo (self, word: str) -> str:
        """Make typo in word."""
        if len (word) < 4:
            return word
        import random
        ops = ['swap', 'double', 'delete']
        op = random.choice (ops)
        
        if op == 'swap' and len (word) > 1:
            i = random.randint(0, len (word) - 2)
            word = word[:i] + word[i+1] + word[i] + word[i+2:]
        elif op == 'double':
            i = random.randint(0, len (word) - 1)
            word = word[:i] + word[i] + word[i:]
        elif op == 'delete' and len (word) > 3:
            i = random.randint(1, len (word) - 2)
            word = word[:i] + word[i+1:]
        
        return word

# Usage
augmentor = DataAugmentor()

# Paraphrase
original = "The weather is nice today"
paraphrases = await augmentor.paraphrase (original)
# ["Today\'s weather is pleasant", "It's a lovely day", ...]

# Add noise
noisy_variations = await augmentor.add_noise (example)
# Now have robust data with realistic user errors!
\`\`\`

## Quality Control for Synthetic Data

\`\`\`python
class SyntheticDataValidator:
    """Validate quality of synthetic data."""
    
    async def validate_batch(
        self,
        synthetic_data: List[Dict],
        validation_criteria: List[str]
    ) -> Dict[str, Any]:
        """Validate synthetic data quality."""
        
        issues = []
        
        for i, example in enumerate (synthetic_data):
            example_issues = await self._validate_example (example, validation_criteria)
            if example_issues:
                issues.append({
                    'example_index': i,
                    'issues': example_issues
                })
        
        quality_score = 1 - (len (issues) / len (synthetic_data))
        
        return {
            'total_examples': len (synthetic_data),
            'issues_found': len (issues),
            'quality_score': quality_score,
            'issues': issues
        }
    
    async def _validate_example(
        self,
        example: Dict,
        criteria: List[str]
    ) -> List[str]:
        """Validate single example."""
        issues = []
        
        # Check format
        if 'input' not in example or 'output' not in example:
            issues.append("missing_required_fields")
        
        # Check for duplicates (simple check)
        if example['input'] == example['output']:
            issues.append("input_equals_output")
        
        # Check length
        if len (example['input'].split()) < 3:
            issues.append("input_too_short")
        
        # Use LLM to check quality
        if "realistic" in criteria:
            is_realistic = await self._check_realistic (example)
            if not is_realistic:
                issues.append("unrealistic")
        
        return issues
    
    async def _check_realistic (self, example: Dict) -> bool:
        """Use LLM to check if example is realistic."""
        prompt = f"""Is this example realistic and natural?

Input: {example['input']}
Output: {example['output']}

Answer with just "yes" or "no"."""

        response = await openai.ChatCompletion.acreate(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        
        return "yes" in response.choices[0].message.content.lower()
    
    def filter_high_quality(
        self,
        synthetic_data: List[Dict],
        min_quality_score: float = 0.8
    ) -> List[Dict]:
        """Keep only high-quality examples."""
        # In practice, validate each and filter
        # Placeholder:
        return synthetic_data[:int (len (synthetic_data) * min_quality_score)]

# Usage
validator = SyntheticDataValidator()

validation = await validator.validate_batch(
    synthetic_data,
    validation_criteria=["realistic", "diverse", "correct"]
)

print(f"Quality Score: {validation['quality_score']:.2%}")
if validation['issues_found'] > 0:
    print(f"Found {validation['issues_found']} issues")

# Filter to high quality
high_quality = validator.filter_high_quality (synthetic_data, min_quality_score=0.9)
\`\`\`

## Hybrid Approach: Synthetic + Real

\`\`\`python
class HybridDatasetBuilder:
    """Combine synthetic and real data optimally."""
    
    def __init__(
        self,
        real_data: List[Dict],
        synthetic_generator: SyntheticDataGenerator
    ):
        self.real_data = real_data
        self.synthetic_generator = synthetic_generator
    
    async def build_hybrid_dataset(
        self,
        target_size: int = 10000,
        real_ratio: float = 0.2  # 20% real, 80% synthetic
    ) -> List[Dict]:
        """Build dataset combining real and synthetic."""
        
        n_real = int (target_size * real_ratio)
        n_synthetic = target_size - n_real
        
        # Sample real data
        import random
        sampled_real = random.sample (self.real_data, min (n_real, len (self.real_data)))
        
        # Generate synthetic to fill gap
        synthetic = await self.synthetic_generator.generate_examples(
            task_description=self._infer_task_description (sampled_real),
            num_examples=n_synthetic
        )
        
        # Combine
        hybrid = sampled_real + synthetic
        random.shuffle (hybrid)
        
        return hybrid
    
    def _infer_task_description (self, examples: List[Dict]) -> str:
        """Infer task from examples."""
        # Use LLM to infer what the task is
        sample = examples[0]
        return f"Generate examples similar to: Input: {sample['input']}, Output: {sample['output']}"

# Usage
hybrid_builder = HybridDatasetBuilder(
    real_data=real_labeled_data,
    synthetic_generator=generator
)

dataset = await hybrid_builder.build_hybrid_dataset(
    target_size=10000,
    real_ratio=0.1  # 10% real (1000), 90% synthetic (9000)
)

print(f"Built hybrid dataset: {len (dataset)} examples")
print("  Real: 10% (ground truth)")
print("  Synthetic: 90% (scale)")
\`\`\`

## Production Checklist

✅ **Generation Strategy**
- [ ] Appropriate method chosen (LLM, bootstrapping, augmentation)
- [ ] Seed examples representative
- [ ] Diversity ensured
- [ ] Edge cases included

✅ **Quality**
- [ ] Validation pipeline implemented
- [ ] Quality metrics tracked
- [ ] Human review of samples
- [ ] Filtering of low-quality examples

✅ **Balance**
- [ ] Optimal real-synthetic ratio determined
- [ ] Cost-quality trade-off analyzed
- [ ] Hybrid approach if beneficial

✅ **Validation**
- [ ] Test on real data to validate synthetic quality
- [ ] Monitor model performance with synthetic data
- [ ] Iterate and improve generation

## Next Steps

You now understand synthetic data generation. Next, learn:
- Fine-tuning fundamentals
- RAG evaluation
- Continuous monitoring
- Building evaluation platforms
`,
};
