/**
 * Prompt Optimization Techniques Section
 * Module 2: Prompt Engineering & Optimization
 */

export const promptoptimizationtechniquesSection = {
  id: 'prompt-optimization-techniques',
  title: 'Prompt Optimization Techniques',
  content: `# Prompt Optimization Techniques

Master systematic approaches to optimize prompts for better performance, lower costs, and higher reliability at scale.

## Overview: From Good to Great

Good prompts work once. **Optimized prompts** work reliably at scale, cost-effectively, and measurably better than alternatives.

### The Optimization Mindset

\`\`\`python
# Typical Journey
# Version 1: "Write code for X" → works 50% of the time
# Version 2: Add examples → works 70% of the time  
# Version 3: Add structure → works 85% of the time
# Version 4: Optimize based on failures → works 95% of the time
# Version 5: A/B test variants → find 98% solution

# Production requires systematic optimization, not guesswork
\`\`\`

## Iterative Prompt Refinement

### The Optimization Loop

\`\`\`python
from typing import List, Dict, Callable
from openai import OpenAI
import time

class PromptOptimizer:
    """
    Systematic prompt optimization framework.
    Used in production AI systems.
    """
    
    def __init__(self, model: str = "gpt-4"):
        self.client = OpenAI()
        self.model = model
        self.history = []
    
    def optimize(
        self,
        initial_prompt: str,
        test_cases: List[Dict],
        evaluation_func: Callable,
        max_iterations: int = 5,
        improvement_threshold: float = 0.05
    ) -> Dict:
        """
        Iteratively optimize prompt based on test performance.
        
        Args:
            initial_prompt: Starting prompt template
            test_cases: List of {'input': ..., 'expected': ...}
            evaluation_func: Function to score outputs (0-1)
            max_iterations: Maximum optimization iterations
            improvement_threshold: Minimum improvement to continue
            
        Returns:
            Optimized prompt and performance history
        """
        
        current_prompt = initial_prompt
        current_score = self._evaluate_prompt (current_prompt, test_cases, evaluation_func)
        
        self.history.append({
            'iteration': 0,
            'prompt': current_prompt,
            'score': current_score,
            'improvements': []
        })
        
        print(f"Initial score: {current_score:.3f}")
        
        for iteration in range(1, max_iterations + 1):
            print(f"\\nIteration {iteration}:")
            
            # Analyze failures
            failures = self._identify_failures (current_prompt, test_cases, evaluation_func)
            
            if not failures:
                print("No failures to learn from!")
                break
            
            # Generate improvements based on failures
            improvements = self._generate_improvements (current_prompt, failures)
            
            # Test each improvement
            best_variant = current_prompt
            best_score = current_score
            
            for improvement in improvements:
                variant = self._apply_improvement (current_prompt, improvement)
                score = self._evaluate_prompt (variant, test_cases, evaluation_func)
                
                print(f"  Variant '{improvement['name']}': {score:.3f}")
                
                if score > best_score:
                    best_variant = variant
                    best_score = score
            
            # Check for improvement
            improvement = best_score - current_score
            
            if improvement < improvement_threshold:
                print(f"\\nImprovement too small ({improvement:.3f}), stopping.")
                break
            
            print(f"  ✓ Improved by {improvement:.3f}")
            
            current_prompt = best_variant
            current_score = best_score
            
            self.history.append({
                'iteration': iteration,
                'prompt': current_prompt,
                'score': current_score,
                'improvement': improvement
            })
        
        return {
            'optimized_prompt': current_prompt,
            'final_score': current_score,
            'initial_score': self.history[0]['score'],
            'total_improvement': current_score - self.history[0]['score'],
            'history': self.history
        }
    
    def _evaluate_prompt(
        self,
        prompt_template: str,
        test_cases: List[Dict],
        evaluation_func: Callable
    ) -> float:
        """Evaluate prompt on test cases."""
        
        scores = []
        
        for test_case in test_cases:
            # Fill template
            prompt = prompt_template.format(**test_case['input'])
            
            # Get LLM response
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            
            output = response.choices[0].message.content
            
            # Evaluate
            score = evaluation_func (output, test_case.get('expected'))
            scores.append (score)
        
        return sum (scores) / len (scores) if scores else 0
    
    def _identify_failures(
        self,
        prompt_template: str,
        test_cases: List[Dict],
        evaluation_func: Callable,
        threshold: float = 0.7
    ) -> List[Dict]:
        """Identify test cases where prompt fails."""
        
        failures = []
        
        for test_case in test_cases:
            prompt = prompt_template.format(**test_case['input'])
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            
            output = response.choices[0].message.content
            score = evaluation_func (output, test_case.get('expected'))
            
            if score < threshold:
                failures.append({
                    'input': test_case['input'],
                    'expected': test_case.get('expected'),
                    'actual': output,
                    'score': score
                })
        
        return failures
    
    def _generate_improvements(
        self,
        current_prompt: str,
        failures: List[Dict]
    ) -> List[Dict]:
        """Generate potential improvements based on failures."""
        
        # Common improvement strategies
        improvements = [
            {
                'name': 'add_examples',
                'description': 'Add examples from failure cases',
                'modify': lambda p: p + "\\n\\nExamples:\\n" + "\\n".join([
                    f"Input: {f['input']}\\nExpected: {f['expected']}"
                    for f in failures[:2]
                ])
            },
            {
                'name': 'add_constraints',
                'description': 'Add explicit constraints',
                'modify': lambda p: p + "\\n\\nConstraints:\\n- Be specific\\n- Show your work\\n- Double-check your answer"
            },
            {
                'name': 'add_format',
                'description': 'Specify output format more clearly',
                'modify': lambda p: p + "\\n\\nFormat your response as:\\n[Your answer here]"
            },
            {
                'name': 'add_cot',
                'description': 'Add chain-of-thought',
                'modify': lambda p: p + "\\n\\nLet\'s solve this step by step:"
            }
        ]
        
        return improvements
    
    def _apply_improvement (self, prompt: str, improvement: Dict) -> str:
        """Apply an improvement to the prompt."""
        return improvement['modify'](prompt)

# Example usage
def email_validation_score (output: str, expected: str) -> float:
    """Score email validation accuracy."""
    output_clean = output.strip().lower()
    expected_clean = expected.strip().lower()
    
    if expected_clean in output_clean:
        return 1.0
    elif ('valid' in output_clean and 'valid' in expected_clean) or \
         ('invalid' in output_clean and 'invalid' in expected_clean):
        return 0.7
    else:
        return 0.0

# Test cases
test_cases = [
    {'input': {'email': 'user@example.com'}, 'expected': 'valid'},
    {'input': {'email': 'invalid.email'}, 'expected': 'invalid'},
    {'input': {'email': 'test@domain.co.uk'}, 'expected': 'valid'},
    {'input': {'email': '@nodomain.com'}, 'expected': 'invalid'},
]

# Initial prompt
initial_prompt = "Is this email valid? {email}\\n\\nAnswer:"

# Optimize
optimizer = PromptOptimizer()
result = optimizer.optimize(
    initial_prompt=initial_prompt,
    test_cases=test_cases,
    evaluation_func=email_validation_score,
    max_iterations=3
)

print(f"\\n{'='*60}")
print(f"OPTIMIZATION COMPLETE")
print(f"{'='*60}")
print(f"Initial score: {result['initial_score']:.3f}")
print(f"Final score: {result['final_score']:.3f}")
print(f"Improvement: {result['total_improvement']:.3f}")
print(f"\\nOptimized prompt:")
print(result['optimized_prompt'])
\`\`\`

## Measuring Prompt Quality

### Defining Success Metrics

\`\`\`python
from typing import Dict, List
import json

class PromptMetrics:
    """
    Comprehensive metrics for prompt performance.
    Track what matters in production.
    """
    
    @staticmethod
    def calculate_metrics(
        outputs: List[str],
        expected: List[str],
        costs: List[float],
        latencies: List[float]
    ) -> Dict:
        """
        Calculate comprehensive prompt performance metrics.
        """
        
        # Accuracy metrics
        exact_matches = sum(1 for o, e in zip (outputs, expected) if o.strip() == e.strip())
        accuracy = exact_matches / len (outputs)
        
        # Consistency (how similar are outputs for same input)
        consistency = PromptMetrics._calculate_consistency (outputs)
        
        # Cost metrics
        avg_cost = sum (costs) / len (costs)
        total_cost = sum (costs)
        
        # Latency metrics
        avg_latency = sum (latencies) / len (latencies)
        p95_latency = sorted (latencies)[int (len (latencies) * 0.95)]
        
        # Quality score (weighted combination)
        quality_score = (
            accuracy * 0.5 +
            consistency * 0.3 +
            min(1.0, 1.0 / (avg_cost + 0.001)) * 0.1 +
            min(1.0, 1.0 / (avg_latency + 0.1)) * 0.1
        )
        
        return {
            'accuracy': accuracy,
            'consistency': consistency,
            'avg_cost': avg_cost,
            'total_cost': total_cost,
            'avg_latency': avg_latency,
            'p95_latency': p95_latency,
            'quality_score': quality_score,
            'total_requests': len (outputs)
        }
    
    @staticmethod
    def _calculate_consistency (outputs: List[str]) -> float:
        """
        Calculate output consistency.
        Higher = more consistent formatting and style.
        """
        
        if len (outputs) < 2:
            return 1.0
        
        # Simple consistency: compare output lengths and formats
        lengths = [len (o) for o in outputs]
        avg_length = sum (lengths) / len (lengths)
        
        # Standard deviation of lengths (normalized)
        variance = sum((l - avg_length) ** 2 for l in lengths) / len (lengths)
        std_dev = variance **0.5
        
        # Consistency score: lower std dev = higher consistency
        consistency = max(0, 1 - (std_dev / (avg_length + 1)))
        
        return consistency

# Example
outputs = ["valid", "valid email", "yes, valid", "valid"]
expected = ["valid", "valid", "valid", "valid"]
costs = [0.001, 0.001, 0.0015, 0.001]
latencies = [0.5, 0.6, 0.7, 0.5]

metrics = PromptMetrics.calculate_metrics (outputs, expected, costs, latencies)

print("Prompt Performance Metrics:")
for key, value in metrics.items():
    if isinstance (value, float):
        print(f"  {key}: {value:.3f}")
    else:
        print(f"  {key}: {value}")
\`\`\`

## A/B Testing Prompts

### Statistical Comparison

\`\`\`python
from scipy import stats
import numpy as np

class PromptABTest:
    """
    A/B test prompt variants with statistical significance.
    Essential for production optimization.
    """
    
    def __init__(self, client=None):
        self.client = client or OpenAI()
    
    def run_ab_test(
        self,
        variant_a: str,
        variant_b: str,
        test_cases: List[Dict],
        evaluation_func: Callable,
        name_a: str = "Variant A",
        name_b: str = "Variant B"
    ) -> Dict:
        """
        Run A/B test comparing two prompt variants.
        """
        
        # Test both variants
        scores_a = self._test_variant (variant_a, test_cases, evaluation_func)
        scores_b = self._test_variant (variant_b, test_cases, evaluation_func)
        
        # Calculate statistics
        mean_a = np.mean (scores_a)
        mean_b = np.mean (scores_b)
        
        # Statistical significance (t-test)
        t_stat, p_value = stats.ttest_ind (scores_a, scores_b)
        
        # Effect size (Cohen\'s d)
        pooled_std = np.sqrt((np.var (scores_a) + np.var (scores_b)) / 2)
        cohens_d = (mean_b - mean_a) / pooled_std if pooled_std > 0 else 0
        
        # Determine winner
        if p_value < 0.05:  # Statistically significant
            if mean_b > mean_a:
                winner = name_b
                improvement = (mean_b - mean_a) / mean_a * 100
            else:
                winner = name_a
                improvement = (mean_a - mean_b) / mean_b * 100
            
            conclusion = f"{winner} wins with {improvement:.1f}% improvement (p={p_value:.4f})"
        else:
            winner = "No clear winner"
            conclusion = f"No statistically significant difference (p={p_value:.4f})"
        
        return {
            name_a: {
                'mean_score': mean_a,
                'std_dev': np.std (scores_a),
                'scores': scores_a
            },
            name_b: {
                'mean_score': mean_b,
                'std_dev': np.std (scores_b),
                'scores': scores_b
            },
            'statistics': {
                't_statistic': t_stat,
                'p_value': p_value,
                'cohens_d': cohens_d,
                'significant': p_value < 0.05
            },
            'winner': winner,
            'conclusion': conclusion
        }
    
    def _test_variant(
        self,
        prompt_template: str,
        test_cases: List[Dict],
        evaluation_func: Callable
    ) -> List[float]:
        """Test a prompt variant on all test cases."""
        
        scores = []
        
        for test_case in test_cases:
            prompt = prompt_template.format(**test_case['input'])
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            
            output = response.choices[0].message.content
            score = evaluation_func (output, test_case.get('expected'))
            scores.append (score)
        
        return scores
    
    def print_results (self, results: Dict):
        """Print formatted A/B test results."""
        
        print("\\n" + "="*60)
        print("A/B TEST RESULTS")
        print("="*60)
        
        for variant_name in [k for k in results.keys() if k not in ['statistics', 'winner', 'conclusion']]:
            variant = results[variant_name]
            print(f"\\n{variant_name}:")
            print(f"  Mean Score: {variant['mean_score']:.3f}")
            print(f"  Std Dev: {variant['std_dev']:.3f}")
        
        print(f"\\nStatistical Significance:")
        print(f"  p-value: {results['statistics']['p_value']:.4f}")
        print(f"  Significant: {results['statistics']['significant']}")
        print(f"  Effect Size (Cohen\'s d): {results['statistics']['cohens_d']:.3f}")
        
        print(f"\\n🏆 {results['conclusion']}")
        print("="*60 + "\\n")

# Example usage
def simple_eval (output: str, expected: str) -> float:
    return 1.0 if expected.lower() in output.lower() else 0.0

variant_a = "Classify this: {text}\\n\\nClassification:"
variant_b = "Classify the following text as positive or negative.\\n\\nText: {text}\\n\\nClassification:"

test_cases = [
    {'input': {'text': 'Great product!'}, 'expected': 'positive'},
    {'input': {'text': 'Terrible experience'}, 'expected': 'negative'},
    {'input': {'text': 'Best ever!'}, 'expected': 'positive'},
]

ab_test = PromptABTest()
results = ab_test.run_ab_test (variant_a, variant_b, test_cases, simple_eval)
ab_test.print_results (results)
\`\`\`

## DSPy: Automated Prompt Optimization

### Introduction to DSPy

\`\`\`python
"""
DSPy is a framework for automatic prompt optimization.
It treats prompts as learnable parameters.

Key concepts:
- Signatures: Define input/output behavior
- Modules: Composable prompt components
- Optimizers: Automatically improve prompts

Installation: pip install dspy-ai
"""

# Example DSPy usage (conceptual)
\`\`\`python
import dspy

# Configure LLM
lm = dspy.OpenAI(model="gpt-3.5-turbo")
dspy.settings.configure (lm=lm)

# Define signature (what the prompt should do)
class SentimentAnalysis (dspy.Signature):
    """Analyze the sentiment of text."""
    text = dspy.InputField()
    sentiment = dspy.OutputField (desc="positive, negative, or neutral")

# Create module
sentiment_module = dspy.Predict(SentimentAnalysis)

# Use it
result = sentiment_module (text="This product is amazing!")
print(result.sentiment)  # "positive"

# Optimize with examples
from dspy.teleprompt import BootstrapFewShot

# Training examples
train_examples = [
    dspy.Example (text="Great!", sentiment="positive").with_inputs('text'),
    dspy.Example (text="Awful", sentiment="negative").with_inputs('text'),
    dspy.Example (text="Okay", sentiment="neutral").with_inputs('text'),
]

# Optimizer automatically improves prompts
optimizer = BootstrapFewShot (metric=lambda example, pred: example.sentiment == pred.sentiment)
optimized_module = optimizer.compile (sentiment_module, trainset=train_examples)

# Optimized module uses better prompts under the hood
result = optimized_module (text="Not bad")
print(result.sentiment)
\`\`\`
\`\`\`

## Prompt Compression

### Reducing Token Count Without Losing Quality

\`\`\`python
class PromptCompressor:
    """
    Compress prompts to reduce token usage and costs.
    Critical for production optimization.
    """
    
    @staticmethod
    def remove_redundancy (prompt: str) -> str:
        """Remove redundant phrases and repetition."""
        
        # Remove common redundant phrases
        redundant_phrases = [
            "please ",
            "I would like you to ",
            "Could you ",
            "Can you ",
        ]
        
        compressed = prompt
        for phrase in redundant_phrases:
            compressed = compressed.replace (phrase, "")
        
        # Remove duplicate lines
        lines = compressed.split('\\n')
        unique_lines = []
        seen = set()
        
        for line in lines:
            if line.strip() and line.strip() not in seen:
                unique_lines.append (line)
                seen.add (line.strip())
            elif not line.strip():
                unique_lines.append (line)
        
        return '\\n'.join (unique_lines)
    
    @staticmethod
    def shorten_instructions (prompt: str) -> str:
        """Make instructions more concise."""
        
        replacements = {
            "You are a helpful assistant that": "",
            "In order to": "To",
            "Make sure to": "",
            "It is important that you": "",
            "You should": "",
            "very": "",
            "really": "",
            "actually": "",
        }
        
        compressed = prompt
        for old, new in replacements.items():
            compressed = compressed.replace (old, new)
        
        return compressed
    
    @staticmethod
    def compress_examples (examples: List[Dict], max_examples: int = 3) -> List[Dict]:
        """Reduce number of examples to most representative."""
        
        if len (examples) <= max_examples:
            return examples
        
        # Select diverse examples (simple selection by length)
        sorted_examples = sorted (examples, key=lambda x: len (x['input']))
        
        # Take shortest, longest, and middle
        indices = [0, len (sorted_examples) // 2, len (sorted_examples) - 1]
        return [sorted_examples[i] for i in indices[:max_examples]]
    
    @staticmethod
    def compress_full_prompt(
        prompt: str,
        preserve_examples: bool = True
    ) -> Dict[str, any]:
        """
        Comprehensive prompt compression.
        Returns compressed prompt and metrics.
        """
        
        import tiktoken
        
        # Count original tokens
        encoding = tiktoken.encoding_for_model("gpt-4")
        original_tokens = len (encoding.encode (prompt))
        
        # Apply compressions
        compressed = prompt
        compressed = PromptCompressor.remove_redundancy (compressed)
        compressed = PromptCompressor.shorten_instructions (compressed)
        
        # Count new tokens
        compressed_tokens = len (encoding.encode (compressed))
        
        # Calculate savings
        token_reduction = original_tokens - compressed_tokens
        cost_reduction_pct = (token_reduction / original_tokens) * 100 if original_tokens > 0 else 0
        
        return {
            'original_prompt': prompt,
            'compressed_prompt': compressed,
            'original_tokens': original_tokens,
            'compressed_tokens': compressed_tokens,
            'tokens_saved': token_reduction,
            'cost_reduction_pct': cost_reduction_pct
        }

# Example usage
verbose_prompt = """
You are a helpful assistant that helps users with code.
Please make sure to provide clear explanations.
Can you analyze this code and tell me if there are any issues?
It is important that you check for bugs very carefully.

Code:
def add (a, b):
    return a + b

Please provide your analysis.
"""

compressor = PromptCompressor()
result = compressor.compress_full_prompt (verbose_prompt)

print("Original Prompt:")
print(result['original_prompt'])
print(f"\\nOriginal Tokens: {result['original_tokens']}")

print("\\n" + "="*60 + "\\n")

print("Compressed Prompt:")
print(result['compressed_prompt'])
print(f"\\nCompressed Tokens: {result['compressed_tokens']}")

print(f"\\nTokens Saved: {result['tokens_saved']}")
print(f"Cost Reduction: {result['cost_reduction_pct']:.1f}%")
\`\`\`

## LLMLingua: Advanced Compression

### Using LLMLingua for Intelligent Compression

\`\`\`python
"""
LLMLingua uses a small LM to intelligently compress prompts
while preserving important information.

Installation: pip install llmlingua

Key features:
- Compresses up to 20x
- Preserves semantic meaning
- Minimal quality loss
- Huge cost savings
"""

# Example (conceptual - requires llmlingua package)
\`\`\`python
from llmlingua import PromptCompressor

# Initialize compressor
compressor = PromptCompressor()

# Long prompt
long_prompt = """
Context: Our company is building a new web application using React and TypeScript.
We need to implement user authentication with JWT tokens, store user data in PostgreSQL,
and provide RESTful APIs for the frontend to consume. The application should be scalable
and handle thousands of concurrent users. We also need to implement rate limiting and
proper error handling throughout the API layer.

Question: How should we structure the backend architecture?
"""

# Compress (target 50% reduction)
compressed = compressor.compress_prompt(
    long_prompt,
    instruction="",  # Optional instruction preservation
    question="",     # Optional question preservation
    rate=0.5         # Compression rate
)

print("Original length:", len (long_prompt))
print("Compressed length:", len (compressed['compressed_prompt']))
print("\\nCompressed:", compressed['compressed_prompt'])
# Result preserves key terms: React, TypeScript, JWT, PostgreSQL, RESTful, scalable, rate limiting
\`\`\`
\`\`\`

## Cost vs Quality Tradeoffs

### Finding the Optimal Balance

\`\`\`python
class CostQualityOptimizer:
    """
    Find optimal prompt balancing cost and quality.
    """
    
    @staticmethod
    def evaluate_tradeoff(
        variants: List[Dict],  # {'name': str, 'prompt': str, 'model': str}
        test_cases: List[Dict],
        evaluation_func: Callable,
        cost_per_1k_tokens: Dict[str, float]
    ) -> Dict:
        """
        Evaluate cost/quality tradeoff for prompt variants.
        """
        
        import tiktoken
        
        results = []
        
        for variant in variants:
            prompt = variant['prompt']
            model = variant['model']
            
            # Calculate cost
            encoding = tiktoken.encoding_for_model (model)
            tokens = len (encoding.encode (prompt))
            cost_per_request = (tokens / 1000) * cost_per_1k_tokens[model]
            
            # Calculate quality (would normally test with LLM)
            # For demo, assume quality scores
            quality = variant.get('quality', 0.8)
            
            # Calculate efficiency score
            efficiency = quality / (cost_per_request + 0.001)
            
            results.append({
                'name': variant['name'],
                'quality': quality,
                'cost': cost_per_request,
                'tokens': tokens,
                'efficiency': efficiency,
                'model': model
            })
        
        # Find best overall (highest efficiency)
        best = max (results, key=lambda x: x['efficiency'])
        
        return {
            'variants': results,
            'best_overall': best['name'],
            'recommendation': CostQualityOptimizer._make_recommendation (results)
        }
    
    @staticmethod
    def _make_recommendation (results: List[Dict]) -> str:
        """Make recommendation based on use case."""
        
        best_quality = max (results, key=lambda x: x['quality'])
        cheapest = min (results, key=lambda x: x['cost'])
        most_efficient = max (results, key=lambda x: x['efficiency'])
        
        return f"""
Recommendations:
- For highest quality: {best_quality['name']} (quality: {best_quality['quality']:.3f}, cost: \${best_quality['cost']:.4f})
- For lowest cost: { cheapest['name'] } (cost: \${ cheapest['cost']:.4f }, quality: {cheapest['quality']:.3f})
- Best balance: { most_efficient['name'] } (efficiency: {most_efficient['efficiency']:.1f})
"""

# Example
variants = [
    {
        'name': 'Detailed GPT-4',
        'prompt': 'Long detailed prompt...' * 50,
        'model': 'gpt-4',
        'quality': 0.95
    },
    {
        'name': 'Concise GPT-4',
        'prompt': 'Short prompt...' * 10,
        'model': 'gpt-4',
        'quality': 0.90
    },
    {
        'name': 'Detailed GPT-3.5',
        'prompt': 'Long detailed prompt...' * 50,
        'model': 'gpt-3.5-turbo',
        'quality': 0.85
    },
]

cost_per_1k = {
    'gpt-4': 0.03,
    'gpt-3.5-turbo': 0.002
}

optimizer = CostQualityOptimizer()
result = optimizer.evaluate_tradeoff (variants, [], None, cost_per_1k)

print("Cost/Quality Analysis:")
for v in result['variants']:
    print(f"\\n{v['name']}:")
print(f"  Quality: {v['quality']:.3f}")
print(f"  Cost: \\$\{v['cost']:.4f}")
print(f"  Efficiency: {v['efficiency']:.1f}")

print(f"\\n{result['recommendation']}")
\`\`\`

## Production Checklist

✅ **Optimization Process**
- Start with baseline measurements
- Identify failure patterns
- Generate improvements systematically
- A/B test variants
- Track metrics over time

✅ **Metrics to Track**
- Accuracy/success rate
- Consistency
- Cost per request
- Latency (avg and p95)
- Overall quality score

✅ **Testing Framework**
- Comprehensive test cases
- Statistical significance testing
- Cost vs quality analysis
- Automated regression testing
- Continuous monitoring

✅ **Compression Techniques**
- Remove redundancy
- Shorten instructions
- Reduce examples
- Use LLMLingua for advanced compression
- Balance compression with quality

✅ **Production Optimization**
- Version all prompts
- Track performance metrics
- Automate testing
- Continuous improvement loop
- Document what works

## Key Takeaways

1. **Optimize systematically, not randomly** - Follow structured process
2. **Measure everything** - Can't improve what you don't measure
3. **A/B test with statistical rigor** - Ensure improvements are real
4. **Compress intelligently** - Save tokens without losing quality
5. **Balance cost and quality** - Find optimal tradeoff for your use case
6. **Automate optimization** - DSPy and similar tools help scale
7. **Learn from failures** - Analyze what went wrong, fix it
8. **Version and track** - Know what works over time
9. **Test on real data** - Synthetic tests miss real-world edge cases
10. **Continuous improvement** - Optimization never stops in production

## Next Steps

Now that you understand prompt optimization techniques, you're ready to explore **Output Format Control** - learning how to enforce specific output structures and formats for reliable parsing and processing.`,
};
