/**
 * Prompt Evaluation & A/B Testing Section
 * Module 16: Evaluation, Data Operations & Fine-Tuning
 */

export const promptEvaluationABTesting = {
  id: 'prompt-evaluation-ab-testing',
  title: 'Prompt Evaluation & A/B Testing',
  content: `# Prompt Evaluation & A/B Testing

Master systematic prompt testing and A/B testing frameworks to optimize LLM performance in production.

## Overview: Why Systematic Prompt Testing Matters

Small prompt changes can have massive impact:

\`\`\`
Prompt A: "Summarize this article"
→ Average quality: 6.5/10, Cost: $0.02

Prompt B: "Summarize this article in 3 bullet points"
→ Average quality: 8.2/10, Cost: $0.01

Result: 26% better quality, 50% lower cost! 🎯
\`\`\`

**Without systematic testing, you're leaving performance on the table.**

### The Challenge

- **Hundreds of prompt variations possible**
- **Need objective comparison methodology**
- **Must balance quality vs cost vs latency**
- **Need statistical significance**
- **Want continuous optimization**

## Prompt Versioning

### Track Prompt Evolution

\`\`\`python
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Any
import hashlib

@dataclass
class PromptVersion:
    """Track prompt versions over time."""
    id: str
    template: str
    variables: Dict[str, str]
    created_at: datetime
    created_by: str
    description: str
    parent_version: Optional[str] = None
    metrics: Dict[str, float] = None
    status: str = "draft"  # draft, testing, active, archived
    
    def render(self, **kwargs) -> str:
        """Render prompt with variables."""
        return self.template.format(**kwargs)
    
    def get_hash(self) -> str:
        """Get unique hash for this version."""
        content = self.template + str(self.variables)
        return hashlib.md5(content.encode()).hexdigest()[:8]

class PromptRegistry:
    """Registry of all prompt versions."""
    
    def __init__(self):
        self.prompts: Dict[str, PromptVersion] = {}
        self.active_prompts: Dict[str, str] = {}  # task -> version_id
    
    def register(self, prompt: PromptVersion) -> str:
        """Register new prompt version."""
        prompt_id = f"{prompt.get_hash()}_{int(prompt.created_at.timestamp())}"
        prompt.id = prompt_id
        self.prompts[prompt_id] = prompt
        return prompt_id
    
    def set_active(self, task: str, version_id: str):
        """Set active version for a task."""
        if version_id not in self.prompts:
            raise ValueError(f"Version {version_id} not found")
        
        self.prompts[version_id].status = "active"
        
        # Archive previous active version
        if task in self.active_prompts:
            old_version = self.active_prompts[task]
            self.prompts[old_version].status = "archived"
        
        self.active_prompts[task] = version_id
    
    def get_active(self, task: str) -> PromptVersion:
        """Get currently active prompt for task."""
        version_id = self.active_prompts.get(task)
        if not version_id:
            raise ValueError(f"No active prompt for task {task}")
        return self.prompts[version_id]
    
    def get_history(self, task: str) -> List[PromptVersion]:
        """Get version history for a task."""
        # Find all versions for this task
        versions = [
            p for p in self.prompts.values()
            if task in p.description  # Simple matching; improve in production
        ]
        return sorted(versions, key=lambda p: p.created_at, reverse=True)

# Usage
registry = PromptRegistry()

# Register initial version
v1 = PromptVersion(
    id="",
    template="Summarize this article:\\n\\n{article}",
    variables={"article": "text"},
    created_at=datetime.now(),
    created_by="alice",
    description="summarization: Initial version"
)
v1_id = registry.register(v1)
registry.set_active("summarization", v1_id)

# Register improved version
v2 = PromptVersion(
    id="",
    template="Summarize this article in 3 bullet points:\\n\\n{article}",
    variables={"article": "text"},
    created_at=datetime.now(),
    created_by="bob",
    description="summarization: Added structure",
    parent_version=v1_id
)
v2_id = registry.register(v2)

# After testing shows v2 is better
registry.set_active("summarization", v2_id)

# Get current active prompt
active = registry.get_active("summarization")
prompt = active.render(article="...")
\`\`\`

## A/B Testing Framework

### Basic A/B Test

\`\`\`python
import random
from collections import defaultdict
from scipy import stats

class ABTest:
    """A/B test framework for prompts."""
    
    def __init__(
        self,
        variant_a: PromptVersion,
        variant_b: PromptVersion,
        traffic_split: float = 0.5
    ):
        """
        Args:
            variant_a: Control prompt
            variant_b: Treatment prompt
            traffic_split: % traffic to B (0.5 = 50/50 split)
        """
        self.variant_a = variant_a
        self.variant_b = variant_b
        self.traffic_split = traffic_split
        
        self.results_a = []
        self.results_b = []
        self.assignments = {}  # user_id -> variant
    
    def assign_variant(self, user_id: str = None) -> str:
        """Assign user to variant. Returns 'A' or 'B'."""
        
        if user_id and user_id in self.assignments:
            # Consistent assignment
            return self.assignments[user_id]
        
        # Random assignment
        variant = 'B' if random.random() < self.traffic_split else 'A'
        
        if user_id:
            self.assignments[user_id] = variant
        
        return variant
    
    def get_prompt(self, user_id: str = None) -> PromptVersion:
        """Get prompt based on A/B assignment."""
        variant = self.assign_variant(user_id)
        return self.variant_b if variant == 'B' else self.variant_a
    
    def record_result(
        self,
        variant: str,
        score: float,
        metadata: Dict[str, Any] = None
    ):
        """Record evaluation result."""
        result = {
            'score': score,
            'metadata': metadata or {},
            'timestamp': time.time()
        }
        
        if variant == 'A':
            self.results_a.append(result)
        else:
            self.results_b.append(result)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Calculate test statistics."""
        
        scores_a = [r['score'] for r in self.results_a]
        scores_b = [r['score'] for r in self.results_b]
        
        if not scores_a or not scores_b:
            return {'error': 'Insufficient data'}
        
        # Basic statistics
        mean_a = sum(scores_a) / len(scores_a)
        mean_b = sum(scores_b) / len(scores_b)
        
        # T-test for significance
        t_stat, p_value = stats.ttest_ind(scores_a, scores_b)
        
        # Effect size (Cohen's d)
        pooled_std = (
            (sum((x - mean_a)**2 for x in scores_a) + 
             sum((x - mean_b)**2 for x in scores_b)) /
            (len(scores_a) + len(scores_b) - 2)
        ) ** 0.5
        
        cohens_d = (mean_b - mean_a) / pooled_std if pooled_std > 0 else 0
        
        # Determine winner
        is_significant = p_value < 0.05
        winner = 'B' if mean_b > mean_a else 'A'
        
        return {
            'variant_a': {
                'mean_score': mean_a,
                'n_samples': len(scores_a),
                'std': pooled_std
            },
            'variant_b': {
                'mean_score': mean_b,
                'n_samples': len(scores_b),
                'std': pooled_std
            },
            'difference': mean_b - mean_a,
            'percent_improvement': ((mean_b - mean_a) / mean_a * 100) if mean_a > 0 else 0,
            'p_value': p_value,
            'is_significant': is_significant,
            'cohens_d': cohens_d,
            'winner': winner if is_significant else 'inconclusive',
            'confidence': 1 - p_value
        }
    
    def should_stop(self, min_samples: int = 100, confidence: float = 0.95) -> bool:
        """Check if we have enough data to make a decision."""
        
        if len(self.results_a) < min_samples or len(self.results_b) < min_samples:
            return False
        
        stats = self.get_statistics()
        
        if 'error' in stats:
            return False
        
        # Stop if we have high confidence
        return stats['is_significant'] and stats['confidence'] >= confidence

# Usage
ab_test = ABTest(
    variant_a=prompt_v1,
    variant_b=prompt_v2,
    traffic_split=0.5
)

# In production API
@app.post("/summarize")
async def summarize(request: SummarizeRequest):
    # Get prompt via A/B test
    prompt_version = ab_test.get_prompt(user_id=request.user_id)
    variant = ab_test.assign_variant(request.user_id)
    
    # Generate summary
    summary = await model.generate(prompt_version.render(article=request.article))
    
    # Evaluate quality (automatic or human feedback)
    quality_score = await evaluate_quality(summary)
    
    # Record result
    ab_test.record_result(variant, quality_score)
    
    # Check if test is conclusive
    if ab_test.should_stop():
        stats = ab_test.get_statistics()
        if stats['winner'] != 'inconclusive':
            print(f"✅ Winner: Variant {stats['winner']}")
            print(f"Improvement: {stats['percent_improvement']:.1f}%")
    
    return {"summary": summary}
\`\`\`

### Multi-Armed Bandit (Advanced)

Automatically allocate more traffic to better performing variants:

\`\`\`python
import numpy as np

class EpsilonGreedyBandit:
    """
    Multi-armed bandit with epsilon-greedy strategy.
    Balances exploration vs exploitation.
    """
    
    def __init__(
        self,
        variants: List[PromptVersion],
        epsilon: float = 0.1
    ):
        """
        Args:
            variants: List of prompt versions to test
            epsilon: Probability of exploration (0.1 = 10% random, 90% best)
        """
        self.variants = variants
        self.epsilon = epsilon
        
        # Track performance
        self.counts = [0] * len(variants)  # Times each variant used
        self.values = [0.0] * len(variants)  # Average reward for each
    
    def select_variant(self) -> int:
        """Select variant using epsilon-greedy strategy."""
        
        # Exploration: random variant
        if random.random() < self.epsilon:
            return random.randint(0, len(self.variants) - 1)
        
        # Exploitation: best performing variant
        # Handle cold start (no data yet)
        if all(c == 0 for c in self.counts):
            return random.randint(0, len(self.variants) - 1)
        
        # Select variant with highest average reward
        return int(np.argmax(self.values))
    
    def update(self, variant_idx: int, reward: float):
        """Update variant statistics with new reward."""
        self.counts[variant_idx] += 1
        n = self.counts[variant_idx]
        
        # Incremental average
        self.values[variant_idx] = (
            (self.values[variant_idx] * (n - 1) + reward) / n
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get current statistics."""
        total_plays = sum(self.counts)
        
        return {
            'variants': [
                {
                    'id': v.id,
                    'plays': self.counts[i],
                    'play_rate': self.counts[i] / total_plays if total_plays > 0 else 0,
                    'average_reward': self.values[i],
                    'description': v.description
                }
                for i, v in enumerate(self.variants)
            ],
            'total_plays': total_plays,
            'best_variant': int(np.argmax(self.values)) if any(c > 0 for c in self.counts) else None
        }

# Usage
bandit = EpsilonGreedyBandit(
    variants=[prompt_v1, prompt_v2, prompt_v3],
    epsilon=0.1  # 10% exploration
)

# In production
@app.post("/generate")
async def generate(request: GenerateRequest):
    # Select variant (more traffic to better performers)
    variant_idx = bandit.select_variant()
    prompt = bandit.variants[variant_idx]
    
    # Generate output
    output = await model.generate(prompt.render(**request.inputs))
    
    # Get reward (quality score)
    reward = await evaluate(output)
    
    # Update bandit
    bandit.update(variant_idx, reward)
    
    # Log statistics periodically
    if request.request_count % 100 == 0:
        stats = bandit.get_statistics()
        print(f"Best variant: {stats['best_variant']}")
        print(f"Performance: {stats['variants'][stats['best_variant']]['average_reward']:.2f}")
    
    return {"output": output}
\`\`\`

## Batch Prompt Evaluation

Evaluate prompts on test set before production:

\`\`\`python
class BatchPromptEvaluator:
    """Evaluate multiple prompts on test set."""
    
    def __init__(
        self,
        test_dataset: List[Dict],
        metrics: List[callable]
    ):
        self.test_dataset = test_dataset
        self.metrics = metrics
    
    async def evaluate_prompt(
        self,
        prompt_version: PromptVersion,
        model_fn: callable
    ) -> Dict[str, Any]:
        """Evaluate single prompt on full test set."""
        
        results = []
        
        for example in self.test_dataset:
            # Render prompt
            prompt = prompt_version.render(**example['inputs'])
            
            # Generate output
            start = time.time()
            output = await model_fn(prompt)
            latency = time.time() - start
            
            # Calculate metrics
            scores = {}
            for metric in self.metrics:
                score = metric(
                    output,
                    example.get('expected_output'),
                    example['inputs']
                )
                scores[metric.__name__] = score
            
            # Calculate cost
            cost = self._estimate_cost(prompt, output)
            
            results.append({
                'input': example['inputs'],
                'output': output,
                'scores': scores,
                'latency': latency,
                'cost': cost
            })
        
        # Aggregate
        aggregate_scores = {}
        for metric in self.metrics:
            metric_name = metric.__name__
            values = [r['scores'][metric_name] for r in results]
            aggregate_scores[metric_name] = {
                'mean': sum(values) / len(values),
                'min': min(values),
                'max': max(values),
                'std': np.std(values)
            }
        
        total_cost = sum(r['cost'] for r in results)
        avg_latency = sum(r['latency'] for r in results) / len(results)
        
        return {
            'prompt_id': prompt_version.id,
            'prompt_template': prompt_version.template,
            'aggregate_scores': aggregate_scores,
            'total_cost': total_cost,
            'avg_latency': avg_latency,
            'n_examples': len(results),
            'detailed_results': results
        }
    
    async def compare_prompts(
        self,
        prompts: List[PromptVersion],
        model_fn: callable
    ) -> pd.DataFrame:
        """Compare multiple prompts side-by-side."""
        
        all_results = []
        
        for prompt in prompts:
            print(f"Evaluating: {prompt.description}...")
            result = await self.evaluate_prompt(prompt, model_fn)
            all_results.append(result)
        
        # Create comparison dataframe
        comparison = []
        
        for result in all_results:
            row = {
                'prompt_id': result['prompt_id'],
                'description': result['prompt_template'][:50] + "...",
                'avg_latency': result['avg_latency'],
                'total_cost': result['total_cost']
            }
            
            # Add metric scores
            for metric_name, scores in result['aggregate_scores'].items():
                row[f'{metric_name}_mean'] = scores['mean']
            
            comparison.append(row)
        
        df = pd.DataFrame(comparison)
        
        # Rank by primary metric
        primary_metric = list(all_results[0]['aggregate_scores'].keys())[0]
        df = df.sort_values(f'{primary_metric}_mean', ascending=False)
        
        return df
    
    def _estimate_cost(self, prompt: str, output: str) -> float:
        """Estimate API cost."""
        input_tokens = len(prompt.split()) * 1.3
        output_tokens = len(output.split()) * 1.3
        total_tokens = input_tokens + output_tokens
        return (total_tokens / 1000) * 0.01  # $0.01 per 1K tokens

# Usage
evaluator = BatchPromptEvaluator(
    test_dataset=test_data,
    metrics=[accuracy_metric, coherence_metric]
)

prompts_to_test = [
    prompt_v1,  # "Summarize this article"
    prompt_v2,  # "Summarize this article in 3 bullet points"
    prompt_v3,  # "Provide a concise summary highlighting key points"
]

comparison = await evaluator.compare_prompts(prompts_to_test, model.generate)

print(comparison)
#   prompt_id  accuracy_mean  coherence_mean  avg_latency  total_cost
# 0 v2         0.85           0.92            1.2s         $0.50
# 1 v3         0.82           0.89            1.5s         $0.65
# 2 v1         0.78           0.85            1.1s         $0.48
\`\`\`

## Cost vs Quality Trade-offs

\`\`\`python
class CostQualityOptimizer:
    """Find optimal prompt considering both cost and quality."""
    
    def __init__(self, quality_weight: float = 0.7):
        """
        Args:
            quality_weight: How much to weight quality vs cost (0-1)
                           0.7 = 70% quality, 30% cost savings
        """
        self.quality_weight = quality_weight
        self.cost_weight = 1 - quality_weight
    
    def score_prompt(
        self,
        quality: float,
        cost: float,
        baseline_cost: float
    ) -> float:
        """
        Calculate composite score balancing quality and cost.
        
        Args:
            quality: Quality score (0-1)
            cost: Cost per request
            baseline_cost: Baseline cost to compare against
        """
        
        # Normalize cost (lower is better)
        cost_score = 1 - (cost / baseline_cost) if baseline_cost > 0 else 0
        cost_score = max(0, cost_score)  # Can't be negative
        
        # Weighted combination
        composite_score = (
            self.quality_weight * quality +
            self.cost_weight * cost_score
        )
        
        return composite_score
    
    def find_optimal(
        self,
        results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Find optimal prompt from evaluation results."""
        
        # Calculate baseline (first result or average)
        baseline_cost = results[0]['cost']
        
        # Score each result
        scored = []
        for result in results:
            composite = self.score_prompt(
                quality=result['quality'],
                cost=result['cost'],
                baseline_cost=baseline_cost
            )
            
            scored.append({
                **result,
                'composite_score': composite
            })
        
        # Sort by composite score
        scored.sort(key=lambda x: x['composite_score'], reverse=True)
        
        return scored[0]

# Usage
optimizer = CostQualityOptimizer(quality_weight=0.7)

results = [
    {'prompt_id': 'v1', 'quality': 0.85, 'cost': 0.02},
    {'prompt_id': 'v2', 'quality': 0.90, 'cost': 0.05},  # Better but expensive
    {'prompt_id': 'v3', 'quality': 0.88, 'cost': 0.015},  # Best value!
]

optimal = optimizer.find_optimal(results)
print(f"Optimal prompt: {optimal['prompt_id']}")
print(f"Quality: {optimal['quality']:.2%}, Cost: \${optimal['cost']}")
print(f"Composite Score: {optimal['composite_score']:.2f}")
\`\`\`

## Production A/B Testing Platform

\`\`\`python
class ProductionABPlatform:
    """Complete A/B testing platform for production."""
    
    def __init__(self):
        self.active_tests: Dict[str, ABTest] = {}
        self.registry = PromptRegistry()
        self.metrics_db = MetricsDatabase()
    
    def create_test(
        self,
        test_name: str,
        control: PromptVersion,
        treatment: PromptVersion,
        traffic_split: float = 0.5,
        target_sample_size: int = 1000
    ) -> str:
        """Create new A/B test."""
        
        test = ABTest(control, treatment, traffic_split)
        test_id = f"{test_name}_{int(time.time())}"
        
        self.active_tests[test_id] = {
            'test': test,
            'target_sample_size': target_sample_size,
            'started_at': time.time(),
            'status': 'running'
        }
        
        return test_id
    
    async def handle_request(
        self,
        test_id: str,
        user_id: str,
        inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle production request through A/B test."""
        
        if test_id not in self.active_tests:
            raise ValueError(f"Test {test_id} not found")
        
        test_data = self.active_tests[test_id]
        test = test_data['test']
        
        # Get prompt via A/B assignment
        prompt = test.get_prompt(user_id)
        variant = test.assign_variant(user_id)
        
        # Generate output
        output = await model.generate(prompt.render(**inputs))
        
        # Store for later evaluation
        await self.metrics_db.store_inference(
            test_id=test_id,
            user_id=user_id,
            variant=variant,
            inputs=inputs,
            output=output,
            prompt_id=prompt.id
        )
        
        return {
            'output': output,
            'variant': variant,
            'test_id': test_id
        }
    
    async def record_feedback(
        self,
        test_id: str,
        inference_id: str,
        score: float
    ):
        """Record user feedback/evaluation score."""
        
        # Get inference details
        inference = await self.metrics_db.get_inference(inference_id)
        
        # Record in A/B test
        test = self.active_tests[test_id]['test']
        test.record_result(inference['variant'], score)
        
        # Check if test should conclude
        if test.should_stop():
            await self.conclude_test(test_id)
    
    async def conclude_test(self, test_id: str):
        """Conclude A/B test and make decision."""
        
        test_data = self.active_tests[test_id]
        test = test_data['test']
        
        stats = test.get_statistics()
        
        # Log results
        logger.info(f"A/B Test {test_id} concluded:")
        logger.info(f"Winner: {stats['winner']}")
        logger.info(f"Improvement: {stats['percent_improvement']:.1f}%")
        logger.info(f"P-value: {stats['p_value']:.4f}")
        
        # Automatically promote winner if significant
        if stats['is_significant'] and stats['winner'] != 'inconclusive':
            winning_prompt = test.variant_b if stats['winner'] == 'B' else test.variant_a
            
            # Set as active
            task_name = test_id.split('_')[0]
            self.registry.set_active(task_name, winning_prompt.id)
            
            logger.info(f"✅ Promoted {winning_prompt.id} to production")
        
        # Update test status
        test_data['status'] = 'concluded'
        test_data['concluded_at'] = time.time()
        test_data['results'] = stats
    
    def get_test_status(self, test_id: str) -> Dict[str, Any]:
        """Get current test status."""
        test_data = self.active_tests[test_id]
        test = test_data['test']
        
        stats = test.get_statistics()
        
        progress = (
            (len(test.results_a) + len(test.results_b)) / 
            test_data['target_sample_size']
        )
        
        return {
            'test_id': test_id,
            'status': test_data['status'],
            'progress': min(progress, 1.0),
            'samples_collected': len(test.results_a) + len(test.results_b),
            'target_samples': test_data['target_sample_size'],
            'current_stats': stats if 'error' not in stats else None,
            'runtime_hours': (time.time() - test_data['started_at']) / 3600
        }

# Usage
platform = ProductionABPlatform()

# Create test
test_id = platform.create_test(
    test_name="summarization",
    control=prompt_v1,
    treatment=prompt_v2,
    traffic_split=0.5,
    target_sample_size=1000
)

# In production API
@app.post("/summarize")
async def summarize(request: SummarizeRequest):
    result = await platform.handle_request(
        test_id=test_id,
        user_id=request.user_id,
        inputs={'article': request.article}
    )
    return result

@app.post("/feedback")
async def feedback(request: FeedbackRequest):
    await platform.record_feedback(
        test_id=request.test_id,
        inference_id=request.inference_id,
        score=request.score
    )

# Check status
status = platform.get_test_status(test_id)
print(f"Progress: {status['progress']:.1%}")
print(f"Status: {status['status']}")
\`\`\`

## Common Pitfalls

### 1. Testing Too Early

**Problem:** Running A/B test before prompt is remotely good.

**Solution:** Do offline evaluation first, then A/B test top 2-3 candidates.

### 2. Insufficient Sample Size

**Problem:** Declaring winner with 10 samples per variant.

**Solution:** Aim for 100+ samples minimum. Use power analysis to determine needed N.

### 3. Ignoring Statistical Significance

**Problem:** "Variant B is winning 52% to 48%, let's ship it!"

**Solution:** Check p-value. Could just be random noise.

### 4. Testing Too Many Things

**Problem:** Testing 10 variants simultaneously, splitting traffic 10 ways.

**Solution:** Test 2-3 variants max. Use multi-armed bandits for more.

## Production Checklist

✅ **Prompt Management**
- [ ] Version control for all prompts
- [ ] Template system with variables
- [ ] Rollback capability
- [ ] Change tracking and audit log

✅ **Testing Infrastructure**
- [ ] A/B testing framework implemented
- [ ] Statistical significance checks
- [ ] Automatic winner selection
- [ ] Traffic allocation control

✅ **Metrics**
- [ ] Quality metrics defined
- [ ] Cost tracking integrated
- [ ] Latency monitoring
- [ ] User feedback collection

✅ **Automation**
- [ ] Automated prompt evaluation on test set
- [ ] CI/CD integration for prompt changes
- [ ] Alerts on performance degradation
- [ ] Auto-promotion of winning variants

## Next Steps

You now understand prompt evaluation and A/B testing. Next, learn:
- Building high-quality evaluation datasets
- Human evaluation workflows
- Data labeling and annotation
- Synthetic data generation
`,
};
