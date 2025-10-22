/**
 * Meta-Prompting & Self-Improvement Section
 * Module 2: Prompt Engineering & Optimization
 */

export const metapromptingselfimprovementSection = {
    id: 'meta-prompting-self-improvement',
    title: 'Meta-Prompting & Self-Improvement',
    content: `# Meta-Prompting & Self-Improvement

Master using LLMs to write, improve, and optimize their own prompts for continuous performance improvement.

## Overview: LLMs Improving Themselves

Meta-prompting is using an LLM to generate or improve prompts for itself or other LLMs. This creates self-improving AI systems.

\`\`\`python
# Traditional: Human writes prompt
human_prompt = "Classify sentiment"

# Meta-prompting: LLM writes its own prompt
meta_prompt = "Generate an optimal prompt for sentiment classification that includes examples and format specification"

# Result: LLM generates better prompt than human might write!
\`\`\`

## LLMs Writing Prompts

\`\`\`python
from openai import OpenAI

client = OpenAI()

def generate_prompt_for_task(
    task_description: str,
    requirements: list = None
) -> str:
    """
    Use LLM to generate an optimal prompt for a task.
    """
    
    requirements_text = ""
    if requirements:
        requirements_text = "\\n".join([f"- {req}" for req in requirements])
    
    meta_prompt = f"""You are an expert prompt engineer. Generate an optimal prompt for the following task:

Task: {task_description}

Requirements:
{requirements_text}

The generated prompt should:
1. Be clear and specific
2. Include relevant examples
3. Specify output format
4. Include necessary constraints
5. Be optimized for reliability

Generate the prompt:"""
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": meta_prompt}],
        temperature=0.7
    )
    
    return response.choices[0].message.content

# Example usage
task = "Extract person names and ages from text"
requirements = [
    "Output as JSON",
    "Handle missing ages gracefully",
    "Work for multiple people",
    "Include examples"
]

generated_prompt = generate_prompt_for_task(task, requirements)
print("Generated Prompt:")
print(generated_prompt)

# Now use this generated prompt for actual task!
\`\`\`

## Prompt Improvement Loops

\`\`\`python
from typing import List, Dict

class PromptSelfImprover:
    """
    Automatically improve prompts using LLM feedback.
    Self-improving prompt system.
    """
    
    def __init__(self, model: str = "gpt-4"):
        self.client = OpenAI()
        self.model = model
    
    def improve_prompt(
        self,
        current_prompt: str,
        failure_examples: List[Dict] = None,
        success_criteria: str = None
    ) -> Dict:
        """
        Use LLM to suggest improvements to a prompt.
        """
        
        improvement_request = f"""Analyze this prompt and suggest improvements:

CURRENT PROMPT:
{current_prompt}

{self._format_failures(failure_examples)}

SUCCESS CRITERIA:
{success_criteria or 'Maximize accuracy and reliability'}

Provide:
1. Issues with current prompt
2. Specific improvements
3. Improved version of the prompt

Analysis:"""
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": improvement_request}],
            temperature=0.7
        )
        
        analysis = response.choices[0].message.content
        
        # Extract improved prompt (simple parsing)
        improved_prompt = self._extract_improved_prompt(analysis)
        
        return {
            'original_prompt': current_prompt,
            'analysis': analysis,
            'improved_prompt': improved_prompt
        }
    
    def _format_failures(self, failures: List[Dict] = None) -> str:
        """Format failure examples for context."""
        if not failures:
            return ""
        
        failures_text = "\\n\\n".join([
            f"Input: {f['input']}\\nExpected: {f['expected']}\\nActual: {f['actual']}"
            for f in failures
        ])
        
        return f"""FAILURE EXAMPLES (these didn't work well):
{failures_text}"""
    
    def _extract_improved_prompt(self, analysis: str) -> str:
        """Extract improved prompt from analysis."""
        # Look for sections like "IMPROVED PROMPT:" or similar
        lines = analysis.split('\\n')
        
        in_prompt_section = False
        prompt_lines = []
        
        for line in lines:
            if 'improved' in line.lower() and 'prompt' in line.lower():
                in_prompt_section = True
                continue
            
            if in_prompt_section:
                if line.strip() and not line.startswith('#'):
                    prompt_lines.append(line)
        
        return '\\n'.join(prompt_lines) if prompt_lines else analysis

# Example usage
improver = PromptSelfImprover()

current = "Classify the sentiment: {text}"

failures = [
    {
        'input': 'It was okay I guess',
        'expected': 'neutral',
        'actual': 'positive'
    },
    {
        'input': 'Not bad',
        'expected': 'positive',
        'actual': 'neutral'
    }
]

result = improver.improve_prompt(
    current_prompt=current,
    failure_examples=failures,
    success_criteria="95%+ accuracy on nuanced sentiment"
)

print("Analysis:")
print(result['analysis'])
print("\\nImproved Prompt:")
print(result['improved_prompt'])
\`\`\`

## Self-Critique and Refinement

\`\`\`python
class SelfCriticSystem:
    """
    System where LLM critiques and improves its own outputs.
    """
    
    def __init__(self, client=None):
        self.client = client or OpenAI()
    
    def generate_and_critique(
        self,
        task: str,
        iterations: int = 3
    ) -> Dict:
        """
        Generate response, critique it, refine iteratively.
        """
        
        current_output = None
        history = []
        
        for i in range(iterations):
            print(f"\\nIteration {i+1}:")
            
            # Generate or refine
            if current_output is None:
                # First generation
                prompt = task
            else:
                # Refinement based on critique
                prompt = f"""Previous attempt:
{current_output}

Critique: {critique}

Improve the response based on the critique:"""
            
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )
            
            current_output = response.choices[0].message.content
            print(f"Output: {current_output[:100]}...")
            
            # Self-critique
            critique_prompt = f"""Critique this response to the task:

Task: {task}

Response: {current_output}

Provide constructive critique focusing on:
1. Accuracy
2. Completeness
3. Clarity
4. Any errors or omissions

Critique:"""
            
            critique_response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": critique_prompt}],
                temperature=0.3
            )
            
            critique = critique_response.choices[0].message.content
            print(f"Critique: {critique[:100]}...")
            
            history.append({
                'iteration': i + 1,
                'output': current_output,
                'critique': critique
            })
            
            # Check if critique is positive
            if 'excellent' in critique.lower() or 'no issues' in critique.lower():
                print("Self-critique satisfied!")
                break
        
        return {
            'final_output': current_output,
            'iterations': len(history),
            'history': history
        }

# Example
critic = SelfCriticSystem()

result = critic.generate_and_critique(
    task="Explain quantum computing in simple terms for a 10-year-old",
    iterations=2
)

print("\\n" + "="*60)
print("FINAL OUTPUT:")
print(result['final_output'])
\`\`\`

## Automated Prompt Generation

\`\`\`python
class AutomaticPromptGenerator:
    """
    Automatically generate prompts from specifications.
    """
    
    def __init__(self, client=None):
        self.client = client or OpenAI()
    
    def generate_from_examples(
        self,
        examples: List[Dict],
        task_description: str = None
    ) -> str:
        """
        Generate prompt from input/output examples.
        LLM infers the pattern.
        """
        
        examples_text = "\\n\\n".join([
            f"Input: {ex['input']}\\nOutput: {ex['output']}"
            for ex in examples
        ])
        
        meta_prompt = f"""Given these input/output examples, generate a clear and comprehensive prompt that would produce these outputs:

{examples_text}

{f"Task description: {task_description}" if task_description else ""}

Generate a prompt that:
1. Clearly describes the transformation
2. Specifies the output format
3. Includes relevant constraints
4. Would work for similar inputs

Generated prompt:"""
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": meta_prompt}]
        )
        
        return response.choices[0].message.content
    
    def generate_from_description(
        self,
        task: str,
        domain: str,
        constraints: List[str] = None
    ) -> str:
        """
        Generate prompt from high-level description.
        """
        
        constraints_text = ""
        if constraints:
            constraints_text = "\\nConstraints:\\n" + "\\n".join([f"- {c}" for c in constraints])
        
        meta_prompt = f"""Generate an optimal prompt for this task:

Domain: {domain}
Task: {task}
{constraints_text}

The prompt should be production-ready, including:
- Clear instructions
- Output format specification
- Relevant examples
- Edge case handling
- Any necessary constraints

Generated prompt:"""
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": meta_prompt}]
        )
        
        return response.choices[0].message.content

# Example 1: Generate from examples
generator = AutomaticPromptGenerator()

examples = [
    {'input': 'user@example.com', 'output': 'valid'},
    {'input': 'invalid.email', 'output': 'invalid'},
    {'input': 'test@domain.co.uk', 'output': 'valid'}
]

prompt_from_examples = generator.generate_from_examples(
    examples=examples,
    task_description="Email validation"
)

print("Prompt from examples:")
print(prompt_from_examples)

# Example 2: Generate from description
prompt_from_desc = generator.generate_from_description(
    task="Summarize technical documentation",
    domain="Software engineering",
    constraints=[
        "Maximum 3 bullet points",
        "Focus on key features only",
        "Use technical terminology"
    ]
)

print("\\nPrompt from description:")
print(prompt_from_desc)
\`\`\`

## Prompt Evolution Systems

\`\`\`python
from typing import List, Callable
import random

class PromptEvolver:
    """
    Evolve prompts using evolutionary algorithms.
    Generate variations, test, keep best.
    """
    
    def __init__(self, client=None):
        self.client = client or OpenAI()
    
    def evolve_prompt(
        self,
        initial_prompt: str,
        test_cases: List[Dict],
        evaluation_func: Callable,
        generations: int = 5,
        population_size: int = 4
    ) -> Dict:
        """
        Evolve prompt over multiple generations.
        """
        
        population = [initial_prompt]
        best_prompt = initial_prompt
        best_score = self._evaluate(initial_prompt, test_cases, evaluation_func)
        
        history = []
        
        for gen in range(generations):
            print(f"\\nGeneration {gen + 1}:")
            
            # Generate variations
            variations = []
            for prompt in population[:2]:  # Mutate best 2
                for _ in range(population_size // 2):
                    variation = self._mutate_prompt(prompt)
                    variations.append(variation)
            
            # Evaluate all
            population_with_scores = []
            for prompt in variations:
                score = self._evaluate(prompt, test_cases, evaluation_func)
                population_with_scores.append((prompt, score))
                print(f"  Variant score: {score:.3f}")
            
            # Keep best
            population_with_scores.sort(key=lambda x: x[1], reverse=True)
            population = [p for p, _ in population_with_scores[:population_size]]
            
            # Track best
            gen_best = population_with_scores[0]
            if gen_best[1] > best_score:
                best_prompt = gen_best[0]
                best_score = gen_best[1]
                print(f"  New best score: {best_score:.3f}")
            
            history.append({
                'generation': gen + 1,
                'best_score': gen_best[1],
                'population_size': len(population)
            })
        
        return {
            'best_prompt': best_prompt,
            'best_score': best_score,
            'initial_score': history[0]['best_score'] if history else best_score,
            'improvement': best_score - (history[0]['best_score'] if history else best_score),
            'history': history
        }
    
    def _mutate_prompt(self, prompt: str) -> str:
        """Generate variation of prompt."""
        
        mutation_prompt = f"""Create a variation of this prompt that might perform better:

Original prompt:
{prompt}

Generate a variation that:
- Keeps the core task the same
- Might be more clear, specific, or effective
- Uses different phrasing or structure
- Could include better examples

Variation:"""
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": mutation_prompt}],
            temperature=0.8  # Higher temperature for diversity
        )
        
        return response.choices[0].message.content
    
    def _evaluate(
        self,
        prompt: str,
        test_cases: List[Dict],
        evaluation_func: Callable
    ) -> float:
        """Evaluate prompt on test cases."""
        
        scores = []
        
        for test_case in test_cases[:3]:  # Limit for speed
            try:
                # Execute prompt
                filled_prompt = prompt.format(**test_case['input'])
                
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": filled_prompt}],
                    temperature=0.3
                )
                
                output = response.choices[0].message.content
                score = evaluation_func(output, test_case.get('expected'))
                scores.append(score)
            except:
                scores.append(0.0)
        
        return sum(scores) / len(scores) if scores else 0.0

# Example usage (conceptual - would need real execution)
def simple_eval(output: str, expected: str) -> float:
    return 1.0 if expected.lower() in output.lower() else 0.0

evolver = PromptEvolver()

# This would actually run evolution
# result = evolver.evolve_prompt(
#     initial_prompt="Classify: {text}",
#     test_cases=[
#         {'input': {'text': 'Great!'}, 'expected': 'positive'},
#         {'input': {'text': 'Bad'}, 'expected': 'negative'},
#     ],
#     evaluation_func=simple_eval,
#     generations=3
# )
\`\`\`

## Continuous Improvement Systems

\`\`\`python
class ContinuousImprovement:
    """
    Production system for continuous prompt improvement.
    """
    
    def __init__(self, client=None):
        self.client = client or OpenAI()
        self.performance_history = []
    
    def monitor_and_improve(
        self,
        prompt_id: str,
        current_prompt: str,
        recent_failures: List[Dict],
        success_rate: float
    ) -> Dict:
        """
        Monitor performance and suggest improvements when needed.
        """
        
        # Check if improvement needed
        if success_rate < 0.90 or len(recent_failures) > 5:
            print(f"Performance below threshold: {success_rate:.1%}")
            print("Generating improved prompt...")
            
            # Analyze failures
            improvement = self._analyze_and_improve(
                current_prompt,
                recent_failures
            )
            
            return {
                'needs_improvement': True,
                'current_performance': success_rate,
                'improved_prompt': improvement['improved_prompt'],
                'reasoning': improvement['reasoning'],
                'recommendation': 'Test and deploy improved prompt'
            }
        
        return {
            'needs_improvement': False,
            'current_performance': success_rate,
            'recommendation': 'Current prompt performing well'
        }
    
    def _analyze_and_improve(
        self,
        prompt: str,
        failures: List[Dict]
    ) -> Dict:
        """Analyze failures and generate improved prompt."""
        
        failures_text = "\\n\\n".join([
            f"Input: {f['input']}\\nExpected: {f['expected']}\\nActual: {f['actual']}"
            for f in failures[:10]  # Limit to recent 10
        ])
        
        analysis_prompt = f"""Analyze why this prompt is failing and suggest improvements:

CURRENT PROMPT:
{prompt}

RECENT FAILURES:
{failures_text}

Provide:
1. Root cause analysis
2. Specific improvements to address failures
3. Improved prompt

Analysis:"""
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": analysis_prompt}]
        )
        
        analysis = response.choices[0].message.content
        
        return {
            'reasoning': analysis,
            'improved_prompt': analysis  # Would extract from analysis
        }

# Example usage in production monitoring
improver = ContinuousImprovement()

# Simulate production monitoring
result = improver.monitor_and_improve(
    prompt_id="sentiment_v1",
    current_prompt="Classify sentiment: {text}",
    recent_failures=[
        {'input': 'not bad', 'expected': 'positive', 'actual': 'neutral'},
        {'input': 'could be better', 'expected': 'negative', 'actual': 'neutral'}
    ],
    success_rate=0.85
)

if result['needs_improvement']:
    print("\\n⚠️ Prompt needs improvement")
    print(f"Current performance: {result['current_performance']:.1%}")
    print(f"Recommendation: {result['recommendation']}")
\`\`\`

## Production Checklist

✅ **Meta-Prompting Setup**
- Automate prompt generation
- Generate from examples
- Generate from descriptions
- Version generated prompts
- Test before deployment

✅ **Self-Improvement Loop**
- Monitor performance metrics
- Collect failure examples
- Trigger improvement automatically
- Test improved versions
- A/B test before full deployment

✅ **Continuous Monitoring**
- Track success rates
- Collect edge cases
- Monitor cost per request
- Track latency
- Alert on degradation

✅ **Evolution System**
- Generate variations
- Test systematically
- Keep best performers
- Archive all versions
- Document improvements

✅ **Production Safeguards**
- Human review of generated prompts
- A/B test improvements
- Gradual rollout
- Rollback capability
- Performance validation

## Key Takeaways

1. **LLMs can write better prompts** - Than humans sometimes
2. **Meta-prompting enables automation** - Generate prompts at scale
3. **Self-critique improves outputs** - Iterative refinement works
4. **Continuous improvement is possible** - Systems that get better over time
5. **Evolution finds optimal prompts** - Genetic algorithm approach
6. **Monitor and adapt** - Production systems need ongoing optimization
7. **Generate from examples** - LLM infers pattern from input/output pairs
8. **Automate improvement cycles** - Don't rely on manual prompt engineering
9. **Test before deploying** - Generated prompts need validation
10. **Human oversight still essential** - But automation scales better

## Next Steps

Congratulations! You've completed the Prompt Engineering & Optimization module. You now have a comprehensive understanding of:

- Prompt engineering fundamentals
- System prompts and role assignment
- Few-shot learning with examples
- Chain-of-thought prompting
- Prompt optimization techniques
- Output format control
- Context management
- Negative prompting and constraints
- Prompt injection security
- Meta-prompting and self-improvement

Next module: **File Processing & Document Understanding** - Learn to parse, manipulate, and understand Excel, PDF, Word, and code files with LLMs.`,
};

