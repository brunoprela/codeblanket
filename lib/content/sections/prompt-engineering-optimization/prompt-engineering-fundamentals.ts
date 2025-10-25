/**
 * Prompt Engineering Fundamentals Section
 * Module 2: Prompt Engineering & Optimization
 */

export const promptengineeringfundamentalsSection = {
  id: 'prompt-engineering-fundamentals',
  title: 'Prompt Engineering Fundamentals',
  content: `# Prompt Engineering Fundamentals

Master the foundations of prompt engineering to build reliable, production-ready AI applications.

## Overview: Why Prompt Engineering Matters

Prompt engineering is the art and science of crafting instructions that get LLMs to produce exactly what you need—reliably, at scale, and cost-effectively.

### The Reality Check

**Bad prompt**: "Write code"
- Vague, unpredictable outputs
- Requires multiple attempts
- Wastes tokens and money
- Frustrates users

**Good prompt**: "Write a Python function called \`calculate_average\` that takes a list of numbers and returns their mean. Include docstring, type hints, and handle edge cases like empty lists."
- Clear, specific instructions
- Consistent outputs
- Works first time
- Production-ready

### Why This Matters in Production

When building tools like Cursor, Claude, or ChatGPT:
- **Reliability**: Users expect consistent behavior
- **Cost**: Bad prompts waste millions in API calls
- **Speed**: Good prompts need fewer retries
- **Quality**: Clear prompts = better outputs
- **Scalability**: Systematic approaches scale to thousands of use cases

## What is Prompt Engineering?

### Definition

**Prompt Engineering** is the practice of designing inputs to language models that reliably produce desired outputs.

It involves:
1. **Understanding model behavior** - How LLMs interpret instructions
2. **Crafting clear instructions** - What to say and how to say it
3. **Providing context** - Background information the model needs
4. **Structuring outputs** - Specifying format and constraints
5. **Iterating systematically** - Testing and refining prompts
6. **Version controlling** - Tracking what works and why

### Not Just Writing Text

Prompt engineering is **not** just writing casual instructions. It's:
- Software engineering applied to natural language
- Creating reusable templates and patterns
- Building systematic evaluation frameworks
- Optimizing for cost, latency, and quality
- Managing prompts like code (version control, testing, deployment)

## Anatomy of a Good Prompt

### The Core Components

Every effective prompt has these elements:

\`\`\`python
"""
1. ROLE/CONTEXT - Who is the AI and what's the situation?
2. TASK - What exactly should be done?
3. CONSTRAINTS - What rules must be followed?
4. FORMAT - How should output be structured?
5. EXAMPLES (optional) - Show desired behavior
"""

# Example: Good prompt structure
prompt = """
[ROLE] You are a Python expert helping developers write clean code.

[TASK] Refactor the following function to be more Pythonic:
{code}

[CONSTRAINTS]
- Use list comprehensions where appropriate
- Add type hints
- Follow PEP 8 style guide
- Keep under 20 lines

[FORMAT] Return only the refactored code with a brief comment explaining changes.
"""
\`\`\`

### Example: From Bad to Good

**❌ Bad Prompt:**
\`\`\`python
prompt = "Make this better: def foo (x): return [i*2 for i in x if i>0]"
\`\`\`

**✅ Good Prompt:**
\`\`\`python
prompt = """
You are a code reviewer focused on readability and best practices.

Task: Improve this Python function by:
1. Adding a descriptive name
2. Including a docstring
3. Adding type hints
4. Improving variable names

Original code:
def foo (x): return [i*2 for i in x if i>0]

Format your response as:
\`\`\`python
# Your improved code here
\`\`\`

Then explain what you changed and why.
"""
\`\`\`

## Instruction Following Principles

### Be Specific and Explicit

LLMs take instructions literally. Vagueness creates inconsistency.

\`\`\`python
from openai import OpenAI

client = OpenAI()

# ❌ Vague instruction
bad_prompt = "Summarize this article."

# ✅ Specific instruction
good_prompt = """
Summarize this article in exactly 3 bullet points.
Each bullet point should:
- Be one sentence (20-30 words)
- Cover a distinct main point
- Start with an action verb
- Be suitable for a busy executive

Article: {article_text}
"""

response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": good_prompt}]
)
\`\`\`

### Use Action Verbs

Start instructions with clear action verbs:

\`\`\`python
# Strong action verbs for different tasks
action_verbs = {
    'analysis': ['Analyze', 'Evaluate', 'Compare', 'Assess', 'Examine'],
    'creation': ['Generate', 'Create', 'Write', 'Compose', 'Design'],
    'transformation': ['Convert', 'Transform', 'Refactor', 'Rewrite', 'Adapt'],
    'extraction': ['Extract', 'Identify', 'Find', 'List', 'Detect'],
    'explanation': ['Explain', 'Describe', 'Clarify', 'Illustrate', 'Detail'],
}

# Example usage
prompt_template = """
{action} the following {input_type} to {desired_output}.

Input:
{input_data}

Requirements:
{requirements}
"""

# Concrete example
prompt = prompt_template.format(
    action="Extract",
    input_type="customer feedback",
    desired_output="list of feature requests with priority levels",
    input_data="The app is great but needs dark mode! Also, sync is buggy...",
    requirements="- Categorize by feature area\\n- Assign priority (high/medium/low)"
)
\`\`\`

### Provide Context

Context helps the model understand the situation and constraints.

\`\`\`python
def create_contextual_prompt(
    task: str,
    user_context: dict,
    constraints: list
) -> str:
    """
    Build a prompt with rich context.
    
    Example:
        task = "Generate SQL query"
        user_context = {
            'database': 'PostgreSQL',
            'tables': ['users', 'orders'],
            'user_level': 'beginner'
        }
        constraints = ['Use JOINs', 'Add comments']
    """
    
    context_block = "\\n".join([f"- {k}: {v}" for k, v in user_context.items()])
    constraints_block = "\\n".join([f"- {c}" for c in constraints])
    
    prompt = f"""
Task: {task}

Context:
{context_block}

Constraints:
{constraints_block}

Please provide your solution with clear explanations suitable for the user's level.
"""
    
    return prompt

# Usage
prompt = create_contextual_prompt(
    task="Generate SQL query to find top 10 customers by order value",
    user_context={
        'database': 'PostgreSQL 14',
        'tables': 'users (id, name), orders (id, user_id, total)',
        'user_level': 'beginner',
        'purpose': 'business report'
    },
    constraints=[
        'Include comments',
        'Use table aliases',
        'Format for readability'
    ]
)

print(prompt)
\`\`\`

## Few-Shot vs Zero-Shot Prompting

### Zero-Shot: No Examples

The model performs the task with just instructions, no examples.

\`\`\`python
# Zero-shot prompt
zero_shot = """
Classify the sentiment of this review as positive, negative, or neutral.
Output only the classification label.

Review: The product arrived late but works great.

Classification:"""

# Works well for simple, clear tasks
\`\`\`

### Few-Shot: Learning from Examples

Provide examples of desired behavior. More powerful and reliable.

\`\`\`python
# Few-shot prompt
few_shot = """
Classify the sentiment of reviews as positive, negative, or neutral.

Examples:

Review: "Amazing quality, exceeded expectations!"
Classification: positive

Review: "Terrible customer service, would not recommend."
Classification: negative

Review: "It\'s okay, nothing special."
Classification: neutral

Now classify this review:
Review: "The product arrived late but works great."
Classification:"""

# More reliable, especially for nuanced tasks
\`\`\`

### When to Use Each

\`\`\`python
def choose_prompting_strategy(
    task_complexity: str,
    model_capability: str,
    consistency_required: bool
) -> str:
    """
    Decide between zero-shot and few-shot.
    
    Returns:
        Strategy recommendation with reasoning
    """
    
    # Simple + powerful model = zero-shot OK
    if task_complexity == 'simple' and model_capability == 'high':
        if not consistency_required:
            return "zero-shot: Simple task, powerful model, flexibility OK"
    
    # Complex or consistency needed = few-shot better
    if task_complexity in ['complex', 'nuanced'] or consistency_required:
        return "few-shot: Provides clear examples, ensures consistency"
    
    # Default to few-shot for production
    return "few-shot: Safer default for production applications"

# Examples
print(choose_prompting_strategy('simple', 'high', False))
# "zero-shot: Simple task, powerful model, flexibility OK"

print(choose_prompting_strategy('nuanced', 'medium', True))
# "few-shot: Provides clear examples, ensures consistency"
\`\`\`

## Prompt Patterns and Templates

### Pattern 1: Task-Examples-Task (TEX)

Sandwich examples between task description and the actual task.

\`\`\`python
tex_pattern = """
{task_description}

Examples:
{examples}

Now complete this task:
{actual_input}
"""

# Concrete implementation
def tex_prompt (task: str, examples: list, input_data: str) -> str:
    """Create TEX pattern prompt."""
    
    example_text = "\\n\\n".join([
        f"Input: {ex['input']}\\nOutput: {ex['output']}"
        for ex in examples
    ])
    
    return tex_pattern.format(
        task_description=task,
        examples=example_text,
        actual_input=input_data
    )

# Usage: Extract dates from text
prompt = tex_prompt(
    task="Extract all dates from the text in YYYY-MM-DD format.",
    examples=[
        {
            'input': "Meeting on March 15th, 2024",
            'output': "2024-03-15"
        },
        {
            'input': "Due by Jan 1st next year",
            'output': "2025-01-01"
        }
    ],
    input_data="Deadline is February 28th, 2024 and review on March 3rd"
)
\`\`\`

### Pattern 2: Chain of Thought Template

Guide the model to show reasoning steps.

\`\`\`python
cot_template = """
{task}

Let\'s solve this step by step:

1. First, {first_step}
2. Then, {second_step}
3. Finally, {final_step}

Input: {input_data}

Solution:
"""

# Example: Math word problems
math_prompt = cot_template.format(
    task="Solve this math word problem",
    first_step="identify the key numbers and what they represent",
    second_step="determine which operation (s) to use",
    final_step="calculate and verify the answer makes sense",
    input_data="Sarah has 15 apples. She gives away 1/3 to her friend and eats 2. How many does she have left?"
)
\`\`\`

### Pattern 3: Format Specification Template

Enforce specific output formats.

\`\`\`python
format_template = """
{task}

Output format (JSON):
{{
  "{field1}": "{description1}",
  "{field2}": "{description2}",
  ...
}}

Input: {input_data}

Output:
"""

# Example: Parse job postings
job_parsing_prompt = """
Extract key information from this job posting.

Output format (JSON):
{{
  "title": "job title",
  "company": "company name",
  "location": "location",
  "salary_range": "salary range or null",
  "requirements": ["requirement1", "requirement2", ...],
  "key_skills": ["skill1", "skill2", ...]
}}

Input: {job_posting}

Output:
"""
\`\`\`

## Context and Specificity

### The Goldilocks Principle

Too little context → Poor outputs
Too much context → Wasted tokens, confusion
Just right → Optimal results

\`\`\`python
def optimize_context(
    base_prompt: str,
    available_context: dict,
    model_context_limit: int = 8000
) -> str:
    """
    Include only relevant context, respecting token limits.
    
    Args:
        base_prompt: Core instruction
        available_context: All possible context
        model_context_limit: Token budget
        
    Returns:
        Optimized prompt with selected context
    """
    import tiktoken
    
    # Token counting
    encoding = tiktoken.encoding_for_model("gpt-4")
    
    def count_tokens (text: str) -> int:
        return len (encoding.encode (text))
    
    base_tokens = count_tokens (base_prompt)
    remaining_tokens = model_context_limit - base_tokens - 500  # safety margin
    
    # Prioritize context by relevance
    context_priority = {
        'user_history': 0.8,
        'current_file': 0.9,
        'related_files': 0.6,
        'documentation': 0.5,
        'examples': 0.7,
    }
    
    # Sort by priority
    sorted_context = sorted(
        available_context.items(),
        key=lambda x: context_priority.get (x[0], 0.5),
        reverse=True
    )
    
    # Add context until budget exhausted
    selected_context = {}
    tokens_used = 0
    
    for key, value in sorted_context:
        tokens = count_tokens (str (value))
        if tokens_used + tokens <= remaining_tokens:
            selected_context[key] = value
            tokens_used += tokens
        else:
            break
    
    # Build final prompt
    context_str = "\\n\\n".join([
        f"{k.upper()}:\\n{v}"
        for k, v in selected_context.items()
    ])
    
    final_prompt = f"{context_str}\\n\\n{base_prompt}"
    
    return final_prompt

# Usage example
prompt = optimize_context(
    base_prompt="Refactor this function to be more efficient.",
    available_context={
        'current_file': "def slow_func (data): ...",  # High priority
        'user_history': "User prefers list comprehensions",  # High priority
        'documentation': "10 pages of Python docs...",  # Lower priority
        'related_files': "5 other files..."  # Medium priority
    },
    model_context_limit=4000
)
\`\`\`

## Testing Prompts Systematically

### Build a Prompt Testing Framework

\`\`\`python
from typing import List, Dict, Callable
from openai import OpenAI
import json

class PromptTester:
    """
    Systematic prompt testing and evaluation.
    Inspired by how Cursor tests prompt variations.
    """
    
    def __init__(self, model: str = "gpt-3.5-turbo"):
        self.client = OpenAI()
        self.model = model
        self.results = []
    
    def test_prompt(
        self,
        prompt_template: str,
        test_cases: List[Dict],
        evaluation_func: Callable,
        name: str = "unnamed"
    ) -> Dict:
        """
        Test a prompt against multiple test cases.
        
        Args:
            prompt_template: Template with {placeholders}
            test_cases: List of dicts with inputs and expected outputs
            evaluation_func: Function to score outputs (returns 0-1)
            name: Name for this test run
            
        Returns:
            Results dict with scores and examples
        """
        
        results = {
            'name': name,
            'prompt': prompt_template,
            'test_cases': len (test_cases),
            'scores': [],
            'examples': []
        }
        
        for i, test_case in enumerate (test_cases):
            # Fill template
            prompt = prompt_template.format(**test_case['input'])
            
            # Get LLM response
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}]
            )
            
            output = response.choices[0].message.content
            
            # Evaluate
            score = evaluation_func (output, test_case.get('expected'))
            
            results['scores'].append (score)
            
            # Save examples (first 3)
            if i < 3:
                results['examples'].append({
                    'input': test_case['input'],
                    'output': output,
                    'expected': test_case.get('expected'),
                    'score': score
                })
        
        # Summary stats
        results['avg_score'] = sum (results['scores']) / len (results['scores'])
        results['min_score'] = min (results['scores'])
        results['pass_rate'] = sum(1 for s in results['scores'] if s >= 0.8) / len (results['scores'])
        
        self.results.append (results)
        
        return results
    
    def compare_prompts (self, prompt_variants: List[Dict]) -> None:
        """
        Compare multiple prompt variants and print results.
        
        Args:
            prompt_variants: List of dicts with 'name', 'template', 'test_cases', 'eval_func'
        """
        
        print("\\n" + "="*70)
        print("PROMPT COMPARISON RESULTS")
        print("="*70)
        
        for variant in prompt_variants:
            results = self.test_prompt(
                prompt_template=variant['template'],
                test_cases=variant['test_cases'],
                evaluation_func=variant['eval_func'],
                name=variant['name']
            )
            
            print(f"\\n{variant['name']}")
            print(f"  Average Score: {results['avg_score']:.2f}")
            print(f"  Pass Rate: {results['pass_rate']:.1%}")
            print(f"  Min Score: {results['min_score']:.2f}")
        
        # Winner
        best = max (self.results, key=lambda x: x['avg_score'])
        print(f"\\n🏆 Winner: {best['name']} (score: {best['avg_score']:.2f})")
        print("="*70 + "\\n")

# Example usage
def sentiment_accuracy (output: str, expected: str) -> float:
    """Score sentiment classification accuracy."""
    output_clean = output.strip().lower()
    expected_clean = expected.strip().lower()
    return 1.0 if output_clean == expected_clean else 0.0

tester = PromptTester()

# Test cases
test_cases = [
    {
        'input': {'review': "Absolutely loved it! Best purchase ever."},
        'expected': 'positive'
    },
    {
        'input': {'review': "Terrible quality, broke after one day."},
        'expected': 'negative'
    },
    {
        'input': {'review': "It\'s okay, nothing special."},
        'expected': 'neutral'
    },
]

# Variant 1: Simple prompt
variant1 = {
    'name': 'Simple Instruction',
    'template': 'Classify sentiment: {review}\\nSentiment:',
    'test_cases': test_cases,
    'eval_func': sentiment_accuracy
}

# Variant 2: Detailed prompt
variant2 = {
    'name': 'Detailed with Examples',
    'template': ''Classify the sentiment as positive, negative, or neutral.

Examples:
"Great product!" → positive
"Didn't work." → negative  
"It\'s fine." → neutral

Review: {review}
Sentiment:'',
    'test_cases': test_cases,
    'eval_func': sentiment_accuracy
}

# Variant 3: Format-constrained
variant3 = {
    'name': 'Format Constrained',
    'template': ''Analyze sentiment. Output ONLY: positive, negative, or neutral

Review: {review}
Sentiment:'',
    'test_cases': test_cases,
    'eval_func': sentiment_accuracy
}

# Run comparison
tester.compare_prompts([variant1, variant2, variant3])
\`\`\`

## Versioning Prompts

### Treat Prompts Like Code

\`\`\`python
from datetime import datetime
from typing import Dict, Optional
import json

class PromptVersion:
    """
    Version control for prompts.
    Track changes, performance, and rollback if needed.
    """
    
    def __init__(self, prompt_id: str):
        self.prompt_id = prompt_id
        self.versions = {}
        self.current_version = None
        self.performance_data = {}
    
    def create_version(
        self,
        template: str,
        description: str,
        variables: Dict[str, str],
        metadata: Optional[Dict] = None
    ) -> str:
        """Create a new prompt version."""
        
        version_id = f"v{len (self.versions) + 1}"
        
        version_data = {
            'id': version_id,
            'template': template,
            'description': description,
            'variables': variables,
            'metadata': metadata or {},
            'created_at': datetime.now().isoformat(),
            'performance': {
                'uses': 0,
                'avg_latency': 0,
                'avg_cost': 0,
                'success_rate': 0
            }
        }
        
        self.versions[version_id] = version_data
        self.current_version = version_id
        
        return version_id
    
    def get_prompt (self, version: Optional[str] = None) -> str:
        """Get prompt template for a version."""
        version = version or self.current_version
        return self.versions[version]['template']
    
    def record_performance(
        self,
        version: str,
        latency: float,
        cost: float,
        success: bool
    ):
        """Record performance metrics for a version."""
        
        perf = self.versions[version]['performance']
        n = perf['uses']
        
        # Update moving averages
        perf['avg_latency'] = (perf['avg_latency'] * n + latency) / (n + 1)
        perf['avg_cost'] = (perf['avg_cost'] * n + cost) / (n + 1)
        perf['success_rate'] = (perf['success_rate'] * n + (1 if success else 0)) / (n + 1)
        perf['uses'] += 1
    
    def compare_versions (self, v1: str, v2: str) -> Dict:
        """Compare performance of two versions."""
        
        perf1 = self.versions[v1]['performance']
        perf2 = self.versions[v2]['performance']
        
        return {
            'latency_diff': perf2['avg_latency'] - perf1['avg_latency'],
            'cost_diff': perf2['avg_cost'] - perf1['avg_cost'],
            'success_diff': perf2['success_rate'] - perf1['success_rate'],
            'recommendation': self._recommend_version (perf1, perf2, v1, v2)
        }
    
    def _recommend_version (self, perf1, perf2, v1, v2):
        """Recommend which version to use."""
        
        # Weight factors
        success_weight = 0.5
        cost_weight = 0.3
        latency_weight = 0.2
        
        # Normalize and score (higher is better)
        score1 = (
            perf1['success_rate'] * success_weight +
            (1 / (perf1['avg_cost'] + 0.001)) * cost_weight +
            (1 / (perf1['avg_latency'] + 0.001)) * latency_weight
        )
        
        score2 = (
            perf2['success_rate'] * success_weight +
            (1 / (perf2['avg_cost'] + 0.001)) * cost_weight +
            (1 / (perf2['avg_latency'] + 0.001)) * latency_weight
        )
        
        return v1 if score1 > score2 else v2
    
    def export (self, filepath: str):
        """Export prompt versions to JSON."""
        with open (filepath, 'w') as f:
            json.dump({
                'prompt_id': self.prompt_id,
                'current_version': self.current_version,
                'versions': self.versions
            }, f, indent=2)

# Usage example
code_review_prompt = PromptVersion("code_review_v1")

# Version 1: Basic
code_review_prompt.create_version(
    template="""Review this code and suggest improvements:

{code}

Provide feedback on:
- Code quality
- Potential bugs
- Best practices
""",
    description="Initial basic version",
    variables={'code': 'str'}
)

# Version 2: More structured
v2 = code_review_prompt.create_version(
    template="""You are an expert code reviewer. Analyze this code systematically.

CODE:
{code}

Provide:
1. BUGS: Any logical errors or issues
2. QUALITY: Code style and readability
3. PERFORMANCE: Efficiency concerns
4. SECURITY: Potential vulnerabilities

Be specific and actionable.
""",
    description="Added structure and categories",
    variables={'code': 'str'}
)

# Simulate performance tracking
code_review_prompt.record_performance (v2, latency=1.2, cost=0.003, success=True)
code_review_prompt.record_performance (v2, latency=1.5, cost=0.004, success=True)

# Export
code_review_prompt.export('prompt_versions.json')
\`\`\`

## Common Mistakes and How to Avoid Them

### Mistake 1: Ambiguous Instructions

\`\`\`python
# ❌ Ambiguous
bad = "Summarize this"

# ✅ Clear
good = "Summarize this article in 3 sentences, focusing on main findings and their implications"
\`\`\`

### Mistake 2: No Output Format

\`\`\`python
# ❌ Format unclear
bad = "Extract the names from this text"

# ✅ Format specified
good = """Extract person names from this text.
Output format: ["Name1", "Name2", ...]
Only include full names, ignore mentions of first names only."""
\`\`\`

### Mistake 3: Overloading Single Prompt

\`\`\`python
# ❌ Too many tasks
bad = "Translate to Spanish, summarize, extract keywords, and check sentiment"

# ✅ Split into steps
step1 = "Translate this to Spanish: {text}"
step2 = "Summarize this Spanish text in 2 sentences: {translated}"
step3 = "Extract 5 keywords from: {summary}"
step4 = "Classify sentiment: {summary}"
\`\`\`

### Mistake 4: Assuming Context

\`\`\`python
# ❌ Assumes model knows context
bad = "Continue the pattern"

# ✅ Provides full context
good = """This sequence follows the pattern: f (n) = 2n + 1
Sequence so far: 1, 3, 5, 7, 9

Generate the next 5 numbers following this pattern."""
\`\`\`

## Production Checklist

✅ **Clear Instructions**
- Use specific action verbs
- Define expected behavior
- Include all necessary context
- Specify output format

✅ **Systematic Testing**
- Create test cases
- Measure performance
- Compare variants
- Track over time

✅ **Version Control**
- Version all prompts
- Track performance
- Enable rollback
- Document changes

✅ **Output Validation**
- Define success criteria
- Validate format
- Check content quality
- Handle failures gracefully

✅ **Cost Monitoring**
- Track token usage
- Measure per-prompt cost
- Optimize prompt length
- Balance quality vs cost

## Key Takeaways

1. **Prompt engineering is software engineering** - Treat prompts like code
2. **Specificity matters** - Vague prompts = inconsistent results
3. **Structure your prompts** - Role, task, constraints, format, examples
4. **Test systematically** - Build evaluation frameworks
5. **Version and track** - Monitor performance over time
6. **Few-shot > zero-shot** - Examples dramatically improve reliability
7. **Context is critical** - Provide what's needed, nothing more
8. **Start simple, iterate** - Begin with basics, refine based on results
9. **Format constraints work** - Specify exact output format
10. **Production needs consistency** - Design for reliability, not one-off success

## Next Steps

Now that you understand prompt engineering fundamentals, you're ready to dive into **System Prompts & Role Assignment** - learning how to define AI behavior and personality for your applications.`,
};
