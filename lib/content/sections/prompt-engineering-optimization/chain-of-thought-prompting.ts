/**
 * Chain-of-Thought Prompting Section
 * Module 2: Prompt Engineering & Optimization
 */

export const chainofthoughtpromptingSection = {
    id: 'chain-of-thought-prompting',
    title: 'Chain-of-Thought Prompting',
    content: `# Chain-of-Thought Prompting

Master Chain-of-Thought (CoT) prompting to enable LLMs to solve complex problems through step-by-step reasoning.

## Overview: Teaching LLMs to Think Step-by-Step

Chain-of-Thought prompting is a breakthrough technique that dramatically improves LLM performance on complex tasks by encouraging the model to show its reasoning process.

### The Problem CoT Solves

\`\`\`python
# Without CoT: Direct answer (often wrong for complex problems)
prompt = "If John has 15 apples and gives away 1/3 to Mary and 2 to Bob, how many does he have left?"
# Model might jump to answer: "10" (incorrect)

# With CoT: Step-by-step reasoning (much more reliable)
prompt = """If John has 15 apples and gives away 1/3 to Mary and 2 to Bob, how many does he have left?

Let's solve this step by step:"""
# Model shows work:
# 1. John starts with 15 apples
# 2. 1/3 of 15 is 5 apples, so Mary gets 5
# 3. Bob gets 2 apples
# 4. Total given away: 5 + 2 = 7
# 5. 15 - 7 = 8 apples left
# Answer: 8 ✓
\`\`\`

## What is Chain-of-Thought Prompting?

### Definition

**Chain-of-Thought (CoT)** prompting is a technique where you prompt the model to generate intermediate reasoning steps before producing the final answer.

\`\`\`python
from openai import OpenAI

client = OpenAI()

def without_cot(problem: str) -> str:
    """Direct answer without reasoning."""
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "user", "content": f"{problem}\\n\\nAnswer:"}
        ]
    )
    return response.choices[0].message.content

def with_cot(problem: str) -> str:
    """Answer with step-by-step reasoning."""
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {
                "role": "user",
                "content": f"""{problem}

Let's solve this step by step:
1."""
            }
        ]
    )
    return response.choices[0].message.content

# Example problem
problem = """A store sells notebooks for $3 each. If you buy 10 or more, 
you get a 20% discount. How much would 12 notebooks cost?"""

print("Without CoT:")
print(without_cot(problem))
print("\\n" + "="*50 + "\\n")

print("With CoT:")
print(with_cot(problem))
# CoT version will show calculation steps, leading to more accurate answer
\`\`\`

## The Magic Phrase: "Let's Think Step by Step"

### Why It Works

This simple phrase activates reasoning mode in LLMs.

\`\`\`python
def cot_templates():
    """Common CoT trigger phrases that work well."""
    
    return {
        'general': "Let's think step by step:",
        'analytical': "Let's analyze this systematically:",
        'problem_solving': "Let's break this problem down:",
        'reasoning': "Let's reason through this carefully:",
        'methodical': "Let's approach this methodically:",
        'detailed': "Let's work through this in detail:",
    }

# Usage examples
def solve_with_cot(problem: str, trigger: str = "Let's think step by step:"):
    """Solve problem with CoT prompting."""
    
    prompt = f"""{problem}

{trigger}"""
    
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.choices[0].message.content

# Test with math problem
math_problem = "If a train travels 120 miles in 2 hours, how far will it travel in 5 hours at the same speed?"

print(solve_with_cot(math_problem))
# Model will show:
# 1. Calculate speed: 120 miles / 2 hours = 60 mph
# 2. Calculate distance: 60 mph × 5 hours = 300 miles
# Answer: 300 miles
\`\`\`

## Breaking Down Complex Problems

### Structured Step-by-Step Prompting

\`\`\`python
def create_structured_cot(
    problem: str,
    steps: list,
    constraints: list = None
) -> str:
    """
    Create a structured CoT prompt with explicit steps.
    More reliable than free-form "think step by step".
    """
    
    steps_text = "\\n".join([f"{i+1}. {step}" for i, step in enumerate(steps)])
    
    constraints_text = ""
    if constraints:
        constraints_text = "\\n\\nConstraints:\\n" + "\\n".join([f"- {c}" for c in constraints])
    
    return f"""{problem}

Solve this by following these steps:
{steps_text}
{constraints_text}

Solution:"""

# Example: Code debugging task
debugging_problem = """
This Python function should calculate factorial but returns wrong results:

def factorial(n):
    result = 0
    for i in range(n):
        result *= i
    return result

What's wrong?
"""

structured_cot = create_structured_cot(
    problem=debugging_problem,
    steps=[
        "Identify what the function is supposed to do",
        "Trace through the code with a simple example (e.g., n=3)",
        "Find where the logic goes wrong",
        "Explain why it's wrong",
        "Provide the corrected code"
    ],
    constraints=[
        "Show your work at each step",
        "Explain your reasoning",
        "Test the corrected code mentally"
    ]
)

print(structured_cot)
\`\`\`

### Domain-Specific CoT Templates

\`\`\`python
class CoTTemplates:
    """Production-ready CoT templates for different domains."""
    
    @staticmethod
    def mathematical_cot(problem: str) -> str:
        """CoT for math problems."""
        return f"""{problem}

Let's solve this mathematically:

Step 1: Identify what we're given
Step 2: Identify what we need to find
Step 3: Determine the formula or method
Step 4: Substitute values and calculate
Step 5: Verify the answer makes sense

Solution:"""
    
    @staticmethod
    def logical_reasoning_cot(problem: str) -> str:
        """CoT for logical reasoning."""
        return f"""{problem}

Let's reason through this logically:

Step 1: List all given facts
Step 2: Identify relationships and rules
Step 3: Draw logical inferences
Step 4: Eliminate impossibilities
Step 5: Reach conclusion

Reasoning:"""
    
    @staticmethod
    def code_analysis_cot(code: str, question: str) -> str:
        """CoT for code analysis."""
        return f"""Analyze this code:

{code}

Question: {question}

Let's analyze systematically:

Step 1: Understand what the code does
Step 2: Identify potential issues
Step 3: Trace execution with example inputs
Step 4: Evaluate against best practices
Step 5: Provide recommendations

Analysis:"""
    
    @staticmethod
    def debugging_cot(code: str, error: str) -> str:
        """CoT for debugging."""
        return f"""Code:
{code}

Error: {error}

Let's debug step by step:

Step 1: Understand the intended behavior
Step 2: Reproduce the error scenario
Step 3: Identify the line/section causing the error
Step 4: Determine root cause
Step 5: Propose fix with explanation

Debugging:"""
    
    @staticmethod
    def design_decision_cot(problem: str, options: list) -> str:
        """CoT for design decisions."""
        options_text = "\\n".join([f"- {opt}" for opt in options])
        
        return f"""{problem}

Options:
{options_text}

Let's evaluate systematically:

Step 1: Define success criteria
Step 2: Analyze each option against criteria
Step 3: Consider trade-offs
Step 4: Identify risks and mitigation
Step 5: Make recommendation

Evaluation:"""

# Usage examples
templates = CoTTemplates()

# Math problem
math_prompt = templates.mathematical_cot(
    "A rectangle has a length of 12 cm and a width of 8 cm. What is its area and perimeter?"
)

# Code analysis
code_prompt = templates.code_analysis_cot(
    code="def process(items):\\n    return [x*2 for x in items if x > 0]",
    question="Is this code efficient for large datasets?"
)

# Design decision
design_prompt = templates.design_decision_cot(
    problem="Choose database for a real-time chat application",
    options=["PostgreSQL", "MongoDB", "Redis", "Cassandra"]
)

print(design_prompt)
\`\`\`

## ReAct Pattern: Reasoning + Acting

### Combining Thought and Action

**ReAct** is a powerful extension of CoT that interleaves reasoning and actions (like tool calls).

\`\`\`python
def react_pattern(task: str, available_tools: list) -> str:
    """
    ReAct pattern: Thought -> Action -> Observation -> Thought -> ...
    Used in advanced agents like Cursor.
    """
    
    tools_text = "\\n".join([f"- {tool}" for tool in available_tools])
    
    return f"""{task}

Available tools:
{tools_text}

Use this pattern:
Thought: [Your reasoning about what to do next]
Action: [The tool/action to use]
Observation: [Result of the action]
... (repeat Thought/Action/Observation as needed)
Thought: [Final reasoning]
Answer: [Final answer]

Begin:

Thought:"""

# Example: Web search task
task = "Find the current population of Tokyo and compare it to New York City."
tools = [
    "search(query): Search the web",
    "calculate(expression): Perform calculations",
    "compare(a, b): Compare two values"
]

prompt = react_pattern(task, tools)
print(prompt)

# Model would generate:
# Thought: I need to find population data for both cities
# Action: search("Tokyo population 2024")
# Observation: Tokyo has 37.4 million people
# Thought: Now I need New York's population
# Action: search("New York City population 2024")
# Observation: New York City has 8.3 million people
# Thought: Now I can compare them
# Action: compare(37.4, 8.3)
# Observation: Tokyo is 4.5x larger
# Answer: Tokyo (37.4M) has 4.5 times more people than NYC (8.3M)
\`\`\`

### Implementing ReAct with Function Calling

\`\`\`python
from typing import List, Dict, Callable
from openai import OpenAI
import json

class ReActAgent:
    """
    Implement ReAct pattern with real tool execution.
    This is how Cursor makes decisions about code edits.
    """
    
    def __init__(self, tools: Dict[str, Callable]):
        """
        Args:
            tools: Dict of {tool_name: tool_function}
        """
        self.client = OpenAI()
        self.tools = tools
        self.max_iterations = 5
    
    def run(self, task: str) -> str:
        """
        Execute task using ReAct pattern.
        """
        
        tools_description = "\\n".join([
            f"- {name}: {func.__doc__}"
            for name, func in self.tools.items()
        ])
        
        conversation = []
        system_prompt = f"""You are a helpful assistant that solves tasks step-by-step.

Available tools:
{tools_description}

Use this format:
Thought: [reasoning about what to do]
Action: tool_name(arguments)
Observation: [result will be provided]

Repeat until you can provide the final answer.
When done: Answer: [your final answer]"""
        
        conversation.append({"role": "system", "content": system_prompt})
        conversation.append({"role": "user", "content": task})
        
        for iteration in range(self.max_iterations):
            # Get model's next thought/action
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=conversation
            )
            
            assistant_message = response.choices[0].message.content
            conversation.append({"role": "assistant", "content": assistant_message})
            
            print(f"\\nIteration {iteration + 1}:")
            print(assistant_message)
            
            # Check if done
            if "Answer:" in assistant_message:
                return assistant_message.split("Answer:")[-1].strip()
            
            # Extract and execute action
            if "Action:" in assistant_message:
                action_line = [line for line in assistant_message.split("\\n") if line.startswith("Action:")][0]
                action = action_line.replace("Action:", "").strip()
                
                # Parse tool call (simple parsing)
                try:
                    tool_name = action.split("(")[0]
                    args_str = action.split("(")[1].split(")")[0]
                    
                    if tool_name in self.tools:
                        # Execute tool
                        result = self.tools[tool_name](args_str)
                        
                        # Add observation
                        observation = f"Observation: {result}"
                        conversation.append({"role": "user", "content": observation})
                        
                        print(observation)
                except Exception as e:
                    print(f"Error executing action: {e}")
                    break
        
        return "Max iterations reached without answer"

# Define tools
def search(query: str) -> str:
    """Search for information (simulated)."""
    # In production, this would call a real search API
    mock_results = {
        "python tutorial": "Python is a high-level programming language...",
        "factorial algorithm": "Factorial can be computed iteratively or recursively...",
    }
    return mock_results.get(query.strip('"'), f"Results for: {query}")

def calculate(expression: str) -> str:
    """Calculate mathematical expression."""
    try:
        result = eval(expression)  # In production, use safe math parser
        return str(result)
    except:
        return "Error in calculation"

# Create agent
agent = ReActAgent({
    'search': search,
    'calculate': calculate
})

# Run task
result = agent.run("What is 15 factorial divided by 100?")
print(f"\\nFinal Result: {result}")
\`\`\`

## Self-Consistency: Multiple Reasoning Paths

### Generate Multiple Solutions and Vote

\`\`\`python
from collections import Counter

class SelfConsistencyCoT:
    """
    Generate multiple reasoning paths and use majority vote.
    More reliable than single CoT, used in production systems.
    """
    
    def __init__(self, model: str = "gpt-4"):
        self.client = OpenAI()
        self.model = model
    
    def solve_with_self_consistency(
        self,
        problem: str,
        num_paths: int = 5,
        temperature: float = 0.7
    ) -> Dict:
        """
        Solve problem with multiple reasoning paths.
        
        Args:
            problem: Problem to solve
            num_paths: Number of independent solutions to generate
            temperature: Higher = more diverse paths
            
        Returns:
            Dict with consensus answer and all paths
        """
        
        cot_prompt = f"""{problem}

Let's think step by step:"""
        
        solutions = []
        
        # Generate multiple reasoning paths
        for i in range(num_paths):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": cot_prompt}],
                temperature=temperature
            )
            
            reasoning = response.choices[0].message.content
            
            # Extract final answer (look for patterns like "Answer: X" or "= X")
            answer = self._extract_answer(reasoning)
            
            solutions.append({
                'reasoning': reasoning,
                'answer': answer
            })
        
        # Find consensus answer
        answers = [sol['answer'] for sol in solutions if sol['answer']]
        answer_counts = Counter(answers)
        consensus = answer_counts.most_common(1)[0] if answer_counts else (None, 0)
        
        return {
            'consensus_answer': consensus[0],
            'confidence': consensus[1] / num_paths if num_paths > 0 else 0,
            'all_solutions': solutions,
            'answer_distribution': dict(answer_counts)
        }
    
    def _extract_answer(self, text: str) -> str:
        """Extract final answer from reasoning text."""
        
        # Look for common answer patterns
        patterns = [
            "Answer:",
            "Final answer:",
            "Therefore,",
            "So the answer is",
            "="
        ]
        
        lines = text.lower().split("\\n")
        
        for line in reversed(lines):  # Start from end
            for pattern in patterns:
                if pattern.lower() in line:
                    # Extract everything after pattern
                    answer = line.split(pattern.lower())[-1].strip()
                    # Clean up
                    answer = answer.rstrip('.').strip()
                    return answer
        
        # If no pattern found, return last line
        return lines[-1].strip() if lines else ""

# Usage
consistency_solver = SelfConsistencyCoT()

problem = """A farmer has 17 sheep. All but 9 die. How many are left?"""

result = consistency_solver.solve_with_self_consistency(
    problem=problem,
    num_paths=3,
    temperature=0.7
)

print("Consensus Answer:", result['consensus_answer'])
print("Confidence:", f"{result['confidence']:.0%}")
print("\\nAnswer Distribution:", result['answer_distribution'])
print("\\nReasoning paths:")
for i, sol in enumerate(result['all_solutions'], 1):
    print(f"\\nPath {i}:")
    print(sol['reasoning'][:200] + "...")
    print(f"Answer: {sol['answer']}")
\`\`\`

## How Cursor Uses CoT for Code Generation

### Reverse Engineering Cursor's Approach

\`\`\`python
def cursor_style_cot(
    user_request: str,
    current_code: str,
    file_context: str
) -> str:
    """
    Approximate how Cursor uses CoT for code generation.
    """
    
    return f"""You are Cursor, an AI code editor.

Current code:
{current_code}

File context:
{file_context}

User request: {user_request}

Let's approach this systematically:

Step 1: Understand current code structure
[Analyze what's already there]

Step 2: Identify what needs to change
[Specific lines or sections to modify]

Step 3: Plan the implementation
[How to make the change while preserving existing functionality]

Step 4: Consider edge cases
[What could go wrong?]

Step 5: Generate the code change
[Minimal diff or complete new code]

Analysis:"""

# Example usage
request = "Add error handling to this function"
current = """
def divide(a, b):
    return a / b
"""
context = "This is a math utilities file with basic operations"

prompt = cursor_style_cot(request, current, context)
print(prompt)

# Cursor would generate reasoning like:
# Step 1: Current code is a simple division function, no error handling
# Step 2: Need to add try-except for ZeroDivisionError and type errors
# Step 3: Wrap in try block, catch exceptions, return None or raise with message
# Step 4: Edge cases: b=0, non-numeric inputs, None values
# Step 5: [generates code with proper error handling]
\`\`\`

## When CoT Improves Results

### Task Types That Benefit Most

\`\`\`python
def should_use_cot(task_type: str, complexity: str) -> Dict[str, any]:
    """
    Determine if CoT is beneficial for a task.
    """
    
    high_benefit_tasks = [
        'mathematical_reasoning',
        'logical_puzzles',
        'multi_step_problems',
        'code_debugging',
        'complex_analysis',
        'planning',
        'strategic_thinking'
    ]
    
    low_benefit_tasks = [
        'simple_classification',
        'keyword_extraction',
        'translation',
        'summarization',
        'sentiment_analysis'
    ]
    
    if task_type in high_benefit_tasks:
        recommendation = 'strongly_recommended'
        reason = f"{task_type} benefits significantly from step-by-step reasoning"
    elif task_type in low_benefit_tasks and complexity == 'simple':
        recommendation = 'not_necessary'
        reason = f"{task_type} is straightforward, CoT adds overhead without benefit"
    else:
        recommendation = 'consider'
        reason = "CoT might help but test both approaches"
    
    return {
        'use_cot': recommendation,
        'reason': reason,
        'expected_improvement': 'high' if task_type in high_benefit_tasks else 'low',
        'cost_increase': 'moderate'  # CoT uses more tokens
    }

# Examples
print(should_use_cot('mathematical_reasoning', 'complex'))
# {'use_cot': 'strongly_recommended', ...}

print(should_use_cot('sentiment_analysis', 'simple'))
# {'use_cot': 'not_necessary', ...}
\`\`\`

## Production CoT Best Practices

### Building a CoT System

\`\`\`python
from typing import Optional, Dict
from openai import OpenAI

class ProductionCoT:
    """
    Production-ready Chain-of-Thought system.
    """
    
    def __init__(self, model: str = "gpt-4"):
        self.client = OpenAI()
        self.model = model
    
    def solve(
        self,
        problem: str,
        task_type: str = 'general',
        steps: Optional[List[str]] = None,
        require_verification: bool = True,
        temperature: float = 0.3  # Lower for consistency
    ) -> Dict:
        """
        Solve problem with CoT, optionally verify.
        """
        
        # Build CoT prompt
        if steps:
            # Structured CoT with specific steps
            steps_text = "\\n".join([f"{i+1}. {step}" for i, step in enumerate(steps)])
            prompt = f"""{problem}

Solve this step by step:
{steps_text}

Solution:"""
        else:
            # Free-form CoT
            prompt = f"""{problem}

Let's solve this step by step:"""
        
        # Get solution
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature
        )
        
        reasoning = response.choices[0].message.content
        
        result = {
            'problem': problem,
            'reasoning': reasoning,
            'answer': self._extract_answer(reasoning),
            'verified': False
        }
        
        # Optional verification step
        if require_verification:
            verification = self._verify_solution(problem, reasoning)
            result['verified'] = verification['is_correct']
            result['verification_reasoning'] = verification['reasoning']
        
        return result
    
    def _extract_answer(self, reasoning: str) -> str:
        """Extract final answer from reasoning."""
        lines = reasoning.split("\\n")
        
        # Look for answer indicators
        for line in reversed(lines):
            if any(word in line.lower() for word in ['answer:', 'therefore:', 'result:']):
                return line.split(":")[-1].strip()
        
        return lines[-1].strip()
    
    def _verify_solution(self, problem: str, reasoning: str) -> Dict:
        """
        Verify the solution makes sense.
        Second pass to catch errors.
        """
        
        verification_prompt = f"""Given this problem and solution, verify if the reasoning is correct.

Problem: {problem}

Solution:
{reasoning}

Check:
1. Are all steps logically sound?
2. Is the arithmetic/logic correct?
3. Does the answer make sense?

Provide: "CORRECT" or "INCORRECT" with brief explanation."""
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": verification_prompt}],
            temperature=0
        )
        
        verification = response.choices[0].message.content
        
        return {
            'is_correct': 'correct' in verification.lower(),
            'reasoning': verification
        }

# Usage
cot_system = ProductionCoT()

problem = "A store has 45 items. They sell 3/5 of them. Then they restock 12 more. How many items do they have?"

result = cot_system.solve(
    problem=problem,
    steps=[
        "Calculate how many items were sold (3/5 of 45)",
        "Calculate remaining items after sales",
        "Add restocked items",
        "Verify calculation makes sense"
    ],
    require_verification=True
)

print("Problem:", result['problem'])
print("\\nReasoning:")
print(result['reasoning'])
print("\\nFinal Answer:", result['answer'])
print("\\nVerified:", result['verified'])
if 'verification_reasoning' in result:
    print("Verification:", result['verification_reasoning'])
\`\`\`

## Production Checklist

✅ **When to Use CoT**
- Complex reasoning tasks
- Multi-step problems
- Debugging and analysis
- Planning and strategy
- Mathematical reasoning

✅ **CoT Implementation**
- Use clear trigger phrases
- Structure steps when possible
- Keep temperature low (0.1-0.4)
- Extract and validate answers
- Consider self-consistency

✅ **Performance Optimization**
- Cache reasoning patterns
- Use structured CoT for consistency
- Implement verification steps
- Monitor token usage
- A/B test with/without CoT

✅ **Production Considerations**
- More tokens = higher cost
- Longer latency
- But much better accuracy
- Essential for complex tasks
- Test cost vs. quality tradeoff

## Key Takeaways

1. **CoT dramatically improves complex reasoning** - 30-50% better on hard tasks
2. **"Let's think step by step" is powerful** - Simple phrase, huge impact
3. **Structured CoT > free-form** - Explicit steps increase reliability
4. **ReAct combines thought and action** - Used in advanced agents
5. **Self-consistency increases confidence** - Multiple paths, majority vote
6. **Cursor uses CoT for code decisions** - Analyzes before generating
7. **Not always necessary** - Simple tasks don't need it, wastes tokens
8. **Verify reasoning when critical** - Second pass catches errors
9. **Lower temperature for CoT** - Want consistent reasoning, not creativity
10. **Production systems structure CoT** - Don't rely on free-form thinking

## Next Steps

Now that you understand Chain-of-Thought prompting, you're ready to explore **Prompt Optimization Techniques** - learning systematic approaches to improve prompt performance through testing and iteration.`,
};

