/**
 * System Prompts & Role Assignment Section
 * Module 2: Prompt Engineering & Optimization
 */

export const systempromptsroleassignmentSection = {
  id: 'system-prompts-role-assignment',
  title: 'System Prompts & Role Assignment',
  content: `# System Prompts & Role Assignment

Master the art of defining AI personality, behavior, and capabilities through system prompts.

## Overview: The Power of System Prompts

System prompts are the foundation of AI behavior. They define:
- **Who the AI is** (role and expertise)
- **How it behaves** (tone, style, personality)
- **What it can/cannot do** (capabilities and boundaries)
- **How it formats outputs** (structure and consistency)

### Real-World Impact

**Cursor\'s system prompt** makes it act like a coding assistant that understands your codebase, suggests edits, and follows your style guidelines.

**ChatGPT's system prompt** makes it helpful, harmless, and honest - defining its entire personality.

**Claude's system prompt** emphasizes safety, nuance, and instruction-following.

## What is a System Prompt?

### The Persistent Context

A **system prompt** is a special message that:
- Sets the AI's role and behavior
- Persists across the entire conversation
- Has higher authority than user messages
- Defines capabilities and constraints
- Establishes output formats

\`\`\`python
from openai import OpenAI

client = OpenAI()

# System prompt defines behavior
response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {
            "role": "system",
            "content": "You are a helpful Python tutor who explains concepts simply with code examples."
        },
        {
            "role": "user",
            "content": "What are decorators?"
        }
    ]
)

print(response.choices[0].message.content)
# Response will be: tutorial-style, Python-focused, with examples
\`\`\`

### System vs User Messages

\`\`\`python
"""
KEY DIFFERENCES:

SYSTEM MESSAGE:
- Sets persistent behavior
- Higher priority than user messages
- Defines the "character" of the AI
- Usually set once at conversation start
- Model treats as authoritative instructions

USER MESSAGE:
- Specific tasks or questions
- Can vary each turn
- What the user wants right now
- Model tries to satisfy while staying in role
"""

# Example showing the difference
def compare_with_without_system():
    client = OpenAI()
    
    # WITHOUT system prompt
    response1 = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": "Explain recursion"}
        ]
    )
    
    # WITH system prompt
    response2 = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are a computer science professor. Explain concepts with formal precision, academic terminology, and cite theoretical foundations."
            },
            {"role": "user", "content": "Explain recursion"}
        ]
    )
    
    print("Without system prompt:")
    print(response1.choices[0].message.content[:200])
    print("\\n" + "="*50 + "\\n")
    print("With system prompt:")
    print(response2.choices[0].message.content[:200])

compare_with_without_system()
\`\`\`

## Role Definition Best Practices

### Be Specific About Expertise

\`\`\`python
# ❌ Vague role
bad_system = "You are a helpful assistant."

# ✅ Specific role with expertise
good_system = """You are a senior full-stack developer with 10 years of experience specializing in:
- React and TypeScript frontend
- Python/FastAPI backend
- PostgreSQL databases
- AWS cloud infrastructure

You provide production-ready code with best practices, proper error handling, and clear comments."""

# The difference in outputs is dramatic
\`\`\`

### Define Personality Traits

\`\`\`python
def create_persona_system_prompt(
    role: str,
    expertise: list,
    personality: dict,
    constraints: list
) -> str:
    """
    Build a comprehensive system prompt with personality.
    
    Args:
        role: Primary role (e.g., "Python tutor")
        expertise: List of areas of knowledge
        personality: Dict of traits (e.g., {"tone": "friendly", "style": "concise"})
        constraints: List of rules/limitations
    """
    
    expertise_str = "\\n".join([f"- {item}" for item in expertise])
    personality_str = "\\n".join([f"- {k}: {v}" for k, v in personality.items()])
    constraints_str = "\\n".join([f"- {item}" for item in constraints])
    
    system_prompt = f"""You are {role}.

EXPERTISE:
{expertise_str}

PERSONALITY:
{personality_str}

CONSTRAINTS:
{constraints_str}

Embody this role consistently in all responses."""
    
    return system_prompt

# Example: Code reviewer
code_reviewer = create_persona_system_prompt(
    role="an experienced code reviewer at a top tech company",
    expertise=[
        "Clean code principles (SOLID, DRY, KISS)",
        "Security best practices",
        "Performance optimization",
        "Testing strategies"
    ],
    personality={
        "tone": "constructive and encouraging",
        "style": "specific and actionable",
        "approach": "focus on teaching, not just fixing"
    },
    constraints=[
        "Always explain WHY, not just WHAT to change",
        "Prioritize critical issues over style preferences",
        "Provide code examples for suggestions",
        "Be respectful of developer's skill level"
    ]
)

print(code_reviewer)
\`\`\`

### Examples of Effective Role Definitions

\`\`\`python
# 1. Technical Documentation Writer
docs_writer = """You are a technical documentation specialist who creates clear, comprehensive docs.

Your writing:
- Uses active voice and present tense
- Includes practical examples for every concept
- Follows the "explain, demonstrate, practice" pattern
- Organizes information hierarchically
- Anticipates reader questions

Always include:
1. Brief overview (2-3 sentences)
2. When/why to use this
3. Step-by-step instructions
4. Code examples with comments
5. Common pitfalls and troubleshooting"""

# 2. Data Analyst
data_analyst = """You are a senior data analyst with expertise in Python (pandas, numpy), SQL, and statistics.

Your analysis approach:
- Start with understanding the business question
- Explore data quality and distributions first
- Use appropriate statistical methods
- Create clear visualizations
- Provide actionable insights

Your outputs:
- Include both code and narrative explanation
- Show intermediate steps in analysis
- Highlight assumptions and limitations
- Suggest next steps or deeper analyses"""

# 3. Product Manager
product_manager = """You are an experienced product manager who thinks strategically about features and user needs.

Your mindset:
- User needs come before technical constraints
- Every feature needs measurable success criteria
- Trade-offs must be explicit
- Consider the full user journey

When analyzing features:
1. User problem being solved
2. Success metrics
3. Technical feasibility
4. Resource requirements
5. Risks and mitigation strategies
6. Alternative solutions"""

# 4. Security Expert
security_expert = """You are a cybersecurity specialist focused on secure coding practices.

Your priorities:
1. Prevent common vulnerabilities (OWASP Top 10)
2. Implement defense in depth
3. Follow principle of least privilege
4. Validate and sanitize all inputs

Review code for:
- SQL injection vulnerabilities
- XSS attack vectors
- Authentication/authorization flaws
- Sensitive data exposure
- Insecure dependencies

Provide specific remediation code, not just vulnerability descriptions."""
\`\`\`

## Setting Behavioral Guidelines

### Tone and Style Control

\`\`\`python
class TonePresets:
    """Pre-defined tone configurations for different use cases."""
    
    PROFESSIONAL = """Maintain a professional, business-appropriate tone:
- Use formal language
- Be concise and direct
- Avoid humor or casualness
- Focus on facts and data"""
    
    FRIENDLY = """Use a warm, friendly, conversational tone:
- Be approachable and encouraging
- Use casual language when appropriate
- Show empathy and understanding
- Make the interaction enjoyable"""
    
    EDUCATIONAL = """Adopt a patient, teaching-focused tone:
- Explain concepts thoroughly
- Use analogies and examples
- Encourage questions
- Build on prior knowledge
- Celebrate understanding"""
    
    TECHNICAL = """Use precise technical communication:
- Employ domain-specific terminology
- Be exact and unambiguous
- Include relevant technical details
- Cite standards and best practices
- Assume technical competence"""
    
    EXECUTIVE = """Communicate for executive audience:
- Lead with key findings/recommendations
- Be extremely concise
- Focus on business impact
- Quantify everything possible
- No technical jargon without explanation"""

def build_system_with_tone (role: str, tone_preset: str) -> str:
    """Combine role with tone preset."""
    return f"""{role}

COMMUNICATION STYLE:
{tone_preset}

Maintain this tone consistently throughout the conversation."""

# Usage
system_prompt = build_system_with_tone(
    role="You are a senior software architect reviewing system designs.",
    tone_preset=TonePresets.TECHNICAL
)
\`\`\`

### Output Format Instructions

\`\`\`python
# System prompt with strict formatting
formatted_system = """You are a code reviewer providing structured feedback.

ALWAYS format your response exactly as:

## Summary
[One sentence overall assessment]

## Critical Issues
[List of must-fix problems with severity: HIGH/MEDIUM/LOW]

## Suggestions
[List of improvements with effort: QUICK/MODERATE/LARGE]

## Positive Notes
[Things done well - always include at least one]

Use markdown formatting. Be specific with line numbers when referencing code."""

# Example usage
from openai import OpenAI

client = OpenAI()

def review_code (code: str) -> str:
    """Get formatted code review."""
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": formatted_system},
            {"role": "user", "content": f"Review this code:\\n\\n{code}"}
        ]
    )
    return response.choices[0].message.content

# The output will ALWAYS follow the specified format
\`\`\`

## Constraints and Boundaries

### What the AI Can and Cannot Do

\`\`\`python
def create_bounded_system_prompt(
    role: str,
    capabilities: list,
    limitations: list,
    forbidden_actions: list
) -> str:
    """
    Define clear boundaries for AI behavior.
    Critical for production safety.
    """
    
    capabilities_str = "\\n".join([f"✓ {item}" for item in capabilities])
    limitations_str = "\\n".join([f"⚠ {item}" for item in limitations])
    forbidden_str = "\\n".join([f"✗ {item}" for item in forbidden_actions])
    
    return f"""You are {role}.

YOU CAN:
{capabilities_str}

YOUR LIMITATIONS:
{limitations_str}

YOU MUST NOT:
{forbidden_str}

If asked to do something outside your capabilities or forbidden, politely explain why you cannot do it and suggest alternatives."""

# Example: Financial advisor AI
financial_advisor = create_bounded_system_prompt(
    role="a financial education assistant",
    capabilities=[
        "Explain financial concepts and terminology",
        "Describe different investment strategies",
        "Help understand financial statements",
        "Provide educational resources"
    ],
    limitations=[
        "Cannot provide personalized investment advice",
        "Cannot predict market movements",
        "Cannot guarantee returns",
        "Cannot access real-time market data"
    ],
    forbidden_actions=[
        "Give specific stock recommendations",
        "Make investment decisions for users",
        "Guarantee financial outcomes",
        "Act as a licensed financial advisor"
    ]
)

print(financial_advisor)
\`\`\`

### Content Filtering in System Prompts

\`\`\`python
# Build-in content filters
content_safe_system = """You are a helpful assistant for a family-friendly application.

CONTENT POLICY:
- Keep all content appropriate for ages 13+
- Avoid controversial political topics
- No explicit violence, profanity, or adult content
- If asked for inappropriate content, politely decline and redirect

MODERATION:
- If you're unsure about content appropriateness, err on the side of caution
- Explain why certain content cannot be provided
- Suggest appropriate alternatives

Remember: User safety and appropriate content are your top priorities."""
\`\`\`

## Scope Limitations

### Keeping AI Focused

\`\`\`python
def create_focused_system_prompt(
    domain: str,
    in_scope: list,
    out_of_scope: list
) -> str:
    """
    Create a system prompt that keeps AI focused on specific domain.
    Prevents scope creep and off-topic responses.
    """
    
    in_scope_str = "\\n".join([f"✓ {item}" for item in in_scope])
    out_of_scope_str = "\\n".join([f"✗ {item}" for item in out_of_scope])
    
    return f"""You are a specialist in {domain}.

YOUR FOCUS AREAS (HELP WITH THESE):
{in_scope_str}

OUTSIDE YOUR SCOPE (POLITELY REDIRECT):
{out_of_scope_str}

When asked about out-of-scope topics, respond: "That\'s outside my area of expertise in {domain}. I recommend consulting a specialist in [relevant field]." """

# Example: SQL assistant
sql_assistant = create_focused_system_prompt(
    domain="SQL database queries and optimization",
    in_scope=[
        "Writing SQL queries (SELECT, INSERT, UPDATE, DELETE)",
        "Query optimization and indexing",
        "Database schema design",
        "Explaining SQL concepts",
        "Debugging SQL errors"
    ],
    out_of_scope=[
        "Application code (Python, JavaScript, etc.)",
        "Frontend development",
        "Database administration (backups, user management)",
        "Non-SQL databases (MongoDB, Redis, etc.)",
        "Server configuration"
    ]
)
\`\`\`

## How Cursor Defines Its System Prompt

### Reverse Engineering Cursor's Behavior

\`\`\`python
# Approximation of Cursor's system prompt
cursor_style_system = """You are an expert AI coding assistant integrated into a code editor.

CONTEXT AWARENESS:
- You have access to the user's current file and cursor position
- You understand the broader codebase structure
- You know recent edits and user patterns

YOUR CAPABILITIES:
- Generate code edits in diff format
- Refactor existing code
- Fix bugs and suggest improvements
- Write new functions/classes from descriptions
- Explain complex code sections

OUTPUT FORMAT:
- For edits: Provide minimal, precise diffs
- For new code: Generate complete, production-ready implementations
- Always preserve user's coding style
- Include necessary imports and type hints
- Add brief comments for complex logic

CODE QUALITY STANDARDS:
- Follow language-specific best practices
- Write maintainable, readable code
- Consider edge cases and error handling
- Prefer simple solutions over clever ones
- Match the existing code style in the file

INTERACTION STYLE:
- Be concise - users want code, not explanations (unless asked)
- If ambiguous, ask clarifying questions
- Suggest improvements proactively but don't be pushy
- Explain reasoning only when it's not obvious"""

# Example of how Cursor might use this
def cursor_style_interaction():
    from openai import OpenAI
    
    client = OpenAI()
    
    # User request
    user_code = """
def calculate_total (items):
    total = 0
    for item in items:
        total += item['price']
    return total
"""
    
    user_request = "Add error handling and type hints"
    
    # Cursor\'s system prompt + context
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": cursor_style_system},
            {
                "role": "user",
                "content": f"""Current code:
{user_code}

Request: {user_request}

Provide the refactored code."""
            }
        ]
    )
    
    print(response.choices[0].message.content)
    # Output will be: code-focused, with type hints, error handling,
    # following best practices, minimal explanation

cursor_style_interaction()
\`\`\`

## Building a Production System Prompt Library

### Reusable System Prompt Components

\`\`\`python
from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class SystemPromptComponent:
    """A reusable component of system prompts."""
    name: str
    content: str
    category: str  # 'role', 'tone', 'format', 'constraint'

class SystemPromptBuilder:
    """
    Build complex system prompts from reusable components.
    Inspired by how production AI apps manage prompts.
    """
    
    def __init__(self):
        self.components: Dict[str, SystemPromptComponent] = {}
        self._register_default_components()
    
    def _register_default_components (self):
        """Register commonly used components."""
        
        # Role components
        self.register(SystemPromptComponent(
            name="expert_developer",
            content="You are an expert software developer with deep knowledge of modern development practices.",
            category="role"
        ))
        
        self.register(SystemPromptComponent(
            name="code_quality_focus",
            content="""Focus on code quality:
- Write clean, maintainable code
- Follow SOLID principles
- Include proper error handling
- Add type hints and documentation""",
            category="behavior"
        ))
        
        # Tone components
        self.register(SystemPromptComponent(
            name="concise_technical",
            content="Be concise and technical. Avoid unnecessary explanations. Users are experienced developers.",
            category="tone"
        ))
        
        self.register(SystemPromptComponent(
            name="educational_friendly",
            content="Be patient and educational. Explain concepts clearly. Assume user is learning.",
            category="tone"
        ))
        
        # Format components
        self.register(SystemPromptComponent(
            name="code_only_output",
            content="Output only code unless specifically asked for explanations. No markdown formatting unless requested.",
            category="format"
        ))
        
        self.register(SystemPromptComponent(
            name="structured_response",
            content="""Structure all responses as:
1. Brief summary
2. Detailed explanation
3. Code example
4. Potential issues/considerations""",
            category="format"
        ))
        
        # Constraint components
        self.register(SystemPromptComponent(
            name="python_only",
            content="Only provide Python solutions. If asked about other languages, convert to Python equivalent.",
            category="constraint"
        ))
        
        self.register(SystemPromptComponent(
            name="security_conscious",
            content="""Always consider security:
- Validate inputs
- Avoid SQL injection
- Never hardcode secrets
- Use parameterized queries
- Follow OWASP guidelines""",
            category="constraint"
        ))
    
    def register (self, component: SystemPromptComponent):
        """Register a new component."""
        self.components[component.name] = component
    
    def build(
        self,
        role: str,
        components: List[str],
        custom_instructions: Optional[str] = None
    ) -> str:
        """
        Build a complete system prompt from components.
        
        Args:
            role: Base role (or name of role component)
            components: List of component names to include
            custom_instructions: Additional custom instructions
            
        Returns:
            Complete system prompt
        """
        
        # Start with role
        if role in self.components:
            prompt_parts = [self.components[role].content]
        else:
            prompt_parts = [role]
        
        # Add selected components by category
        categories = ['behavior', 'tone', 'format', 'constraint']
        
        for category in categories:
            category_components = [
                self.components[name].content
                for name in components
                if name in self.components and self.components[name].category == category
            ]
            
            if category_components:
                prompt_parts.append (f"\\n{category.upper()}:")
                prompt_parts.extend (category_components)
        
        # Add custom instructions
        if custom_instructions:
            prompt_parts.append (f"\\nADDITIONAL INSTRUCTIONS:\\n{custom_instructions}")
        
        return "\\n\\n".join (prompt_parts)
    
    def list_components (self, category: Optional[str] = None) -> List[str]:
        """List available components, optionally filtered by category."""
        if category:
            return [
                name for name, comp in self.components.items()
                if comp.category == category
            ]
        return list (self.components.keys())

# Usage example
builder = SystemPromptBuilder()

# Build a code review system prompt
code_review_prompt = builder.build(
    role="expert_developer",
    components=[
        "code_quality_focus",
        "concise_technical",
        "structured_response",
        "security_conscious"
    ],
    custom_instructions="Focus on Python best practices and suggest pytest test cases for any function."
)

print(code_review_prompt)
print("\\n" + "="*50 + "\\n")

# Build a teaching assistant prompt
teaching_prompt = builder.build(
    role="You are a patient programming instructor",
    components=[
        "educational_friendly",
        "structured_response",
        "python_only"
    ],
    custom_instructions="Always include a 'Try it yourself' exercise at the end."
)

print(teaching_prompt)
\`\`\`

## A/B Testing System Prompts

### Comparing System Prompt Variations

\`\`\`python
from typing import List, Dict, Callable
from openai import OpenAI
import time

class SystemPromptExperiment:
    """
    A/B test different system prompts to find the best one.
    Essential for production optimization.
    """
    
    def __init__(self, model: str = "gpt-3.5-turbo"):
        self.client = OpenAI()
        self.model = model
    
    def run_experiment(
        self,
        variants: Dict[str, str],
        test_queries: List[str],
        evaluation_func: Callable,
        runs_per_variant: int = 3
    ) -> Dict:
        """
        Test multiple system prompt variants.
        
        Args:
            variants: Dict of {name: system_prompt}
            test_queries: List of user queries to test
            evaluation_func: Function to score responses (0-1)
            runs_per_variant: How many times to run each test
            
        Returns:
            Results with scores and recommendations
        """
        
        results = {
            variant_name: {
                'scores': [],
                'avg_latency': 0,
                'total_tokens': 0,
                'responses': []
            }
            for variant_name in variants
        }
        
        # Test each variant
        for variant_name, system_prompt in variants.items():
            print(f"Testing variant: {variant_name}...")
            
            latencies = []
            token_counts = []
            
            for query in test_queries:
                for _ in range (runs_per_variant):
                    start = time.time()
                    
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": query}
                        ]
                    )
                    
                    latency = time.time() - start
                    latencies.append (latency)
                    
                    output = response.choices[0].message.content
                    tokens = response.usage.total_tokens
                    token_counts.append (tokens)
                    
                    # Evaluate
                    score = evaluation_func (output, query)
                    results[variant_name]['scores'].append (score)
                    
                    # Save first response as example
                    if len (results[variant_name]['responses']) < len (test_queries):
                        results[variant_name]['responses'].append({
                            'query': query,
                            'response': output,
                            'score': score
                        })
            
            # Aggregate metrics
            results[variant_name]['avg_latency'] = sum (latencies) / len (latencies)
            results[variant_name]['total_tokens'] = sum (token_counts)
            results[variant_name]['avg_score'] = sum (results[variant_name]['scores']) / len (results[variant_name]['scores'])
        
        return results
    
    def print_results (self, results: Dict):
        """Print formatted experiment results."""
        
        print("\\n" + "="*70)
        print("SYSTEM PROMPT EXPERIMENT RESULTS")
        print("="*70 + "\\n")
        
        for variant_name, metrics in results.items():
            print(f"{variant_name}:")
            print(f"  Average Score: {metrics['avg_score']:.3f}")
            print(f"  Avg Latency: {metrics['avg_latency']:.2f}s")
            print(f"  Total Tokens: {metrics['total_tokens']}")
            print()
        
        # Determine winner
        winner = max (results.items(), key=lambda x: x[1]['avg_score'])
        print(f"🏆 Winner: {winner[0]} (score: {winner[1]['avg_score']:.3f})")
        print("="*70 + "\\n")
        
        # Show example responses from winner
        print(f"Example responses from {winner[0]}:\\n")
        for i, example in enumerate (winner[1]['responses'][:2], 1):
            print(f"Query {i}: {example['query']}")
            print(f"Response: {example['response'][:150]}...")
            print(f"Score: {example['score']:.2f}\\n")

# Example usage
def evaluate_code_quality (response: str, query: str) -> float:
    """Simple evaluation: check for key quality indicators."""
    score = 0.0
    
    # Check for code block
    if "\`\`\`" in response:
        score += 0.3
    
    # Check for type hints
    if "def " in response and "->" in response:
    score += 0.2
    
    # Check for docstring
    if '"""' in response:
    score += 0.2
    
    # Check for error handling
    if "try" in response or "except" in response or "raise" in response:
    score += 0.2
    
    # Check for comments
    if response.count("#") >= 2:
    score += 0.1
    
    return min (score, 1.0)

# Define variants to test
variants = {
        "Basic": "You are a helpful coding assistant.",

        "Detailed": """You are an expert Python developer who writes clean, well-documented code.
Always include:
            - Type hints
- Docstrings
    - Error handling
- Comments for complex logic""",

"Concise": """You are a senior developer. Write production-ready Python code.
Be concise.Code speaks for itself.Include only essential comments.""",
}

# Test queries
test_queries = [
    "Write a function to calculate factorial",
    "Create a class for managing a shopping cart",
    "Write a function to validate email addresses"
]

# Run experiment
experiment = SystemPromptExperiment()
results = experiment.run_experiment(
    variants = variants,
    test_queries = test_queries,
    evaluation_func = evaluate_code_quality,
    runs_per_variant = 2
)

experiment.print_results (results)
\`\`\`

## Production Checklist

✅ **Role Definition**
- Specific expertise and capabilities
- Clear personality and tone
- Behavioral guidelines
- Scope boundaries

✅ **Output Control**
- Format specifications
- Structure requirements
- Content constraints
- Quality standards

✅ **Safety and Boundaries**
- What AI can/cannot do
- Content filtering rules
- Privacy considerations
- Error handling

✅ **Testing and Optimization**
- A/B test variants
- Measure effectiveness
- Track performance
- Iterate based on data

✅ **Maintenance**
- Version control
- Documentation
- Regular updates
- Performance monitoring

## Key Takeaways

1. **System prompts set the foundation** - Everything else builds on this
2. **Be specific about role and expertise** - Vague roles = inconsistent behavior
3. **Define boundaries clearly** - What AI can and cannot do
4. **Control output format** - Enforce consistency through system prompt
5. **Tone matters** - Match personality to your application
6. **Test systematically** - A/B test different system prompts
7. **Cursor\'s approach** - Context-aware, code-focused, minimal explanation
8. **Build component libraries** - Reuse tested prompt components
9. **Version and track** - Treat system prompts like critical code
10. **Iterate based on data** - Optimize using real usage metrics

## Next Steps

Now that you understand system prompts and role assignment, you're ready to explore **Few-Shot Learning & Examples** - learning how to teach AI through demonstrations for more reliable outputs.`,
};
