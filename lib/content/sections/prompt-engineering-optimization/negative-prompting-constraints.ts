/**
 * Negative Prompting & Constraints Section
 * Module 2: Prompt Engineering & Optimization
 */

export const negativepromptingconstraintsSection = {
    id: 'negative-prompting-constraints',
    title: 'Negative Prompting & Constraints',
    content: `# Negative Prompting & Constraints

Master techniques to guide LLMs by defining what NOT to do, setting boundaries, and enforcing constraints for reliable production outputs.

## Overview: The Power of Constraints

Sometimes telling an LLM what NOT to do is more effective than telling it what TO do.

\`\`\`python
# Without constraints - unpredictable
"Explain recursion"
# Might be: too long, too short, too technical, or too simple

# With constraints - predictable
"""Explain recursion.

Constraints:
- Maximum 3 sentences
- No code examples
- Assume high school level
- Do not use jargon without defining it"""
# Now output is predictable and useful
\`\`\`

## What Not To Do Instructions

\`\`\`python
def create_constrained_prompt(
    task: str,
    do_list: list,
    dont_list: list
) -> str:
    """
    Create prompt with explicit dos and don'ts.
    """
    
    do_text = "\\n".join([f"✓ {item}" for item in do_list])
    dont_text = "\\n".join([f"✗ {item}" for item in dont_list])
    
    return f"""{task}

DO:
{do_text}

DO NOT:
{dont_text}

Response:"""

# Example: Code generation with constraints
prompt = create_constrained_prompt(
    task="Write a Python function to validate email addresses",
    do_list=[
        "Use regex for validation",
        "Add docstring with examples",
        "Include type hints",
        "Handle edge cases"
    ],
    dont_list=[
        "Use external libraries (only use stdlib)",
        "Write more than 20 lines",
        "Include test code",
        "Add print statements"
    ]
)

print(prompt)
\`\`\`

## Avoiding Unwanted Behaviors

\`\`\`python
class BehaviorConstraints:
    """Common constraints for controlling LLM behavior."""
    
    CONCISE = """
Be extremely concise:
- No introductory phrases ("Sure, I can help")
- No unnecessary explanations
- Get straight to the point
- Maximum 3 sentences unless specifically needed
"""
    
    NO_APOLOGIZING = """
Do not apologize or show uncertainty:
- No "I'm sorry"
- No "I cannot be sure"
- No hedging language
- State facts directly
"""
    
    NO_HALLUCINATION = """
Prevent hallucinations:
- Only use information provided in context
- If unsure, say "I don't have enough information"
- Do not make up facts, URLs, or references
- Cite sources when available
"""
    
    NO_REPETITION = """
Avoid repetition:
- Do not repeat the question
- Do not summarize what you just said
- Each sentence should add new information
- Be direct and efficient
"""
    
    NO_OVEREXPLAINING = """
Do not over-explain:
- Assume user has basic knowledge
- Skip obvious information
- Focus on what's unique or important
- Trust the user's intelligence
"""
    
    FACTUAL_ONLY = """
Stick to facts:
- No opinions or speculation
- No subjective language
- Only verifiable information
- Quote sources when possible
"""

def apply_constraints(base_prompt: str, constraints: list) -> str:
    """Apply multiple behavior constraints."""
    
    constraints_text = "\\n\\n".join(constraints)
    
    return f"""{base_prompt}

IMPORTANT CONSTRAINTS:
{constraints_text}

Now respond:"""

# Usage
prompt = apply_constraints(
    base_prompt="Explain machine learning",
    constraints=[
        BehaviorConstraints.CONCISE,
        BehaviorConstraints.NO_HALLUCINATION,
        BehaviorConstraints.FACTUAL_ONLY
    ]
)
\`\`\`

## Content Filtering in Prompts

\`\`\`python
class ContentFilters:
    """Content filters for safe, appropriate outputs."""
    
    @staticmethod
    def family_friendly() -> str:
        return """
Content must be family-friendly:
- No profanity or offensive language
- No violent or disturbing content
- No adult themes
- Suitable for ages 13+
"""
    
    @staticmethod
    def professional() -> str:
        return """
Maintain professional tone:
- Business-appropriate language
- No slang or casual expressions
- Formal grammar and structure
- No humor unless specifically requested
"""
    
    @staticmethod
    def brand_safe() -> str:
        return """
Brand-safe content only:
- No controversial topics
- No political statements
- No religious content
- No potentially offensive material
"""
    
    @staticmethod
    def educational() -> str:
        return """
Educational content standards:
- Age-appropriate explanations
- Factually accurate
- Pedagogically sound
- Encourage critical thinking
"""

# Example: Educational content with filters
def create_safe_content_prompt(topic: str, age_group: str) -> str:
    """Create prompt with content safety filters."""
    
    return f"""Create educational content about: {topic}

Target audience: {age_group}

{ContentFilters.family_friendly()}
{ContentFilters.educational()}

Additional requirements:
- Use examples relevant to age group
- No external links
- Include 2-3 key takeaways
- Keep under 200 words

Content:"""

prompt = create_safe_content_prompt("Internet safety", "ages 10-12")
\`\`\`

## Safety Constraints

\`\`\`python
class SafetyConstraints:
    """Safety constraints for production AI systems."""
    
    PERSONAL_INFO = """
PRIVACY PROTECTION:
- Never request personal information (SSN, passwords, etc.)
- Do not store or repeat sensitive data
- Redact any personal info in examples
- Warn if user shares sensitive data
"""
    
    LEGAL_COMPLIANCE = """
LEGAL BOUNDARIES:
- Do not provide legal advice
- Do not make medical diagnoses
- Do not give financial investment advice
- Include appropriate disclaimers
"""
    
    SECURITY = """
SECURITY CONSTRAINTS:
- Do not generate malicious code
- No instructions for illegal activities
- No bypassing security measures
- Promote security best practices
"""
    
    HARMFUL_CONTENT = """
HARM PREVENTION:
- No instructions for dangerous activities
- No self-harm related content
- No hate speech or discrimination
- Report concerns appropriately
"""

def create_safe_prompt(task: str, safety_level: str = "high") -> str:
    """Create prompt with appropriate safety constraints."""
    
    constraints = []
    
    if safety_level in ["medium", "high"]:
        constraints.append(SafetyConstraints.PERSONAL_INFO)
        constraints.append(SafetyConstraints.HARMFUL_CONTENT)
    
    if safety_level == "high":
        constraints.append(SafetyConstraints.LEGAL_COMPLIANCE)
        constraints.append(SafetyConstraints.SECURITY)
    
    constraints_text = "\\n\\n".join(constraints)
    
    return f"""{task}

SAFETY REQUIREMENTS:
{constraints_text}

Proceed with task:"""
\`\`\`

## Scope Limitations

\`\`\`python
def create_scoped_prompt(
    domain: str,
    in_scope: list,
    out_of_scope: list,
    redirect_message: str = "That's outside my expertise."
) -> str:
    """
    Create prompt that keeps AI focused on specific domain.
    """
    
    in_scope_text = "\\n".join([f"• {item}" for item in in_scope])
    out_of_scope_text = "\\n".join([f"• {item}" for item in out_of_scope])
    
    return f"""You are a specialist in {domain}.

YOUR EXPERTISE (help with these):
{in_scope_text}

OUTSIDE YOUR SCOPE (politely decline):
{out_of_scope_text}

If asked about out-of-scope topics, respond:
"{redirect_message} I specialize in {domain}."

Maintain this focus strictly."""

# Example: SQL assistant
sql_assistant_prompt = create_scoped_prompt(
    domain="SQL database queries",
    in_scope=[
        "Writing SELECT, INSERT, UPDATE, DELETE queries",
        "Query optimization and indexing",
        "Schema design best practices",
        "Debugging SQL errors"
    ],
    out_of_scope=[
        "Application code (Python, JavaScript, etc.)",
        "Frontend development",
        "Server administration",
        "NoSQL databases"
    ],
    redirect_message="I focus exclusively on SQL."
)
\`\`\`

## Edge Case Handling

\`\`\`python
class EdgeCaseConstraints:
    """Constraints for handling edge cases properly."""
    
    @staticmethod
    def null_empty_handling() -> str:
        return """
Handle edge cases explicitly:
- Empty inputs: return appropriate error/empty result
- Null values: check and handle gracefully
- Invalid inputs: validate and provide clear error messages
- Boundary conditions: test and handle min/max values
"""
    
    @staticmethod
    def error_scenarios() -> str:
        return """
Error handling requirements:
- Never ignore potential errors
- Provide specific error messages
- Include error handling in all code
- Consider failure modes
- Test edge cases
"""
    
    @staticmethod
    def data_validation() -> str:
        return """
Input validation:
- Validate all inputs before processing
- Check types, ranges, formats
- Sanitize user input
- Provide validation error messages
- Use defensive programming
"""

# Example: Function generation with edge case handling
def generate_robust_function(task: str) -> str:
    """Generate function with comprehensive edge case handling."""
    
    return f"""{task}

CODE REQUIREMENTS:
- Include comprehensive error handling
- Handle all edge cases explicitly
- Add input validation
- Include docstring with edge cases
- Add type hints

{EdgeCaseConstraints.null_empty_handling()}
{EdgeCaseConstraints.error_scenarios()}
{EdgeCaseConstraints.data_validation()}

Generate code:"""

prompt = generate_robust_function(
    "Write a function to divide two numbers"
)
\`\`\`

## Format Constraints

\`\`\`python
class FormatConstraints:
    """Constraints for output format."""
    
    @staticmethod
    def length_limit(max_words: int = None, max_sentences: int = None) -> str:
        """Constrain output length."""
        
        constraints = []
        if max_words:
            constraints.append(f"- Maximum {max_words} words")
        if max_sentences:
            constraints.append(f"- Maximum {max_sentences} sentences")
        
        return "LENGTH CONSTRAINTS:\\n" + "\\n".join(constraints)
    
    @staticmethod
    def structure_requirement(structure: dict) -> str:
        """Require specific structure."""
        
        structure_text = "\\n".join([
            f"{i+1}. {section}" 
            for i, section in enumerate(structure.get('sections', []))
        ])
        
        return f"""REQUIRED STRUCTURE:
{structure_text}

Follow this structure exactly. Do not add or remove sections."""
    
    @staticmethod
    def tone_requirements(tone: str, avoid: list) -> str:
        """Specify tone and what to avoid."""
        
        avoid_text = "\\n".join([f"- {item}" for item in avoid])
        
        return f"""TONE: {tone}

AVOID:
{avoid_text}"""

# Example: Blog post with format constraints
def create_blog_prompt(topic: str) -> str:
    return f"""Write a blog post about: {topic}

{FormatConstraints.length_limit(max_words=500, max_sentences=20)}

{FormatConstraints.structure_requirement({
    'sections': [
        'Compelling headline',
        'Introduction (2-3 sentences)',
        '3 main points (each with example)',
        'Conclusion with call-to-action'
    ]
})}

{FormatConstraints.tone_requirements(
    tone="Professional but conversational",
    avoid=[
        "Clickbait language",
        "Excessive exclamation marks",
        "Marketing jargon",
        "Passive voice"
    ]
)}

Write post:"""
\`\`\`

## Building Guardrails

\`\`\`python
from typing import List, Callable

class PromptGuardrails:
    """
    Comprehensive guardrail system for prompts.
    Combines multiple constraint types.
    """
    
    def __init__(self):
        self.constraints = []
    
    def add_content_filter(self, filter_type: str):
        """Add content filtering constraint."""
        if filter_type == "family_friendly":
            self.constraints.append(ContentFilters.family_friendly())
        elif filter_type == "professional":
            self.constraints.append(ContentFilters.professional())
    
    def add_safety_constraint(self, constraint_type: str):
        """Add safety constraint."""
        if constraint_type == "privacy":
            self.constraints.append(SafetyConstraints.PERSONAL_INFO)
        elif constraint_type == "security":
            self.constraints.append(SafetyConstraints.SECURITY)
    
    def add_behavior_constraint(self, behavior: str):
        """Add behavior constraint."""
        if behavior == "concise":
            self.constraints.append(BehaviorConstraints.CONCISE)
        elif behavior == "no_hallucination":
            self.constraints.append(BehaviorConstraints.NO_HALLUCINATION)
    
    def add_custom_constraint(self, constraint: str):
        """Add custom constraint."""
        self.constraints.append(constraint)
    
    def build_prompt(self, base_task: str) -> str:
        """Build final prompt with all guardrails."""
        
        if not self.constraints:
            return base_task
        
        guardrails_text = "\\n\\n".join([
            f"CONSTRAINT {i+1}:\\n{constraint}"
            for i, constraint in enumerate(self.constraints)
        ])
        
        return f"""{base_task}

{'='*60}
GUARDRAILS (MUST FOLLOW):
{'='*60}

{guardrails_text}

{'='*60}

Now complete the task while respecting ALL constraints:"""

# Example: Production prompt with comprehensive guardrails
guardrails = PromptGuardrails()

# Add various constraints
guardrails.add_content_filter("professional")
guardrails.add_safety_constraint("privacy")
guardrails.add_behavior_constraint("concise")
guardrails.add_behavior_constraint("no_hallucination")
guardrails.add_custom_constraint("""
CUSTOM REQUIREMENT:
- Cite sources for all factual claims
- Use bullet points for lists
- Include confidence level for uncertain information
""")

# Build final prompt
task = "Summarize the latest AI developments"
final_prompt = guardrails.build_prompt(task)

print(final_prompt)
\`\`\`

## Production Checklist

✅ **Behavior Constraints**
- Define what NOT to do
- Set tone and style boundaries
- Prevent unwanted patterns
- Enforce conciseness
- Avoid hallucinations

✅ **Content Filtering**
- Age-appropriate content
- Brand safety
- Professional standards
- Legal compliance
- Cultural sensitivity

✅ **Safety Guardrails**
- Privacy protection
- Security boundaries
- Harmful content prevention
- Legal disclaimer requirements
- Ethical guidelines

✅ **Scope Management**
- Define expertise boundaries
- Handle out-of-scope requests
- Redirect appropriately
- Maintain focus
- Clear domain limits

✅ **Format Enforcement**
- Length constraints
- Structure requirements
- Tone specifications
- Output format rules
- Validation criteria

## Key Takeaways

1. **Constraints improve reliability** - Tell LLM what NOT to do
2. **Negative prompts prevent issues** - Proactive problem prevention
3. **Safety constraints are essential** - Protect users and system
4. **Scope limitations help** - Keep AI focused on expertise
5. **Content filters ensure appropriateness** - Brand and age safety
6. **Behavior constraints control output** - Conciseness, tone, style
7. **Edge cases need explicit handling** - Don't assume LLM knows
8. **Format constraints ensure parseability** - Structured, predictable outputs
9. **Layer multiple constraints** - Comprehensive guardrails
10. **Test constraint effectiveness** - Verify they work in practice

## Next Steps

Now that you understand negative prompting and constraints, you're ready to explore **Prompt Injection & Security** - learning how to protect your prompts from malicious manipulation and ensure secure AI applications.`,
};

