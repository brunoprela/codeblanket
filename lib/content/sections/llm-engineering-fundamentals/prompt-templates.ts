/**
 * Prompt Templates & Variables Section
 * Module 1: LLM Engineering Fundamentals
 */

export const prompttemplatesSection = {
  id: 'prompt-templates',
  title: 'Prompt Templates & Variables',
  content: `# Prompt Templates & Variables

Master template systems to build maintainable, reusable prompts for production LLM applications.

## Why Prompt Templates?

Hard-coding prompts leads to unmaintainable code. Templates solve this.

### The Problem with Hard-Coded Prompts

\`\`\`python
# ❌ BAD: Hard-coded prompts scattered throughout code

def translate_text(text: str):
    prompt = f"Translate this to Spanish: {text}"
    return call_llm(prompt)

def summarize_text(text: str):
    prompt = f"Summarize this: {text}"
    return call_llm(prompt)

def extract_email(text: str):
    prompt = f"Extract the email address from: {text}"
    return call_llm(prompt)

"""
Problems:
1. Hard to update prompts
2. No consistency
3. Can't A/B test
4. No version control
5. Duplication of logic
"""

# ✅ GOOD: Template-based approach

TEMPLATES = {
    'translate': "Translate the following text to {language}:\\n\\n{text}",
    'summarize': "Summarize the following text in {num_sentences} sentences:\\n\\n{text}",
    'extract_email': "Extract the email address from this text. Return only the email.\\n\\n{text}"
}

def translate_text(text: str, language: str = "Spanish"):
    prompt = TEMPLATES['translate'].format(language=language, text=text)
    return call_llm(prompt)

# Now can update all prompts in one place!
\`\`\`

## Basic String Templates

Start with Python's built-in string formatting.

### String Format Method

\`\`\`python
# Basic template with format()
template = """You are a {role} expert.

Task: {task}

Input:
{input}

Provide a detailed response."""

# Use it
prompt = template.format(
    role="Python",
    task="Explain this code",
    input="def fib(n): return n if n < 2 else fib(n-1) + fib(n-2)"
)

print(prompt)
\`\`\`

### f-Strings (Python 3.6+)

\`\`\`python
# f-strings - simpler syntax
role = "data scientist"
task = "analyze this dataset"
data = "..."

prompt = f"""You are a {role}.

Task: {task}

Data:
{data}

Provide insights."""

print(prompt)
\`\`\`

### Template Variables

\`\`\`python
from string import Template

# Template with $-based variables
template = Template("""
System: You are a $role expert.

User: $query

Respond professionally.
""")

# Substitute variables
prompt = template.substitute(
    role="software engineer",
    query="How do I optimize database queries?"
)

print(prompt)

# Safe substitute - won't error on missing variables
prompt = template.safe_substitute(role="engineer")
# $query remains as-is if not provided
\`\`\`

## Advanced Template Systems

Build a reusable template manager.

### Template Manager Class

\`\`\`python
from typing import Dict, Any
from pathlib import Path

class PromptTemplateManager:
    """
    Manage prompt templates centrally.
    """
    
    def __init__(self):
        self.templates: Dict[str, str] = {}
    
    def add_template(self, name: str, template: str):
        """Add a template."""
        self.templates[name] = template
    
    def get_template(self, name: str) -> str:
        """Get a template by name."""
        if name not in self.templates:
            raise ValueError(f"Template '{name}' not found")
        return self.templates[name]
    
    def render(self, name: str, **kwargs) -> str:
        """
        Render a template with variables.
        """
        template = self.get_template(name)
        
        try:
            return template.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"Missing variable: {e}")
    
    def list_templates(self) -> list:
        """List all template names."""
        return list(self.templates.keys())
    
    def load_from_file(self, filepath: str):
        """Load template from file."""
        path = Path(filepath)
        template_name = path.stem
        template_content = path.read_text()
        self.add_template(template_name, template_content)

# Usage
manager = PromptTemplateManager()

# Add templates
manager.add_template(
    'code_review',
    """You are an expert code reviewer.

Review this code for:
1. Bugs and errors
2. Performance issues
3. Best practices

Code:
\`\`\`{ language }
{ code }
\`\`\`

Provide specific, actionable feedback."""
)

manager.add_template(
    'summarize',
    """Summarize the following text in {num_sentences} sentences.

Text:
{text}

Summary:"""
)

# Use templates
prompt1 = manager.render(
    'code_review',
    language='python',
    code='def add(a, b): return a + b'
)

prompt2 = manager.render(
    'summarize',
    num_sentences=3,
    text='Long article text here...'
)

print("Templates available:", manager.list_templates())
\`\`\`

## Jinja2 Templates

For complex templates, use Jinja2 - the standard template engine.

### Basic Jinja2 Usage

\`\`\`python
# pip install jinja2

from jinja2 import Template

# Simple template
template = Template("""
You are a {{ role }} assistant.

{% if context %}
Context: {{ context }}
{% endif %}

User Query: {{ query }}

Respond helpfully.
""")

# Render
prompt = template.render(
    role="Python",
    context="The user is learning programming",
    query="What are variables?"
)

print(prompt)
\`\`\`

### Jinja2 with Conditionals and Loops

\`\`\`python
from jinja2 import Template

template = Template("""
You are a {{ role }} expert.

{% if examples %}
Here are some examples:
{% for example in examples %}
{{ loop.index }}. {{ example }}
{% endfor %}
{% endif %}

Now help with: {{ task }}

{% if constraints %}
Constraints:
{% for constraint in constraints %}
- {{ constraint }}
{% endfor %}
{% endif %}
""")

prompt = template.render(
    role="Python",
    examples=["Example 1", "Example 2", "Example 3"],
    task="Write a function to sort a list",
    constraints=["Use built-in functions only", "Add type hints"]
)

print(prompt)
\`\`\`

### Jinja2 Template Manager

\`\`\`python
from jinja2 import Environment, FileSystemLoader, Template
from pathlib import Path

class Jinja2TemplateManager:
    """
    Template manager using Jinja2.
    """
    
    def __init__(self, templates_dir: str = None):
        if templates_dir:
            # Load from directory
            self.env = Environment(
                loader=FileSystemLoader(templates_dir),
                trim_blocks=True,
                lstrip_blocks=True
            )
        else:
            # No directory - templates added manually
            self.env = Environment()
            self.templates = {}
    
    def add_template(self, name: str, template_string: str):
        """Add template from string."""
        self.templates[name] = Template(template_string)
    
    def render(self, name: str, **kwargs) -> str:
        """Render template with variables."""
        if hasattr(self, 'env') and self.env.loader:
            # Load from file
            template = self.env.get_template(f"{name}.txt")
        else:
            # Load from memory
            template = self.templates.get(name)
            if not template:
                raise ValueError(f"Template '{name}' not found")
        
        return template.render(**kwargs)

# Usage
manager = Jinja2TemplateManager()

# Add template
manager.add_template('qa', """
You are a question-answering system.

{% if context %}
Context: {{ context }}
{% endif %}

Question: {{ question }}

{% if format == "detailed" %}
Provide a detailed answer with examples.
{% else %}
Provide a concise answer.
{% endif %}
""")

# Render different versions
detailed = manager.render(
    'qa',
    question="What is Python?",
    context="Programming languages",
    format="detailed"
)

concise = manager.render(
    'qa',
    question="What is Python?",
    format="concise"
)

print("DETAILED:")
print(detailed)
print("\\nCONCISE:")
print(concise)
\`\`\`

## Langchain Prompt Templates

LangChain provides robust prompt management.

### LangChain PromptTemplate

\`\`\`python
# pip install langchain

from langchain.prompts import PromptTemplate

# Simple template
template = PromptTemplate(
    input_variables=["topic", "detail_level"],
    template="""
Explain {topic} at a {detail_level} level.

Provide clear, accurate information.
"""
)

# Use it
prompt = template.format(topic="neural networks", detail_level="beginner")
print(prompt)

# Or with format_prompt (returns PromptValue)
prompt_value = template.format_prompt(topic="APIs", detail_level="intermediate")
print(prompt_value.to_string())
\`\`\`

### LangChain ChatPromptTemplate

\`\`\`python
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

# Multi-message template
system_template = "You are a {role} expert who {behavior}."
system_message = SystemMessagePromptTemplate.from_template(system_template)

human_template = "{task}\\n\\nInput: {input}"
human_message = HumanMessagePromptTemplate.from_template(human_template)

# Combine
chat_prompt = ChatPromptTemplate.from_messages([
    system_message,
    human_message
])

# Format
messages = chat_prompt.format_prompt(
    role="Python",
    behavior="explains concepts clearly with examples",
    task="Explain list comprehensions",
    input="I want to filter a list"
).to_messages()

# Messages are ready for OpenAI API
for msg in messages:
    print(f"{msg.type}: {msg.content}")
\`\`\`

### LangChain FewShotPromptTemplate

\`\`\`python
from langchain.prompts import FewShotPromptTemplate, PromptTemplate

# Example formatter
example_template = """
Input: {input}
Output: {output}
"""

example_prompt = PromptTemplate(
    input_variables=["input", "output"],
    template=example_template
)

# Examples
examples = [
    {
        "input": "Happy",
        "output": "Positive"
    },
    {
        "input": "Angry",
        "output": "Negative"
    },
    {
        "input": "Okay",
        "output": "Neutral"
    }
]

# Few-shot template
few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="Classify the sentiment of each word:",
    suffix="Input: {input}\\nOutput:",
    input_variables=["input"]
)

# Use it
prompt = few_shot_prompt.format(input="Excited")
print(prompt)
\`\`\`

## Production Template System

Build a complete template system for production.

### Complete Template System

\`\`\`python
from typing import Dict, List, Optional, Any
from pathlib import Path
import json
import yaml
from jinja2 import Environment, Template

class ProductionTemplateSystem:
    """
    Production-ready template system.
    
    Features:
    - Load templates from files or strings
    - Variable validation
    - Template versioning
    - Template inheritance
    - Default values
    """
    
    def __init__(self, templates_dir: Optional[str] = None):
        self.templates: Dict[str, Dict] = {}
        self.templates_dir = Path(templates_dir) if templates_dir else None
        self.env = Environment()
    
    def register_template(
        self,
        name: str,
        template_string: str,
        required_vars: List[str] = None,
        default_vars: Dict[str, Any] = None,
        version: str = "1.0",
        description: str = ""
    ):
        """
        Register a template with metadata.
        """
        self.templates[name] = {
            'template': Template(template_string),
            'required_vars': required_vars or [],
            'default_vars': default_vars or {},
            'version': version,
            'description': description
        }
    
    def load_from_yaml(self, filepath: str):
        """
        Load templates from YAML file.
        
        Format:
        templates:
          template_name:
            template: "..."
            required_vars: [...]
            default_vars: {...}
        """
        with open(filepath) as f:
            data = yaml.safe_load(f)
        
        for name, config in data['templates'].items():
            self.register_template(
                name=name,
                template_string=config['template'],
                required_vars=config.get('required_vars', []),
                default_vars=config.get('default_vars', {}),
                version=config.get('version', '1.0'),
                description=config.get('description', '')
            )
    
    def render(
        self,
        name: str,
        validate: bool = True,
        **kwargs
    ) -> str:
        """
        Render template with variable validation.
        """
        if name not in self.templates:
            raise ValueError(f"Template '{name}' not found")
        
        template_data = self.templates[name]
        template = template_data['template']
        required_vars = template_data['required_vars']
        default_vars = template_data['default_vars']
        
        # Merge defaults with provided vars
        variables = {**default_vars, **kwargs}
        
        # Validate required variables
        if validate:
            missing = [var for var in required_vars if var not in variables]
            if missing:
                raise ValueError(f"Missing required variables: {missing}")
        
        return template.render(**variables)
    
    def get_template_info(self, name: str) -> Dict:
        """Get metadata about a template."""
        if name not in self.templates:
            raise ValueError(f"Template '{name}' not found")
        
        template_data = self.templates[name]
        return {
            'name': name,
            'version': template_data['version'],
            'description': template_data['description'],
            'required_vars': template_data['required_vars'],
            'default_vars': template_data['default_vars']
        }
    
    def list_templates(self) -> List[Dict]:
        """List all templates with metadata."""
        return [
            self.get_template_info(name)
            for name in self.templates.keys()
        ]

# Usage
system = ProductionTemplateSystem()

# Register templates
system.register_template(
    name='code_review',
    template_string="""You are an expert {{ language }} code reviewer.

Review the following code:
\`\`\`{ { language } }
{ { code } }
\`\`\`

{% if focus_areas %}
Focus on:
{% for area in focus_areas %}
- {{ area }}
{% endfor %}
{% endif %}

Provide specific, actionable feedback.""",
    required_vars=['language', 'code'],
    default_vars={'focus_areas': ['bugs', 'performance', 'style']},
    version='1.0',
    description='Code review template'
)

system.register_template(
    name='summarize',
    template_string="""Summarize the following text in {{ num_sentences }} sentences:

{{ text }}

{% if style %}
Style: {{ style }}
{% endif %}""",
    required_vars=['text'],
    default_vars={'num_sentences': 3, 'style': 'concise'},
    version='1.1',
    description='Text summarization template'
)

# List templates
print("Available templates:")
for tmpl in system.list_templates():
    print(f"- {tmpl['name']} (v{tmpl['version']}): {tmpl['description']}")

# Render with validation
try:
    prompt = system.render(
        'code_review',
        language='python',
        code='def add(a, b): return a + b',
        focus_areas=['performance', 'testing']
    )
    print("\\n" + prompt)
except ValueError as e:
    print(f"Error: {e}")
\`\`\`

## Template Best Practices

\`\`\`python
"""
TEMPLATE BEST PRACTICES:

1. CENTRALIZE TEMPLATES
   ✅ Store in one place
   ✅ Easy to update
   ✅ Version control friendly

2. USE VARIABLES
   ✅ Don't hard-code values
   ✅ Make reusable
   ✅ Allow customization

3. VALIDATE INPUTS
   ✅ Check required variables
   ✅ Validate types
   ✅ Provide clear errors

4. SET DEFAULTS
   ✅ Common values
   ✅ Reduce boilerplate
   ✅ Make usage easier

5. VERSION TEMPLATES
   ✅ Track changes
   ✅ A/B test versions
   ✅ Rollback if needed

6. DOCUMENT TEMPLATES
   ✅ Describe purpose
   ✅ List variables
   ✅ Show examples

7. USE INHERITANCE
   ✅ Base templates
   ✅ Override sections
   ✅ Reduce duplication

8. SEPARATE CONCERNS
   ✅ System prompts
   ✅ User prompts
   ✅ Output format

Example Template File Structure:

templates/
├── system/
│   ├── code_assistant.txt
│   ├── data_analyst.txt
│   └── writer.txt
├── tasks/
│   ├── summarize.txt
│   ├── translate.txt
│   └── extract.txt
└── outputs/
    ├── json.txt
    ├── markdown.txt
    └── csv.txt
"""
\`\`\`

## Template Composition

Combine templates for complex prompts.

\`\`\`python
class ComposableTemplateSystem:
    """
    System that supports template composition.
    """
    
    def __init__(self):
        self.templates = {}
    
    def register(self, name: str, template: str):
        """Register a template."""
        self.templates[name] = Template(template)
    
    def compose(self, *template_names, separator: str = "\\n\\n", **kwargs) -> str:
        """
        Compose multiple templates together.
        """
        parts = []
        
        for name in template_names:
            if name not in self.templates:
                raise ValueError(f"Template '{name}' not found")
            
            template = self.templates[name]
            rendered = template.render(**kwargs)
            parts.append(rendered)
        
        return separator.join(parts)

# Usage
system = ComposableTemplateSystem()

# Register components
system.register('system_role', "You are a {{ role }} expert.")

system.register('task_description', """
Task: {{ task }}

{% if requirements %}
Requirements:
{% for req in requirements %}
- {{ req }}
{% endfor %}
{% endif %}
""")

system.register('output_format', """
Output format: {{ format }}
""")

# Compose them
prompt = system.compose(
    'system_role',
    'task_description',
    'output_format',
    separator='\\n\\n',
    role='Python',
    task='Write a sorting function',
    requirements=['Add type hints', 'Handle edge cases'],
    format='Python code with comments'
)

print(prompt)
\`\`\`

## Key Takeaways

1. **Don't hard-code prompts** - use templates
2. **Start with string format** - f-strings or .format()
3. **Use Jinja2** for complex templates
4. **Validate variables** before rendering
5. **Set sensible defaults** to reduce boilerplate
6. **Version your templates** for A/B testing
7. **Centralize templates** in one place
8. **Use LangChain** if already in your stack
9. **Compose templates** for complex prompts
10. **Document your templates** with examples

## Next Steps

Now you have reusable templates. Next: **Output Parsing & Structured Data** - learning to reliably extract structured data from LLM outputs.`,
};
