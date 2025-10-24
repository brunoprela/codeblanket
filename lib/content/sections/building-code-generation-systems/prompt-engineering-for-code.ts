/**
 * Prompt Engineering for Code Section
 * Module 5: Building Code Generation Systems
 */

export const promptengineeringforcodeSection = {
  id: 'prompt-engineering-for-code',
  title: 'Prompt Engineering for Code',
  content: `# Prompt Engineering for Code

Master the art of crafting prompts that generate high-quality, contextual code like Cursor does.

## Overview: Code Prompts are Different

Prompting for code generation requires different techniques than general prompting:

- **Context is king**: More relevant context = better code
- **Specificity matters**: Vague prompts = generic code
- **Examples help**: Show the style you want
- **Constraints guide**: Tell it what NOT to do
- **Structure matters**: How you format the prompt affects output

### How Cursor Constructs Prompts

When you use Cursor to edit code, it doesn't just send your request. It builds a sophisticated prompt with:

1. **File context**: Current file content
2. **Related files**: Imported modules, base classes
3. **Project structure**: File tree
4. **Language context**: Framework conventions
5. **Your request**: Natural language instruction
6. **Output format**: How to structure the response

## Essential Context Types

### 1. File Tree Context

Showing the project structure helps LLMs understand organization:

\`\`\`python
def get_file_tree(root_path: str, max_depth: int = 3) -> str:
    """Generate a file tree representation for LLM context."""
    import os
    from pathlib import Path
    
    tree_lines = []
    
    def walk_dir(path: Path, prefix: str = "", depth: int = 0):
        if depth > max_depth:
            return
        
        try:
            entries = sorted(path.iterdir(), key=lambda x: (not x.is_dir(), x.name))
        except PermissionError:
            return
        
        for i, entry in enumerate(entries):
            is_last = i == len(entries) - 1
            current_prefix = "└── " if is_last else "├── "
            
            # Skip common ignore patterns
            if entry.name in {'.git', '__pycache__', 'node_modules', '.venv'}:
                continue
            
            tree_lines.append(f"{prefix}{current_prefix}{entry.name}")
            
            if entry.is_dir():
                extension = "    " if is_last else "│   "
                walk_dir(entry, prefix + extension, depth + 1)
    
    tree_lines.append(root_path)
    walk_dir(Path(root_path))
    
    return "\\n".join(tree_lines)

# Usage in prompt
project_tree = get_file_tree("/path/to/project")
prompt = f"""You are editing code in this project:

Project Structure:
{project_tree}

Task: Add a new API endpoint for user authentication
"""
\`\`\`

### 2. Current File Context

Provide the file being edited:

\`\`\`python
def format_file_context(
    file_path: str,
    content: str,
    cursor_line: Optional[int] = None
) -> str:
    """Format file content with context markers."""
    lines = content.split("\\n")
    
    # Add line numbers
    numbered_lines = []
    for i, line in enumerate(lines, 1):
        marker = " <-- CURSOR HERE" if i == cursor_line else ""
        numbered_lines.append(f"{i:4d} | {line}{marker}")
    
    return f"""File: {file_path}
{'='*60}
{"".join(numbered_lines)}
{'='*60}
"""

# Usage
file_context = format_file_context(
    "app/routes.py",
    current_file_content,
    cursor_line=45
)

prompt = f"""{file_context}

Task: Add error handling to the function at the cursor
"""
\`\`\`

### 3. Related Files Context

Include imported files and dependencies:

\`\`\`python
import ast
from pathlib import Path
from typing import List, Set

def extract_imports(code: str) -> Set[str]:
    """Extract all import statements from Python code."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return set()
    
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for name in node.names:
                imports.add(name.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module)
    
    return imports

def find_related_files(
    current_file: str,
    project_root: str,
    max_files: int = 3
) -> List[tuple[str, str]]:
    """Find and return content of related files."""
    with open(current_file) as f:
        code = f.read()
    
    imports = extract_imports(code)
    related_files = []
    
    for imp in imports:
        # Convert import to file path
        # e.g., "myapp.utils" -> "myapp/utils.py"
        file_path = Path(project_root) / imp.replace(".", "/")
        
        for suffix in [".py", "/__init__.py"]:
            full_path = str(file_path) + suffix
            if Path(full_path).exists():
                with open(full_path) as f:
                    content = f.read()
                related_files.append((full_path, content))
                break
        
        if len(related_files) >= max_files:
            break
    
    return related_files

# Usage
related = find_related_files("app/routes.py", "/project/root")

related_context = ""
for path, content in related:
    related_context += f"""
Related File: {path}
{'-'*60}
{content[:500]}  # First 500 chars
{'-'*60}

"""

prompt = f"""Current file context:
{current_file_context}

{related_context}

Task: Refactor to use the utilities from the related files
"""
\`\`\`

### 4. Function Signature Context

For editing functions, provide signatures of related functions:

\`\`\`python
def extract_function_signatures(code: str) -> List[str]:
    """Extract function signatures from code."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    
    signatures = []
    
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            # Get arguments
            args = []
            for arg in node.args.args:
                arg_str = arg.arg
                if arg.annotation:
                    arg_str += f": {ast.unparse(arg.annotation)}"
                args.append(arg_str)
            
            # Get return type
            return_type = ""
            if node.returns:
                return_type = f" -> {ast.unparse(node.returns)}"
            
            signature = f"def {node.name}({', '.join(args)}){return_type}"
            
            # Add docstring if present
            docstring = ast.get_docstring(node)
            if docstring:
                signature += f"\\n    ''{docstring}''"
            
            signatures.append(signature)
    
    return signatures

# Usage
signatures = extract_function_signatures(current_file_content)
sig_context = "\\n\\n".join(signatures)

prompt = f"""Available functions in this file:
{sig_context}

Task: Write a new function that uses these existing functions
"""
\`\`\`

### 5. Type Definitions Context

For typed languages, provide type information:

\`\`\`python
def extract_type_definitions(code: str) -> str:
    """Extract class and type definitions."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return ""
    
    definitions = []
    
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            # Class definition
            class_def = f"class {node.name}"
            
            # Base classes
            if node.bases:
                bases = [ast.unparse(b) for b in node.bases]
                class_def += f"({', '.join(bases)})"
            
            class_def += ":"
            
            # Get docstring
            docstring = ast.get_docstring(node)
            if docstring:
                class_def += f"\\n    ''{docstring}''"
            
            # Get methods (just signatures)
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    args = [arg.arg for arg in item.args.args]
                    class_def += f"\\n    def {item.name}({', '.join(args)}): ..."
            
            definitions.append(class_def)
    
    return "\\n\\n".join(definitions)

# Usage
types = extract_type_definitions(current_file_content)

prompt = f"""Type definitions in this file:
{types}

Task: Add a new method to the User class that validates email
"""
\`\`\`

## Effective Prompt Patterns

### Pattern 1: The Context Sandwich

Structure: Context → Request → Constraints

\`\`\`python
def create_context_sandwich_prompt(
    context: str,
    request: str,
    constraints: List[str]
) -> str:
    """Create a well-structured prompt."""
    constraints_str = "\\n".join(f"- {c}" for c in constraints)
    
    return f"""# Context
{context}

# Task
{request}

# Constraints
{constraints_str}

# Output
Provide only the code, no explanations.
"""

# Usage
prompt = create_context_sandwich_prompt(
    context=file_context,
    request="Add input validation to the login function",
    constraints=[
        "Use pydantic for validation",
        "Return clear error messages",
        "Don't modify function signature",
        "Add type hints"
    ]
)
\`\`\`

### Pattern 2: Example-Driven Generation

Show examples of the style you want:

\`\`\`python
def create_example_driven_prompt(
    task: str,
    examples: List[tuple[str, str]]  # (before, after) pairs
) -> str:
    """Create prompt with before/after examples."""
    examples_str = ""
    for i, (before, after) in enumerate(examples, 1):
        examples_str += f"""
Example {i}:

BEFORE:
{before}

AFTER:
{after}

"""
    
    return f"""I want you to transform code following these examples:

{examples_str}

Now apply the same transformation to:

BEFORE:
{task}

AFTER:
"""

# Usage
examples = [
    (
        "def get_user(id):",
        "def get_user(user_id: int) -> User:"
    ),
    (
        "def save(data):",
        "def save(data: dict) -> bool:"
    )
]

prompt = create_example_driven_prompt(
    "def process(items):",
    examples
)
\`\`\`

### Pattern 3: Incremental Refinement

Build up context incrementally:

\`\`\`python
class PromptBuilder:
    """Build complex prompts incrementally."""
    
    def __init__(self):
        self.sections = {}
    
    def add_context(self, name: str, content: str):
        """Add a context section."""
        self.sections[f"context_{name}"] = content
        return self
    
    def add_examples(self, examples: List[str]):
        """Add code examples."""
        self.sections['examples'] = "\\n\\n".join(examples)
        return self
    
    def add_constraints(self, constraints: List[str]):
        """Add constraints."""
        self.sections['constraints'] = "\\n".join(f"- {c}" for c in constraints)
        return self
    
    def set_task(self, task: str):
        """Set the main task."""
        self.sections['task'] = task
        return self
    
    def build(self) -> str:
        """Build the final prompt."""
        sections_order = [
            'context_file',
            'context_types',
            'context_related',
            'examples',
            'task',
            'constraints'
        ]
        
        parts = []
        for key in sections_order:
            if key in self.sections:
                title = key.replace('_', ' ').title()
                parts.append(f"# {title}")
                parts.append(self.sections[key])
                parts.append("")
        
        return "\\n".join(parts)

# Usage
prompt = (PromptBuilder()
    .add_context('file', current_file_content)
    .add_context('types', type_definitions)
    .add_examples([example1, example2])
    .set_task("Add async/await to all database operations")
    .add_constraints([
        "Maintain backward compatibility",
        "Add proper error handling",
        "Update type hints"
    ])
    .build()
)
\`\`\`

## Token Optimization Techniques

Prompts can get large. Optimize token usage:

### 1. Intelligent Truncation

\`\`\`python
def smart_truncate_code(
    code: str,
    max_lines: int = 100,
    focus_line: Optional[int] = None
) -> str:
    """Truncate code while keeping relevant context."""
    lines = code.split("\\n")
    
    if len(lines) <= max_lines:
        return code
    
    if focus_line is None:
        # Just take first max_lines
        return "\\n".join(lines[:max_lines]) + "\\n... (truncated)"
    
    # Keep context around focus line
    context_size = max_lines // 2
    start = max(0, focus_line - context_size)
    end = min(len(lines), focus_line + context_size)
    
    truncated_lines = []
    
    if start > 0:
        truncated_lines.append("... (lines 1-{start} omitted)")
    
    truncated_lines.extend(lines[start:end])
    
    if end < len(lines):
        truncated_lines.append(f"... (lines {end+1}-{len(lines)} omitted)")
    
    return "\\n".join(truncated_lines)

# Usage
optimized = smart_truncate_code(
    large_file_content,
    max_lines=50,
    focus_line=125  # Where the edit will happen
)
\`\`\`

### 2. Summarize Irrelevant Parts

\`\`\`python
def summarize_functions(code: str) -> str:
    """Replace function bodies with summaries."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    
    # Extract function info
    functions = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            docstring = ast.get_docstring(node) or "No description"
            
            # Get signature
            args = [arg.arg for arg in node.args.args]
            sig = f"def {node.name}({', '.join(args)})"
            
            functions.append(f"{sig}:\\n    ''{docstring}''\\n    ...")
    
    return "\\n\\n".join(functions)

# Usage
# Instead of including full function bodies (expensive)
summarized = summarize_functions(utils_file)
prompt = f"""Available utility functions:
{summarized}

Task: Use these utilities to implement the feature
"""
\`\`\`

### 3. Selective Context Loading

\`\`\`python
from typing import Dict
import tiktoken

class ContextManager:
    """Manage context within token budget."""
    
    def __init__(self, max_tokens: int = 4000):
        self.max_tokens = max_tokens
        self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.encoding.encode(text))
    
    def build_optimized_context(
        self,
        required: Dict[str, str],
        optional: Dict[str, str]
    ) -> str:
        """Build context within token budget."""
        context_parts = []
        tokens_used = 0
        
        # Add required context first
        for name, content in required.items():
            tokens = self.count_tokens(content)
            if tokens_used + tokens > self.max_tokens:
                # Truncate this part
                target_tokens = self.max_tokens - tokens_used - 100
                content = self._truncate_to_tokens(content, target_tokens)
                tokens = self.count_tokens(content)
            
            context_parts.append(f"# {name}\\n{content}")
            tokens_used += tokens
        
        # Add optional context if room
        for name, content in optional.items():
            tokens = self.count_tokens(content)
            if tokens_used + tokens <= self.max_tokens:
                context_parts.append(f"# {name}\\n{content}")
                tokens_used += tokens
            else:
                break  # No more room
        
        return "\\n\\n".join(context_parts)
    
    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Truncate text to approximate token count."""
        tokens = self.encoding.encode(text)
        if len(tokens) <= max_tokens:
            return text
        
        truncated = self.encoding.decode(tokens[:max_tokens])
        return truncated + "\\n... (truncated)"

# Usage
manager = ContextManager(max_tokens=4000)

context = manager.build_optimized_context(
    required={
        "Current File": current_file_content,
        "Task": task_description
    },
    optional={
        "Related File 1": related_file_1,
        "Related File 2": related_file_2,
        "Project Structure": file_tree
    }
)
\`\`\`

## Output Format Control

Tell the LLM exactly how to format its output:

### Format Specification

\`\`\`python
def create_formatted_prompt(
    context: str,
    task: str,
    output_format: str = "code_only"
) -> str:
    """Create prompt with explicit output format."""
    
    formats = {
        "code_only": """
Output ONLY the code, no explanations or markdown.
Start directly with the code.
""",
        
        "markdown": """
Output the code in a markdown code block:
\`\`\`python
# your code here
\`\`\`
""",
        
        "diff": """
Output the changes as a unified diff:
\`\`\`diff
- old line
+ new line
    \`\`\`
""",
        
        "json": """
Output as JSON with this structure:
{
    "code": "the generated code",
    "explanation": "brief explanation",
    "changes": ["list of changes made"]
}
"""
    }
    
    format_instruction = formats.get(output_format, formats["code_only"])
    
    return f"""{context}

Task: {task}

{format_instruction}
"""

# Usage for different scenarios
# 1. Simple generation - just want code
prompt = create_formatted_prompt(context, task, "code_only")

# 2. Need diff for editing
prompt = create_formatted_prompt(context, task, "diff")

# 3. Need structured response
prompt = create_formatted_prompt(context, task, "json")
\`\`\`

## Language-Specific Prompting

Different languages need different approaches:

### Python-Specific

\`\`\`python
PYTHON_STYLE_GUIDE = """
Follow these Python conventions:
- Use snake_case for functions and variables
- Use PascalCase for classes
- Add type hints to all functions
- Include docstrings for public functions
- Follow PEP 8 style guide
- Use f-strings for formatting
- Prefer list comprehensions for simple iterations
"""

def create_python_prompt(task: str, context: str) -> str:
    return f"""{PYTHON_STYLE_GUIDE}

{context}

Task: {task}
"""
\`\`\`

### JavaScript/TypeScript-Specific

\`\`\`python
TYPESCRIPT_STYLE_GUIDE = """
Follow these TypeScript conventions:
- Use camelCase for functions and variables
- Use PascalCase for classes and interfaces
- Add explicit types to all parameters and returns
- Use async/await over promises
- Use const for non-reassigned variables
- Prefer arrow functions
- Use template literals for strings
"""
\`\`\`

## How Cursor Does It

Cursor's prompt construction (reverse-engineered):

\`\`\`python
class CursorStylePromptBuilder:
    """Replicate Cursor's prompt construction approach."""
    
    def build_cursor_prompt(
        self,
        current_file: str,
        current_file_content: str,
        cursor_position: int,
        user_instruction: str,
        project_root: str
    ) -> str:
        """Build a Cursor-style prompt."""
        
        # 1. Project context
        file_tree = self._get_file_tree(project_root)
        
        # 2. Current file with line numbers
        file_context = self._format_file_with_cursor(
            current_file_content,
            cursor_position
        )
        
        # 3. Related files (imported modules)
        related_files = self._get_related_files(
            current_file,
            current_file_content,
            project_root
        )
        
        # 4. Language-specific context
        language = self._detect_language(current_file)
        language_context = self._get_language_context(language)
        
        # 5. Build the prompt
        prompt = f"""You are an expert programmer. You are editing this file:

PROJECT STRUCTURE:
{file_tree}

CURRENT FILE: {current_file}
{file_context}

{related_files}

{language_context}

USER REQUEST:
{user_instruction}

INSTRUCTIONS:
1. Make only the necessary changes
2. Preserve existing code style
3. Don't break existing functionality
4. Output only the modified code sections

OUTPUT FORMAT:
Provide the changes as:
<<<<<<< SEARCH
[exact lines to replace]
=======
[new lines]
>>>>>>> REPLACE
"""
        return prompt
    
    def _format_file_with_cursor(
        self,
        content: str,
        cursor_pos: int
    ) -> str:
        """Format file with cursor indicator."""
        lines = content.split("\\n")
        formatted = []
        
        for i, line in enumerate(lines):
            indicator = " <-- CURSOR" if i == cursor_pos else ""
            formatted.append(f"{i+1:4d} | {line}{indicator}")
        
        return "\\n".join(formatted)
    
    # ... other helper methods

# Usage
builder = CursorStylePromptBuilder()
prompt = builder.build_cursor_prompt(
    current_file="app/routes.py",
    current_file_content=file_content,
    cursor_position=42,
    user_instruction="Add error handling",
    project_root="/path/to/project"
)
\`\`\`

## Testing Your Prompts

Always test and iterate on prompts:

\`\`\`python
class PromptTester:
    """Test prompt effectiveness."""
    
    def test_prompt_variants(
        self,
        variants: List[str],
        test_cases: List[str],
        evaluate_fn: callable
    ) -> Dict[str, float]:
        """Test multiple prompt variants."""
        results = {}
        
        for i, prompt_template in enumerate(variants):
            scores = []
            
            for test_case in test_cases:
                prompt = prompt_template.format(task=test_case)
                generated = self._generate_code(prompt)
                score = evaluate_fn(generated, test_case)
                scores.append(score)
            
            avg_score = sum(scores) / len(scores)
            results[f"Variant {i+1}"] = avg_score
        
        return results
    
    def _generate_code(self, prompt: str) -> str:
        """Generate code from prompt."""
        # Call your LLM here
        pass

# Usage - test different prompt structures
variants = [
    "Write a function to {task}",
    "Create a Python function that {task}\\nUse type hints.",
    "Task: {task}\\n\\nRequirements:\\n- Type hints\\n- Docstring\\n- Error handling"
]

test_cases = [
    "calculate the factorial of a number",
    "reverse a string",
    "find the maximum value in a list"
]

def evaluate(code: str, task: str) -> float:
    """Evaluate generated code quality (0-1)."""
    # Your evaluation logic
    pass

tester = PromptTester()
results = tester.test_prompt_variants(variants, test_cases, evaluate)

print("Prompt variant scores:")
for variant, score in results.items():
    print(f"{variant}: {score:.2f}")
\`\`\`

## Best Practices Summary

### ✅ DO:
1. **Include relevant context**: file, types, imports
2. **Provide examples** of desired style
3. **Specify output format** explicitly
4. **Add constraints** to guide generation
5. **Optimize for tokens** when needed
6. **Use lower temperature** (0.2-0.3)
7. **Test prompt variants** systematically
8. **Build prompts incrementally**

### ❌ DON'T:
1. **Send entire project** as context
2. **Be vague** in instructions
3. **Forget language-specific conventions**
4. **Skip format specifications**
5. **Ignore token limits**
6. **Use high temperature** for code
7. **Assume one prompt fits all**
8. **Skip testing prompts**

## Next Steps

With solid prompt engineering, you'll learn:
- Generating complete files from scratch
- Editing existing code with diffs
- Multi-file code generation
- Building production systems

Remember: **Context + Specificity + Format = Quality Code Generation**
`,
};
