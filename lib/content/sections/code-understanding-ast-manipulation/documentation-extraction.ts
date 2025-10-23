export default {
  id: 'documentation-extraction',
  title: 'Documentation & Comment Extraction',
  content: `
# Documentation & Comment Extraction

## Introduction

Comments and docstrings contain valuable human-written context that ASTs alone can't provide. When Cursor suggests code completions or generates documentation, it uses this extracted context to understand intent, usage patterns, and domain knowledge.

**Why Documentation Extraction Matters:**

Modern AI coding tools need to:
- Extract docstrings from functions and classes
- Parse structured documentation (Google, NumPy, Sphinx styles)
- Understand inline comments
- Generate missing documentation
- Summarize code purpose
- Build context for LLMs

This section teaches you to extract and use this information effectively.

## Deep Technical Explanation

### Types of Documentation

**1. Docstrings:**
\`\`\`python
def calculate(x: int, y: int) -> int:
    """Calculate sum of two numbers.
    
    Args:
        x: First number
        y: Second number
        
    Returns:
        Sum of x and y
    """
    return x + y
\`\`\`

**2. Inline Comments:**
\`\`\`python
# Initialize configuration
config = {}  # Store user preferences
\`\`\`

**3. Type Comments:**
\`\`\`python
x = []  # type: List[int]
\`\`\`

**4. TODO/FIXME Comments:**
\`\`\`python
# TODO: Add error handling
# FIXME: This breaks with negative numbers
\`\`\`

### Documentation Formats

**Google Style:**
\`\`\`python
def function(arg1, arg2):
    """Summary line.
    
    Extended description.
    
    Args:
        arg1 (int): Description of arg1
        arg2 (str): Description of arg2
        
    Returns:
        bool: Description of return value
        
    Raises:
        ValueError: If arg1 is negative
    """
\`\`\`

**NumPy Style:**
\`\`\`python
def function(arg1, arg2):
    """
    Summary line.
    
    Extended description.
    
    Parameters
    ----------
    arg1 : int
        Description of arg1
    arg2 : str
        Description of arg2
        
    Returns
    -------
    bool
        Description of return value
    """
\`\`\`

**Sphinx Style:**
\`\`\`python
def function(arg1, arg2):
    """Summary line.
    
    :param arg1: Description of arg1
    :type arg1: int
    :param arg2: Description of arg2
    :type arg2: str
    :return: Description of return value
    :rtype: bool
    """
\`\`\`

## Code Implementation

### Docstring Extractor

\`\`\`python
import ast
from dataclasses import dataclass
from typing import Optional, List, Dict
import re

@dataclass
class DocstringInfo:
    summary: str
    description: str
    parameters: Dict[str, str]
    returns: Optional[str]
    raises: Dict[str, str]
    examples: List[str]
    raw: str

class DocstringExtractor(ast.NodeVisitor):
    """
    Extract and parse docstrings from code.
    Supports multiple documentation formats.
    """
    
    def __init__(self):
        self.docstrings: Dict[str, DocstringInfo] = {}
    
    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Extract function docstrings."""
        docstring = ast.get_docstring(node)
        if docstring:
            parsed = self._parse_docstring(docstring)
            self.docstrings[node.name] = parsed
        
        self.generic_visit(node)
    
    def visit_ClassDef(self, node: ast.ClassDef):
        """Extract class docstrings."""
        docstring = ast.get_docstring(node)
        if docstring:
            parsed = self._parse_docstring(docstring)
            self.docstrings[node.name] = parsed
        
        self.generic_visit(node)
    
    def visit_Module(self, node: ast.Module):
        """Extract module docstring."""
        docstring = ast.get_docstring(node)
        if docstring:
            parsed = self._parse_docstring(docstring)
            self.docstrings['<module>'] = parsed
        
        self.generic_visit(node)
    
    def _parse_docstring(self, docstring: str) -> DocstringInfo:
        """
        Parse docstring into structured format.
        Attempts to detect and parse Google/NumPy/Sphinx styles.
        """
        lines = docstring.strip().split('\\n')
        
        # Extract summary (first non-empty line)
        summary = lines[0].strip() if lines else ""
        
        # Detect format and parse
        if 'Args:' in docstring or 'Returns:' in docstring:
            return self._parse_google_style(docstring)
        elif 'Parameters' in docstring and '----------' in docstring:
            return self._parse_numpy_style(docstring)
        elif ':param' in docstring or ':return:' in docstring:
            return self._parse_sphinx_style(docstring)
        else:
            # Plain docstring
            return DocstringInfo(
                summary=summary,
                description=docstring,
                parameters={},
                returns=None,
                raises={},
                examples=[],
                raw=docstring
            )
    
    def _parse_google_style(self, docstring: str) -> DocstringInfo:
        """Parse Google-style docstring."""
        lines = docstring.split('\\n')
        
        summary = lines[0].strip()
        description_lines = []
        parameters = {}
        returns = None
        raises = {}
        examples = []
        
        current_section = None
        current_indent = 0
        
        for i, line in enumerate(lines[1:], 1):
            stripped = line.strip()
            
            if stripped == 'Args:':
                current_section = 'args'
                continue
            elif stripped == 'Returns:':
                current_section = 'returns'
                continue
            elif stripped == 'Raises:':
                current_section = 'raises'
                continue
            elif stripped == 'Examples:':
                current_section = 'examples'
                continue
            
            if current_section == 'args':
                # Parse parameter: name (type): description
                match = re.match(r'\\s*(\\w+)\\s*\\((.+?)\\):\\s*(.+)', line)
                if match:
                    param_name, param_type, param_desc = match.groups()
                    parameters[param_name] = f"{param_type}: {param_desc}"
            
            elif current_section == 'returns':
                if stripped:
                    returns = stripped
            
            elif current_section == 'raises':
                match = re.match(r'\\s*(\\w+):\\s*(.+)', line)
                if match:
                    exc_type, exc_desc = match.groups()
                    raises[exc_type] = exc_desc
            
            elif current_section == 'examples':
                if stripped:
                    examples.append(stripped)
            
            elif current_section is None and stripped:
                description_lines.append(stripped)
        
        return DocstringInfo(
            summary=summary,
            description=' '.join(description_lines),
            parameters=parameters,
            returns=returns,
            raises=raises,
            examples=examples,
            raw=docstring
        )
    
    def _parse_numpy_style(self, docstring: str) -> DocstringInfo:
        """Parse NumPy-style docstring."""
        # Simplified NumPy parser
        lines = docstring.split('\\n')
        summary = lines[0].strip()
        parameters = {}
        returns = None
        
        in_params = False
        current_param = None
        
        for line in lines[1:]:
            if 'Parameters' in line:
                in_params = True
                continue
            elif 'Returns' in line:
                in_params = False
                continue
            
            if in_params:
                # Check for parameter definition
                if ':' in line and not line.strip().startswith(' '):
                    match = re.match(r'(\\w+)\\s*:\\s*(.+)', line.strip())
                    if match:
                        current_param, param_type = match.groups()
                        parameters[current_param] = param_type
        
        return DocstringInfo(
            summary=summary,
            description="",
            parameters=parameters,
            returns=returns,
            raises={},
            examples=[],
            raw=docstring
        )
    
    def _parse_sphinx_style(self, docstring: str) -> DocstringInfo:
        """Parse Sphinx-style docstring."""
        lines = docstring.split('\\n')
        summary = lines[0].strip()
        parameters = {}
        returns = None
        
        for line in lines[1:]:
            # :param name: description
            param_match = re.match(r':param\\s+(\\w+):\\s*(.+)', line.strip())
            if param_match:
                param_name, param_desc = param_match.groups()
                parameters[param_name] = param_desc
            
            # :return: description
            return_match = re.match(r':return:\\s*(.+)', line.strip())
            if return_match:
                returns = return_match.group(1)
        
        return DocstringInfo(
            summary=summary,
            description="",
            parameters=parameters,
            returns=returns,
            raises={},
            examples=[],
            raw=docstring
        )
    
    def generate_summary(self) -> str:
        """Generate summary of all docstrings."""
        lines = ["=== Documentation Summary ===\\n"]
        
        for name, doc in self.docstrings.items():
            lines.append(f"## {name}")
            lines.append(f"Summary: {doc.summary}")
            
            if doc.parameters:
                lines.append("Parameters:")
                for param, desc in doc.parameters.items():
                    lines.append(f"  - {param}: {desc}")
            
            if doc.returns:
                lines.append(f"Returns: {doc.returns}")
            
            if doc.raises:
                lines.append("Raises:")
                for exc, desc in doc.raises.items():
                    lines.append(f"  - {exc}: {desc}")
            
            lines.append("")
        
        return "\\n".join(lines)

# Example usage
code = """
\"\"\"Module for data processing utilities.\"\"\"

def calculate_sum(numbers: list, initial: int = 0) -> int:
    \"\"\"Calculate sum of numbers with initial value.
    
    This function adds all numbers in the list to an initial value.
    
    Args:
        numbers (list): List of numbers to sum
        initial (int): Starting value for sum
        
    Returns:
        int: Total sum of all numbers plus initial value
        
    Raises:
        TypeError: If numbers is not a list
        
    Examples:
        >>> calculate_sum([1, 2, 3], 10)
        16
    \"\"\"
    if not isinstance(numbers, list):
        raise TypeError("numbers must be a list")
    return sum(numbers) + initial

class DataProcessor:
    \"\"\"Process and transform data.
    
    This class provides methods for data processing operations.
    
    Attributes:
        config (dict): Configuration settings
    \"\"\"
    
    def transform(self, data: str) -> str:
        \"\"\"Transform input data.
        
        :param data: Input data to transform
        :type data: str
        :return: Transformed data
        :rtype: str
        \"\"\"
        return data.upper()
"""

extractor = DocstringExtractor()
tree = ast.parse(code)
extractor.visit(tree)

print(extractor.generate_summary())
\`\`\`

### Comment Extractor

\`\`\`python
import ast
import tokenize
import io
from dataclasses import dataclass
from typing import List

@dataclass
class Comment:
    text: str
    line: int
    type: str  # 'inline', 'block', 'todo', 'fixme', 'note'

class CommentExtractor:
    """
    Extract comments from Python code.
    AST doesn't preserve comments, so we use tokenize.
    """
    
    def __init__(self, code: str):
        self.code = code
        self.comments: List[Comment] = []
        self._extract_comments()
    
    def _extract_comments(self):
        """Extract all comments from code."""
        try:
            tokens = tokenize.generate_tokens(io.StringIO(self.code).readline)
            
            for token in tokens:
                if token.type == tokenize.COMMENT:
                    comment_text = token.string[1:].strip()  # Remove #
                    comment_type = self._classify_comment(comment_text)
                    
                    self.comments.append(Comment(
                        text=comment_text,
                        line=token.start[0],
                        type=comment_type
                    ))
        except tokenize.TokenError:
            # Handle incomplete code gracefully
            pass
    
    def _classify_comment(self, text: str) -> str:
        """Classify comment type."""
        text_lower = text.lower()
        
        if text_lower.startswith('todo'):
            return 'todo'
        elif text_lower.startswith('fixme'):
            return 'fixme'
        elif text_lower.startswith('note'):
            return 'note'
        elif text_lower.startswith('hack'):
            return 'hack'
        elif text_lower.startswith('xxx'):
            return 'xxx'
        else:
            return 'inline'
    
    def get_todos(self) -> List[Comment]:
        """Get all TODO comments."""
        return [c for c in self.comments if c.type == 'todo']
    
    def get_fixmes(self) -> List[Comment]:
        """Get all FIXME comments."""
        return [c for c in self.comments if c.type == 'fixme']
    
    def generate_report(self) -> str:
        """Generate comment report."""
        lines = [f"=== Found {len(self.comments)} Comments ===\\n"]
        
        # Group by type
        by_type = {}
        for comment in self.comments:
            by_type.setdefault(comment.type, []).append(comment)
        
        for comment_type, comments in sorted(by_type.items()):
            lines.append(f"{comment_type.upper()}:")
            for comment in comments:
                lines.append(f"  Line {comment.line}: {comment.text}")
            lines.append("")
        
        return "\\n".join(lines)

# Example usage
code = """
# This module handles user authentication
# TODO: Add OAuth support

def login(username, password):
    # Validate credentials
    # FIXME: This is vulnerable to timing attacks
    if username == "admin" and password == "secret":
        return True
    
    # NOTE: Log failed attempts
    log_failure(username)
    
    # HACK: Temporary workaround for bug #123
    return False

# XXX: This entire function needs rewrite
def legacy_code():
    pass
"""

extractor = CommentExtractor(code)
print(extractor.generate_report())

# Get action items
todos = extractor.get_todos()
fixmes = extractor.get_fixmes()
print(f"\\n⚠️  {len(todos)} TODOs and {len(fixmes)} FIXMEs need attention")
\`\`\`

### Documentation Generator

\`\`\`python
import ast
from typing import List

class DocumentationGenerator(ast.NodeVisitor):
    """
    Generate missing documentation for code.
    Creates docstring templates based on function signatures.
    """
    
    def __init__(self):
        self.functions_without_docs: List[ast.FunctionDef] = []
    
    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Find functions without docstrings."""
        if not ast.get_docstring(node):
            self.functions_without_docs.append(node)
        self.generic_visit(node)
    
    def generate_docstring(self, func: ast.FunctionDef, style: str = 'google') -> str:
        """
        Generate docstring template for a function.
        
        Args:
            func: Function AST node
            style: Documentation style ('google', 'numpy', 'sphinx')
            
        Returns:
            Generated docstring template
        """
        if style == 'google':
            return self._generate_google_style(func)
        elif style == 'numpy':
            return self._generate_numpy_style(func)
        elif style == 'sphinx':
            return self._generate_sphinx_style(func)
        return ""
    
    def _generate_google_style(self, func: ast.FunctionDef) -> str:
        """Generate Google-style docstring."""
        lines = [f'"""Summary of {func.name}.', '', '']
        
        # Parameters
        if func.args.args:
            lines.append('Args:')
            for arg in func.args.args:
                arg_type = "type"
                if arg.annotation:
                    arg_type = ast.unparse(arg.annotation)
                lines.append(f'    {arg.arg} ({arg_type}): Description of {arg.arg}')
            lines.append('')
        
        # Returns
        if func.returns:
            return_type = ast.unparse(func.returns)
            lines.append('Returns:')
            lines.append(f'    {return_type}: Description of return value')
            lines.append('')
        
        lines.append('"""')
        return '\\n'.join(lines)
    
    def _generate_numpy_style(self, func: ast.FunctionDef) -> str:
        """Generate NumPy-style docstring."""
        lines = [f'"""', f'Summary of {func.name}.', '', '']
        
        # Parameters
        if func.args.args:
            lines.append('Parameters')
            lines.append('----------')
            for arg in func.args.args:
                arg_type = "type"
                if arg.annotation:
                    arg_type = ast.unparse(arg.annotation)
                lines.append(f'{arg.arg} : {arg_type}')
                lines.append(f'    Description of {arg.arg}')
            lines.append('')
        
        # Returns
        if func.returns:
            return_type = ast.unparse(func.returns)
            lines.append('Returns')
            lines.append('-------')
            lines.append(return_type)
            lines.append('    Description of return value')
            lines.append('')
        
        lines.append('"""')
        return '\\n'.join(lines)
    
    def _generate_sphinx_style(self, func: ast.FunctionDef) -> str:
        """Generate Sphinx-style docstring."""
        lines = [f'"""Summary of {func.name}.', '']
        
        # Parameters
        for arg in func.args.args:
            arg_type = "type"
            if arg.annotation:
                arg_type = ast.unparse(arg.annotation)
            lines.append(f':param {arg.arg}: Description of {arg.arg}')
            lines.append(f':type {arg.arg}: {arg_type}')
        
        # Returns
        if func.returns:
            return_type = ast.unparse(func.returns)
            lines.append(f':return: Description of return value')
            lines.append(f':rtype: {return_type}')
        
        lines.append('"""')
        return '\\n'.join(lines)
    
    def generate_all_missing(self, style: str = 'google') -> Dict[str, str]:
        """Generate docstrings for all functions missing them."""
        docstrings = {}
        for func in self.functions_without_docs:
            docstring = self.generate_docstring(func, style)
            docstrings[func.name] = docstring
        return docstrings

# Example usage
code = """
def calculate(x: int, y: int) -> int:
    return x + y

def process_data(items: list, strict: bool = False) -> dict:
    results = {}
    for item in items:
        results[item] = process(item)
    return results

def legacy_function(arg1, arg2, arg3):
    # No type hints, no docstring
    return arg1 + arg2 + arg3
"""

generator = DocumentationGenerator()
tree = ast.parse(code)
generator.visit(tree)

print(f"Found {len(generator.functions_without_docs)} functions without documentation\\n")

# Generate docstrings in different styles
print("=== Google Style ===")
for func in generator.functions_without_docs[:1]:
    print(generator.generate_docstring(func, 'google'))

print("\\n=== NumPy Style ===")
for func in generator.functions_without_docs[:1]:
    print(generator.generate_docstring(func, 'numpy'))

print("\\n=== Sphinx Style ===")
for func in generator.functions_without_docs[:1]:
    print(generator.generate_docstring(func, 'sphinx'))
\`\`\`

### Complete Documentation System

\`\`\`python
class ComprehensiveDocumentationSystem:
    """
    Complete documentation analysis and generation system.
    Combines extraction, validation, and generation.
    """
    
    def __init__(self, code: str):
        self.code = code
        self.tree = ast.parse(code)
        
        # Extract existing documentation
        self.docstring_extractor = DocstringExtractor()
        self.docstring_extractor.visit(self.tree)
        
        # Extract comments
        self.comment_extractor = CommentExtractor(code)
        
        # Find missing documentation
        self.doc_generator = DocumentationGenerator()
        self.doc_generator.visit(self.tree)
    
    def analyze_documentation_coverage(self) -> Dict[str, any]:
        """Analyze how well code is documented."""
        # Count functions
        total_functions = 0
        documented_functions = 0
        
        for node in ast.walk(self.tree):
            if isinstance(node, ast.FunctionDef):
                total_functions += 1
                if ast.get_docstring(node):
                    documented_functions += 1
        
        coverage = (documented_functions / total_functions * 100) if total_functions > 0 else 0
        
        return {
            'total_functions': total_functions,
            'documented_functions': documented_functions,
            'undocumented_functions': total_functions - documented_functions,
            'coverage_percent': round(coverage, 2),
            'total_comments': len(self.comment_extractor.comments),
            'todos': len(self.comment_extractor.get_todos()),
            'fixmes': len(self.comment_extractor.get_fixmes())
        }
    
    def generate_documentation_report(self) -> str:
        """Generate comprehensive documentation report."""
        stats = self.analyze_documentation_coverage()
        
        lines = ["=" * 60]
        lines.append("DOCUMENTATION ANALYSIS REPORT")
        lines.append("=" * 60)
        
        # Coverage stats
        lines.append("\\n## Coverage Statistics")
        lines.append(f"  Total Functions: {stats['total_functions']}")
        lines.append(f"  Documented: {stats['documented_functions']}")
        lines.append(f"  Missing Docs: {stats['undocumented_functions']}")
        lines.append(f"  Coverage: {stats['coverage_percent']}%")
        
        # Visual indicator
        if stats['coverage_percent'] >= 80:
            lines.append("  ✅ Good documentation coverage")
        elif stats['coverage_percent'] >= 50:
            lines.append("  ⚠️  Moderate documentation coverage")
        else:
            lines.append("  🚨 Poor documentation coverage")
        
        # Comment stats
        lines.append(f"\\n## Comments")
        lines.append(f"  Total Comments: {stats['total_comments']}")
        lines.append(f"  TODOs: {stats['todos']}")
        lines.append(f"  FIXMEs: {stats['fixmes']}")
        
        # Existing documentation
        if self.docstring_extractor.docstrings:
            lines.append(f"\\n{self.docstring_extractor.generate_summary()}")
        
        # Action items
        lines.append("\\n## Action Items")
        if stats['undocumented_functions'] > 0:
            lines.append(f"  📝 Add docstrings to {stats['undocumented_functions']} functions")
        if stats['todos'] > 0:
            lines.append(f"  ✅ Complete {stats['todos']} TODO items")
        if stats['fixmes'] > 0:
            lines.append(f"  🔧 Fix {stats['fixmes']} FIXME items")
        
        return "\\n".join(lines)
    
    def generate_llm_context(self) -> str:
        """
        Generate documentation context for LLM.
        This is what Cursor sends to provide code context.
        """
        lines = ["# Code Documentation Context\\n"]
        
        # Add existing docstrings
        lines.append("## Documented Functions")
        for name, doc in self.docstring_extractor.docstrings.items():
            if name != '<module>':
                lines.append(f"### {name}")
                lines.append(doc.summary)
                if doc.parameters:
                    params = ", ".join(f"{p}: {d}" for p, d in doc.parameters.items())
                    lines.append(f"Parameters: {params}")
                lines.append("")
        
        # Add relevant comments
        todos = self.comment_extractor.get_todos()
        if todos:
            lines.append("## TODO Items")
            for todo in todos:
                lines.append(f"- Line {todo.line}: {todo.text}")
            lines.append("")
        
        return "\\n".join(lines)

# Test the comprehensive system
code = """
\"\"\"User authentication module.\"\"\"

def authenticate(username: str, password: str) -> bool:
    \"\"\"Authenticate user credentials.
    
    Args:
        username (str): User's username
        password (str): User's password
        
    Returns:
        bool: True if authenticated, False otherwise
    \"\"\"
    # TODO: Add rate limiting
    # FIXME: Password should be hashed
    return validate_credentials(username, password)

def validate_credentials(username, password):
    # Missing docstring!
    return username == "admin" and password == "secret"

class UserManager:
    # TODO: Add user management methods
    
    def create_user(self, username, email):
        # Missing docstring!
        pass
"""

system = ComprehensiveDocumentationSystem(code)
print(system.generate_documentation_report())

print("\\n" + "=" * 60)
print("LLM CONTEXT")
print("=" * 60)
print(system.generate_llm_context())
\`\`\`

## Real-World Case Study: How Cursor Uses Documentation

Cursor leverages documentation extensively:

**1. Context Building:**
\`\`\`python
def process_user(user_id: int) -> User:
    """Fetch and process user data.
    
    Args:
        user_id: Unique user identifier
        
    Returns:
        Processed user object
    """
    # When generating code here, Cursor uses the docstring
    # to understand what this function should do
\`\`\`

**2. Smart Completions:**
\`\`\`python
# When you start typing a function call
authenticate(  # <-- Cursor shows: "username: str, password: str"
# From the docstring, not just the signature!
\`\`\`

**3. Documentation Generation:**
\`\`\`python
def new_function(x, y):
    # Cursor can generate: "Add docstring"
    # Produces complete Google/NumPy/Sphinx style docstring
\`\`\`

**4. TODO Tracking:**
\`\`\`python
# TODO: Implement caching
# Cursor highlights TODOs and can list them all
\`\`\`

## Common Pitfalls

### 1. Not Preserving Docstring Format

\`\`\`python
# ❌ Wrong: Loses formatting
docstring = ast.get_docstring(node)
# Multi-line formatting is lost!

# ✅ Correct: Use raw docstring if needed
if node.body and isinstance(node.body[0], ast.Expr):
    if isinstance(node.body[0].value, ast.Constant):
        raw_docstring = node.body[0].value.value
\`\`\`

### 2. Assuming Format

\`\`\`python
# ❌ Wrong: Assumes Google style
lines = docstring.split('Args:')
# Breaks if NumPy or Sphinx style!

# ✅ Correct: Detect format first
if 'Args:' in docstring:
    parse_google_style(docstring)
elif 'Parameters' in docstring:
    parse_numpy_style(docstring)
\`\`\`

### 3. Missing Comments

\`\`\`python
# ❌ Wrong: Only uses AST
# Comments are not in AST!

# ✅ Correct: Use tokenize module
import tokenize
# Can extract comments
\`\`\`

## Production Checklist

### Extraction
- [ ] Extract module docstrings
- [ ] Extract class docstrings
- [ ] Extract function/method docstrings
- [ ] Extract inline comments
- [ ] Extract type comments
- [ ] Detect documentation format

### Parsing
- [ ] Support Google style
- [ ] Support NumPy style
- [ ] Support Sphinx style
- [ ] Handle malformed docstrings
- [ ] Parse parameter descriptions
- [ ] Extract examples

### Generation
- [ ] Generate missing docstrings
- [ ] Match existing style
- [ ] Include type information
- [ ] Generate examples
- [ ] Add TODO placeholders
- [ ] Format consistently

### Analysis
- [ ] Calculate documentation coverage
- [ ] Find missing docstrings
- [ ] Track TODO/FIXME items
- [ ] Validate docstring format
- [ ] Check parameter documentation
- [ ] Verify examples

### Integration
- [ ] Export to various formats
- [ ] Generate API documentation
- [ ] Provide LLM context
- [ ] Support documentation tools
- [ ] Enable search/indexing

## Summary

Documentation extraction provides essential context:

- **Docstring Parsing**: Extract structured documentation
- **Comment Analysis**: Find TODOs, FIXMEs, and notes
- **Format Detection**: Support multiple documentation styles
- **Generation**: Create missing documentation
- **Coverage Analysis**: Measure documentation quality

These capabilities enable AI coding tools like Cursor to understand not just what code does syntactically, but why it exists and how it should be used—critical for providing truly intelligent assistance that respects the developer's intent and project conventions.

In the next section, we'll explore code similarity and clone detection—finding duplicate and similar code patterns.
`,
};
