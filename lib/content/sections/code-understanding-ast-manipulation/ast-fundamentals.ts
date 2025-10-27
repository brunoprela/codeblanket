const astFundamentals = {
  id: 'ast-fundamentals',
  title: 'Abstract Syntax Trees (AST) Fundamentals',
  content: `
# Abstract Syntax Trees (AST) Fundamentals

## Introduction

Understanding Abstract Syntax Trees (ASTs) is fundamental to building sophisticated code analysis and generation tools like Cursor, GitHub Copilot, and modern IDEs. An AST is a tree representation of the abstract syntactic structure of source code, where each node represents a construct in the programming language.

**Why ASTs Matter for AI Code Tools:**

When Cursor analyzes your code to provide intelligent suggestions, it doesn't just work with raw text. It parses the code into an AST to understand:
- Function definitions and their boundaries
- Variable scopes and declarations
- Import statements and dependencies
- Code structure and hierarchies
- Semantic relationships between code elements

This structured representation allows AI tools to provide context-aware suggestions, accurate refactoring, and intelligent code generation that respects the existing codebase structure.

## Deep Technical Explanation

### From Source Code to AST: The Parsing Pipeline

The journey from source code to an AST involves several stages:

**1. Lexical Analysis (Tokenization):**
The source code string is broken down into tokens—the smallest meaningful units like keywords, identifiers, operators, and literals.

\`\`\`python
# Source code
def add (x, y):
    return x + y

# Tokens produced
[
    ('def', KEYWORD),
    ('add', IDENTIFIER),
    ('(', DELIMITER),
    ('x', IDENTIFIER),
    (',', DELIMITER),
    ('y', IDENTIFIER),
    (')', DELIMITER),
    (':', DELIMITER),
    ('return', KEYWORD),
    ('x', IDENTIFIER),
    ('+', OPERATOR),
    ('y', IDENTIFIER)
]
\`\`\`

**2. Syntactic Analysis (Parsing):**
Tokens are organized into a parse tree based on grammar rules, then simplified into an AST by removing syntactic details.

**3. AST Construction:**
The parser builds a tree where:
- Each node represents a language construct
- Node types correspond to grammar rules (FunctionDef, BinOp, Return, etc.)
- Children represent sub-components
- Syntactic noise (parentheses, semicolons) is abstracted away

### AST vs Parse Tree vs Tokens

**Tokens (Flat List):**
- One-dimensional sequence
- Includes all syntactic elements
- No structural information
- Fast to produce but limited usefulness

**Parse Tree (Concrete Syntax Tree):**
- Represents exact grammar derivation
- Includes all syntactic elements (parentheses, semicolons)
- Very detailed but noisy
- Directly follows grammar rules

**AST (Abstract Syntax Tree):**
- Abstracts away syntactic details
- Focuses on semantic structure
- Smaller and easier to work with
- Perfect for analysis and transformation

\`\`\`python
# For the expression: (x + y) * 2

# Parse Tree (verbose):
Expression
  ├── '('
  ├── Expression
  │   ├── Identifier: x
  │   ├── Operator: +
  │   └── Identifier: y
  ├── ')'
  ├── Operator: *
  └── Number: 2

# AST (clean):
BinOp
  ├── op: Mult
  ├── left: BinOp
  │   ├── op: Add
  │   ├── left: Name (id='x')
  │   └── right: Name (id='y')
  └── right: Constant (value=2)
\`\`\`

### AST Node Structure

Every AST node typically contains:

1. **Node Type**: The kind of construct (FunctionDef, If, While, BinOp, etc.)
2. **Attributes**: Type-specific data (function name, operator type, value)
3. **Children**: Sub-nodes forming the tree structure
4. **Location Info**: Line numbers, column offsets for error reporting

\`\`\`python
# Python\'s ast module representation
ast.FunctionDef(
    name='add',                    # Function name
    args=ast.arguments(            # Parameters
        args=[
            ast.arg (arg='x', annotation=None),
            ast.arg (arg='y', annotation=None)
        ],
        defaults=[]
    ),
    body=[                         # Function body
        ast.Return(
            value=ast.BinOp(
                left=ast.Name (id='x'),
                op=ast.Add(),
                right=ast.Name (id='y')
            )
        )
    ],
    decorator_list=[],             # Decorators
    returns=None,                  # Return annotation
    lineno=1,                      # Line number
    col_offset=0                   # Column offset
)
\`\`\`

## Code Implementation

### Basic AST Parsing and Inspection

\`\`\`python
import ast
import json
from typing import Any, Dict

def parse_code (source_code: str) -> ast.AST:
    """
    Parse Python source code into an AST.
    
    Args:
        source_code: Python source code as string
        
    Returns:
        Root AST node
        
    Raises:
        SyntaxError: If code has syntax errors
    """
    try:
        tree = ast.parse (source_code)
        return tree
    except SyntaxError as e:
        print(f"Syntax error at line {e.lineno}: {e.msg}")
        raise

# Example: Parse a simple function
code = """
def calculate_total (items, tax_rate=0.1):
    """Calculate total with tax."""
    subtotal = sum (item['price'] for item in items)
    tax = subtotal * tax_rate
    return subtotal + tax

class ShoppingCart:
    def __init__(self):
        self.items = []
    
    def add_item (self, item):
        self.items.append (item)
"""

tree = parse_code (code)
print(f"Root node type: {type (tree).__name__}")
print(f"Body has {len (tree.body)} top-level statements")
\`\`\`

### Walking the AST

\`\`\`python
import ast

class ASTVisitor (ast.NodeVisitor):
    """
    Custom AST visitor that tracks what it finds.
    This is how Cursor analyzes your codebase structure.
    """
    
    def __init__(self):
        self.functions = []
        self.classes = []
        self.imports = []
        self.variables = []
    
    def visit_FunctionDef (self, node: ast.FunctionDef):
        """Called for each function definition."""
        self.functions.append({
            'name': node.name,
            'line': node.lineno,
            'args': [arg.arg for arg in node.args.args],
            'docstring': ast.get_docstring (node)
        })
        # Continue visiting child nodes
        self.generic_visit (node)
    
    def visit_ClassDef (self, node: ast.ClassDef):
        """Called for each class definition."""
        methods = [
            n.name for n in node.body 
            if isinstance (n, ast.FunctionDef)
        ]
        self.classes.append({
            'name': node.name,
            'line': node.lineno,
            'methods': methods,
            'bases': [self._get_name (base) for base in node.bases]
        })
        self.generic_visit (node)
    
    def visit_Import (self, node: ast.Import):
        """Called for 'import x' statements."""
        for alias in node.names:
            self.imports.append({
                'type': 'import',
                'module': alias.name,
                'asname': alias.asname,
                'line': node.lineno
            })
    
    def visit_ImportFrom (self, node: ast.ImportFrom):
        """Called for 'from x import y' statements."""
        for alias in node.names:
            self.imports.append({
                'type': 'from',
                'module': node.module,
                'name': alias.name,
                'asname': alias.asname,
                'line': node.lineno
            })
    
    def visit_Assign (self, node: ast.Assign):
        """Called for variable assignments."""
        # Only track module-level assignments
        for target in node.targets:
            if isinstance (target, ast.Name):
                self.variables.append({
                    'name': target.id,
                    'line': node.lineno
                })
        self.generic_visit (node)
    
    def _get_name (self, node):
        """Extract name from various node types."""
        if isinstance (node, ast.Name):
            return node.id
        elif isinstance (node, ast.Attribute):
            return f"{self._get_name (node.value)}.{node.attr}"
        return str (node)

# Analyze the code
visitor = ASTVisitor()
visitor.visit (tree)

print("\\n=== Functions Found ===")
for func in visitor.functions:
    print(f"  {func['name']}({', '.join (func['args'])}) at line {func['line']}")
    if func['docstring']:
        print(f"    Doc: {func['docstring'][:50]}...")

print("\\n=== Classes Found ===")
for cls in visitor.classes:
    print(f"  {cls['name']} at line {cls['line']}")
    print(f"    Methods: {', '.join (cls['methods'])}")
\`\`\`

### AST Visualization

\`\`\`python
import ast

def visualize_ast (node: ast.AST, indent: int = 0) -> str:
    """
    Create a visual representation of an AST.
    Useful for understanding structure during development.
    """
    result = []
    prefix = "  " * indent
    
    # Node type
    node_name = node.__class__.__name__
    result.append (f"{prefix}{node_name}")
    
    # Node fields
    for field_name, field_value in ast.iter_fields (node):
        if isinstance (field_value, list):
            if field_value:
                result.append (f"{prefix}  {field_name}:")
                for item in field_value:
                    if isinstance (item, ast.AST):
                        result.append (visualize_ast (item, indent + 2))
                    else:
                        result.append (f"{prefix}    {item}")
        elif isinstance (field_value, ast.AST):
            result.append (f"{prefix}  {field_name}:")
            result.append (visualize_ast (field_value, indent + 2))
        else:
            result.append (f"{prefix}  {field_name}: {field_value}")
    
    return "\\n".join (result)

# Visualize a simple expression
simple_code = "result = (x + y) * 2"
simple_tree = ast.parse (simple_code)
print(visualize_ast (simple_tree))
\`\`\`

### Finding Specific Patterns

\`\`\`python
import ast
from typing import List

class PatternFinder (ast.NodeVisitor):
    """
    Find specific code patterns in an AST.
    Similar to how linters and code analysis tools work.
    """
    
    def __init__(self):
        self.issues = []
    
    def visit_FunctionDef (self, node: ast.FunctionDef):
        # Check for functions without docstrings
        if not ast.get_docstring (node):
            self.issues.append({
                'type': 'missing_docstring',
                'function': node.name,
                'line': node.lineno,
                'message': f"Function '{node.name}' has no docstring"
            })
        
        # Check for functions with too many parameters
        if len (node.args.args) > 5:
            self.issues.append({
                'type': 'too_many_params',
                'function': node.name,
                'line': node.lineno,
                'count': len (node.args.args),
                'message': f"Function '{node.name}' has {len (node.args.args)} parameters (max 5 recommended)"
            })
        
        # Check for deeply nested code
        max_depth = self._calculate_depth (node.body)
        if max_depth > 4:
            self.issues.append({
                'type': 'deep_nesting',
                'function': node.name,
                'line': node.lineno,
                'depth': max_depth,
                'message': f"Function '{node.name}' has nesting depth {max_depth} (max 4 recommended)"
            })
        
        self.generic_visit (node)
    
    def visit_Compare (self, node: ast.Compare):
        # Check for comparison with True/False
        for op, comparator in zip (node.ops, node.comparators):
            if isinstance (comparator, ast.Constant):
                if comparator.value is True or comparator.value is False:
                    self.issues.append({
                        'type': 'compare_with_boolean',
                        'line': node.lineno,
                        'message': f"Comparing with {comparator.value} is unnecessary"
                    })
        self.generic_visit (node)
    
    def visit_Try (self, node: ast.Try):
        # Check for bare except clauses
        for handler in node.handlers:
            if handler.type is None:
                self.issues.append({
                    'type': 'bare_except',
                    'line': handler.lineno,
                    'message': "Bare 'except:' clause catches all exceptions (anti-pattern)"
                })
        self.generic_visit (node)
    
    def _calculate_depth (self, body: List[ast.stmt], depth: int = 0) -> int:
        """Calculate maximum nesting depth in a code block."""
        max_depth = depth
        for node in body:
            if isinstance (node, (ast.If, ast.For, ast.While, ast.With)):
                nested_body = getattr (node, 'body', [])
                nested_depth = self._calculate_depth (nested_body, depth + 1)
                max_depth = max (max_depth, nested_depth)
        return max_depth

# Analyze code for patterns
bad_code = """
def process_data (a, b, c, d, e, f):
    # Too many parameters, no docstring
    if condition1:
        if condition2:
            if condition3:
                if condition4:
                    if condition5:
                        pass  # Deep nesting
    
    if value == True:  # Bad comparison
        pass
    
    try:
        risky_operation()
    except:  # Bare except
        pass
"""

tree = ast.parse (bad_code)
finder = PatternFinder()
finder.visit (tree)

print("\\n=== Code Issues Found ===")
for issue in finder.issues:
    print(f"[{issue['type']}] Line {issue['line']}: {issue['message']}")
\`\`\`

### AST Manipulation Basics

\`\`\`python
import ast

class NameTransformer (ast.NodeTransformer):
    """
    Transform AST by modifying nodes.
    This is how refactoring tools rename variables.
    """
    
    def __init__(self, old_name: str, new_name: str):
        self.old_name = old_name
        self.new_name = new_name
        self.count = 0
    
    def visit_Name (self, node: ast.Name):
        """Replace variable names."""
        if node.id == self.old_name:
            self.count += 1
            node.id = self.new_name
        return node
    
    def visit_FunctionDef (self, node: ast.FunctionDef):
        """Also handle function names and parameters."""
        if node.name == self.old_name:
            self.count += 1
            node.name = self.new_name
        
        # Handle parameters
        for arg in node.args.args:
            if arg.arg == self.old_name:
                self.count += 1
                arg.arg = self.new_name
        
        # Continue visiting children
        self.generic_visit (node)
        return node

# Example: Rename a variable throughout code
code = """
def calculate (value):
    result = value * 2
    return result

def process (value):
    temp = calculate (value)
    return temp + value
"""

tree = ast.parse (code)
transformer = NameTransformer('value', 'input_value')
new_tree = transformer.visit (tree)

# Fix missing locations after transformation
ast.fix_missing_locations (new_tree)

# Convert back to code
import astunparse  # pip install astunparse
new_code = astunparse.unparse (new_tree)
print(f"\\nRenamed {transformer.count} occurrences")
print("\\nTransformed code:")
print(new_code)
\`\`\`

## Real-World Case Study: How Cursor Uses ASTs

Cursor leverages ASTs extensively for intelligent code understanding:

**1. Context Building:**
When you ask Cursor to modify a function, it:
- Parses the entire file into an AST
- Locates the target function node
- Extracts function signature, parameters, return type
- Identifies all variables used in the function
- Finds related imports and dependencies
- Builds a context window with relevant code

**2. Intelligent Suggestions:**
\`\`\`python
# When you're typing inside a function:
def calculate_discount (price, customer):
    # Cursor types here, it knows:
    # - 'price' is a parameter (from AST)
    # - 'customer' is a parameter (from AST)
    # - Customer might have attributes (from type hints in AST)
    # - Should suggest: customer.membership_level, price * 0.9, etc.
\`\`\`

**3. Refactoring Operations:**
When Cursor refactors code:
- Parse AST to find all references
- Transform AST nodes systematically
- Ensure scope correctness using AST structure
- Regenerate code from modified AST

**4. Code Generation:**
When generating new code, Cursor:
- Analyzes existing AST patterns
- Matches coding style from AST structure
- Ensures generated code fits AST structure of file
- Validates that generated AST is well-formed

\`\`\`python
# How Cursor might analyze this for context:
class UserService:
    def __init__(self, db):
        self.db = db
    
    def get_user (self, user_id):
        # When editing here, Cursor\'s AST analysis knows:
        # - self.db is available (from __init__)
        # - user_id is the parameter
        # - This is an instance method
        # - Return type should be User or None
        pass  # <-- Cursor provides intelligent completion here
\`\`\`

## Hands-On Exercise

Build an AST analyzer that extracts a "code map" for an AI tool:

\`\`\`python
import ast
from dataclasses import dataclass
from typing import List, Dict, Optional

@dataclass
class FunctionInfo:
    name: str
    line: int
    params: List[str]
    return_type: Optional[str]
    docstring: Optional[str]
    calls: List[str]  # Functions this function calls
    
@dataclass
class ClassInfo:
    name: str
    line: int
    methods: List[str]
    attributes: List[str]
    bases: List[str]

class CodeMapBuilder (ast.NodeVisitor):
    """
    Build a comprehensive map of code structure.
    This is similar to what Cursor does for code understanding.
    """
    
    def __init__(self):
        self.functions: Dict[str, FunctionInfo] = {}
        self.classes: Dict[str, ClassInfo] = {}
        self.current_function: Optional[str] = None
    
    def visit_FunctionDef (self, node: ast.FunctionDef):
        # Extract parameter names
        params = [arg.arg for arg in node.args.args]
        
        # Extract return type if annotated
        return_type = None
        if node.returns:
            return_type = ast.unparse (node.returns)
        
        # Get docstring
        docstring = ast.get_docstring (node)
        
        # Track function
        func_info = FunctionInfo(
            name=node.name,
            line=node.lineno,
            params=params,
            return_type=return_type,
            docstring=docstring,
            calls=[]
        )
        self.functions[node.name] = func_info
        
        # Visit body to find function calls
        old_func = self.current_function
        self.current_function = node.name
        self.generic_visit (node)
        self.current_function = old_func
    
    def visit_Call (self, node: ast.Call):
        # Track function calls
        if self.current_function:
            if isinstance (node.func, ast.Name):
                func_name = node.func.id
                self.functions[self.current_function].calls.append (func_name)
            elif isinstance (node.func, ast.Attribute):
                func_name = node.func.attr
                self.functions[self.current_function].calls.append (func_name)
        
        self.generic_visit (node)
    
    def visit_ClassDef (self, node: ast.ClassDef):
        # Extract methods
        methods = [
            n.name for n in node.body 
            if isinstance (n, ast.FunctionDef)
        ]
        
        # Extract class attributes
        attributes = []
        for n in node.body:
            if isinstance (n, ast.Assign):
                for target in n.targets:
                    if isinstance (target, ast.Name):
                        attributes.append (target.id)
        
        # Extract base classes
        bases = []
        for base in node.bases:
            if isinstance (base, ast.Name):
                bases.append (base.id)
        
        class_info = ClassInfo(
            name=node.name,
            line=node.lineno,
            methods=methods,
            attributes=attributes,
            bases=bases
        )
        self.classes[node.name] = class_info
        
        self.generic_visit (node)
    
    def generate_summary (self) -> str:
        """Generate a text summary suitable for LLM context."""
        lines = ["# Code Structure Summary\\n"]
        
        if self.classes:
            lines.append("## Classes\\n")
            for cls in self.classes.values():
                lines.append (f"### {cls.name} (line {cls.line})")
                if cls.bases:
                    lines.append (f"  Inherits: {', '.join (cls.bases)}")
                lines.append (f"  Methods: {', '.join (cls.methods)}")
                if cls.attributes:
                    lines.append (f"  Attributes: {', '.join (cls.attributes)}")
                lines.append("")
        
        if self.functions:
            lines.append("## Functions\\n")
            for func in self.functions.values():
                signature = f"{func.name}({', '.join (func.params)})"
                if func.return_type:
                    signature += f" -> {func.return_type}"
                lines.append (f"### {signature} (line {func.line})")
                if func.docstring:
                    lines.append (f"  {func.docstring.split (chr(10))[0]}")
                if func.calls:
                    lines.append (f"  Calls: {', '.join (set (func.calls))}")
                lines.append("")
        
        return "\\n".join (lines)

# Test with example code
example_code = """
class DataProcessor:
    def __init__(self, config):
        self.config = config
        self.cache = {}
    
    def process (self, data: list) -> dict:
        ''Process input data and return results.''
        cleaned = self.clean_data (data)
        result = self.transform (cleaned)
        return result
    
    def clean_data (self, data: list) -> list:
        ''Remove invalid entries.''
        return [x for x in data if self.validate (x)]
    
    def transform (self, data: list) -> dict:
        ''Transform data to output format.''
        return {'processed': data, 'count': len (data)}
    
    def validate (self, item) -> bool:
        ''Check if item is valid.''
        return item is not None

def main():
    processor = DataProcessor({'strict': True})
    result = processor.process([1, 2, None, 3])
    print(result)
"""

builder = CodeMapBuilder()
tree = ast.parse (example_code)
builder.visit (tree)

print(builder.generate_summary())
\`\`\`

**Exercise Tasks:**1. Extend the CodeMapBuilder to track variable assignments
2. Add detection for method calls within classes
3. Build a dependency graph showing which functions call which
4. Generate a JSON representation suitable for an API

## Common Pitfalls

### 1. Forgetting to Fix Missing Locations

\`\`\`python
# ❌ Wrong: Modified AST has invalid location info
tree = ast.parse (code)
transformer = MyTransformer()
new_tree = transformer.visit (tree)
code = ast.unparse (new_tree)  # May cause issues

# ✅ Correct: Always fix locations after modification
tree = ast.parse (code)
transformer = MyTransformer()
new_tree = transformer.visit (tree)
ast.fix_missing_locations (new_tree)  # Required!
code = ast.unparse (new_tree)
\`\`\`

### 2. Not Handling All Node Types

\`\`\`python
# ❌ Wrong: Only handles Name nodes
class VariableFinder (ast.NodeVisitor):
    def visit_Name (self, node):
        print(node.id)
        # Forgot to call generic_visit!

# ✅ Correct: Always call generic_visit to traverse children
class VariableFinder (ast.NodeVisitor):
    def visit_Name (self, node):
        print(node.id)
        self.generic_visit (node)  # Important!
\`\`\`

### 3. Assuming AST Structure

\`\`\`python
# ❌ Wrong: Assumes specific structure
def get_function_name (node):
    return node.body[0].name  # Crashes if body is empty

# ✅ Correct: Check node types and structure
def get_function_name (node):
    if not isinstance (node, ast.Module):
        return None
    if not node.body:
        return None
    first = node.body[0]
    if not isinstance (first, ast.FunctionDef):
        return None
    return first.name
\`\`\`

### 4. Not Preserving Type Information

\`\`\`python
# ❌ Wrong: Loses type hints during transformation
def transform_function (func_node):
    new_node = ast.FunctionDef(
        name=func_node.name,
        args=func_node.args,
        body=func_node.body,
        decorator_list=[]
    )
    # Lost: returns, type_comment

# ✅ Correct: Preserve all attributes
def transform_function (func_node):
    new_node = ast.FunctionDef(
        name=func_node.name,
        args=func_node.args,
        body=func_node.body,
        decorator_list=func_node.decorator_list,
        returns=func_node.returns,  # Preserve
        type_comment=func_node.type_comment  # Preserve
    )
    return new_node
\`\`\`

### 5. Inefficient AST Traversal

\`\`\`python
# ❌ Wrong: Re-parsing for every analysis
def count_functions (code):
    tree = ast.parse (code)
    return len([n for n in ast.walk (tree) if isinstance (n, ast.FunctionDef)])

def count_classes (code):
    tree = ast.parse (code)  # Parsing again!
    return len([n for n in ast.walk (tree) if isinstance (n, ast.ClassDef)])

# ✅ Correct: Parse once, analyze multiple times
tree = ast.parse (code)
function_count = len([n for n in ast.walk (tree) if isinstance (n, ast.FunctionDef)])
class_count = len([n for n in ast.walk (tree) if isinstance (n, ast.ClassDef)])
\`\`\`

## Production Checklist

### Code Parsing
- [ ] Handle syntax errors gracefully with clear error messages
- [ ] Support different Python versions (syntax differences)
- [ ] Cache parsed ASTs for frequently accessed files
- [ ] Implement timeout for parsing very large files
- [ ] Validate AST structure after modifications

### AST Analysis
- [ ] Use appropriate visitor pattern for your use case
- [ ] Implement efficient traversal (don't re-parse unnecessarily)
- [ ] Track source locations for error reporting
- [ ] Handle edge cases (empty files, single expressions)
- [ ] Support incremental parsing for large codebases

### Performance
- [ ] Profile AST operations for bottlenecks
- [ ] Consider lazy evaluation for large ASTs
- [ ] Implement caching strategy for unchanged files
- [ ] Use multiprocessing for analyzing multiple files
- [ ] Monitor memory usage with large codebases

### Error Handling
- [ ] Catch and handle SyntaxError appropriately
- [ ] Provide meaningful error messages with context
- [ ] Handle malformed ASTs from transformations
- [ ] Validate transformed ASTs before unparsing
- [ ] Log AST operations for debugging

### Integration
- [ ] Design clean API for AST operations
- [ ] Support multiple output formats (JSON, dict, custom)
- [ ] Provide utilities for common operations
- [ ] Document AST structure assumptions
- [ ] Version your AST schema if persisting

### Testing
- [ ] Test with various Python syntax features
- [ ] Include edge cases (decorators, comprehensions, walrus operator)
- [ ] Verify transformation correctness
- [ ] Test error handling paths
- [ ] Benchmark performance with realistic codebases

## Summary

Abstract Syntax Trees are the foundation of modern code analysis tools:

- **Structured Representation**: ASTs convert code from text to structured trees
- **Language Understanding**: Enable semantic analysis beyond text matching
- **Transformation Power**: Allow systematic code modifications
- **Tool Building**: Essential for IDEs, linters, formatters, and AI code tools
- **Production Ready**: Python\'s \`ast\` module is battle-tested and efficient

Understanding ASTs is crucial for building AI coding assistants like Cursor, as they provide the structural understanding needed for intelligent code suggestions, refactoring, and generation.

In the next sections, we'll dive deeper into Python-specific AST analysis, multi-language parsing with tree-sitter, and building complete code understanding engines.
`,
};

export default astFundamentals;
