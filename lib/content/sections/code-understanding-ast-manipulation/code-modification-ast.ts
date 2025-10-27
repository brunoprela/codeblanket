const codeModificationAst = {
  id: 'code-modification-ast',
  title: 'Code Modification with AST',
  content: `
# Code Modification with AST

## Introduction

Reading and analyzing code is valuable, but the real power comes from modifying it programmatically. This is how Cursor performs refactoring, code generation, and automated fixes. When you ask Cursor to "add error handling to all API calls" or "rename this variable everywhere," it's using AST modification.

**Why AST Modification Matters:**

Modern AI coding tools need to:
- Refactor code safely (rename, extract, inline)
- Generate new code that fits existing style
- Apply fixes automatically
- Transform code patterns
- Maintain syntactic correctness
- Preserve formatting when possible

This section teaches you to build these capabilities.

## Deep Technical Explanation

### AST Transformation Approaches

**1. NodeTransformer (Python\'s ast module):**
- Visits and modifies nodes in-place
- Returns modified or new nodes
- Can add, remove, or transform nodes
- Automatically updates parent references

**2. NodeVisitor + Manual Construction:**
- Read-only traversal
- Build new AST from scratch
- Full control over structure
- More verbose but explicit

**3. LibCST (Concrete Syntax Trees):**
- Preserves formatting and comments
- Lossless round-trip (code → CST → code)
- More complex but maintains style
- Best for production refactoring

### Transformation Patterns

**Replace:**
\`\`\`python
# Before
x = old_function()

# After (replacing function call)
x = new_function()
\`\`\`

**Insert:**
\`\`\`python
# Before
def process (data):
    return transform (data)

# After (inserting validation)
def process (data):
    validate (data)  # Inserted
    return transform (data)
\`\`\`

**Delete:**
\`\`\`python
# Before
def process (data):
    debug_print(data)  # Remove this
    return transform (data)

# After
def process (data):
    return transform (data)
\`\`\`

**Wrap:**
\`\`\`python
# Before
result = api_call()

# After (wrapping in try/except)
try:
    result = api_call()
except APIError:
    handle_error()
\`\`\`

## Code Implementation

### Basic AST Transformation

\`\`\`python
import ast
from typing import Any

class VariableRenamer (ast.NodeTransformer):
    """
    Rename variables throughout code.
    This is how refactoring tools implement "rename symbol".
    """
    
    def __init__(self, old_name: str, new_name: str):
        self.old_name = old_name
        self.new_name = new_name
        self.rename_count = 0
    
    def visit_Name (self, node: ast.Name) -> ast.Name:
        """Replace variable names."""
        if node.id == self.old_name:
            node.id = self.new_name
            self.rename_count += 1
        return node
    
    def visit_arg (self, node: ast.arg) -> ast.arg:
        """Rename function parameters."""
        if node.arg == self.old_name:
            node.arg = self.new_name
            self.rename_count += 1
        return node
    
    def visit_FunctionDef (self, node: ast.FunctionDef) -> ast.FunctionDef:
        """Rename function names."""
        if node.name == self.old_name:
            node.name = self.new_name
            self.rename_count += 1
        self.generic_visit (node)
        return node

# Example usage
code = """
def calculate (value):
    result = value * 2
    return result

total = calculate (value=10)
print(f"Result: {result}")
"""

tree = ast.parse (code)

# Rename 'value' to 'input_value'
renamer = VariableRenamer('value', 'input_value')
new_tree = renamer.visit (tree)

# Important: Fix missing locations after transformation
ast.fix_missing_locations (new_tree)

# Convert back to code
new_code = ast.unparse (new_tree)
print("=== Renamed Code ===")
print(new_code)
print(f"\\nRenamed {renamer.rename_count} occurrences")
\`\`\`

### Function Extraction

\`\`\`python
import ast
from typing import List, Set

class FunctionExtractor (ast.NodeTransformer):
    """
    Extract selected code into a new function.
    This implements the "Extract Method" refactoring.
    """
    
    def __init__(self, start_line: int, end_line: int, new_func_name: str):
        self.start_line = start_line
        self.end_line = end_line
        self.new_func_name = new_func_name
        self.extracted_stmts: List[ast.stmt] = []
        self.required_params: Set[str] = set()
        self.return_vars: Set[str] = set()
    
    def extract (self, tree: ast.Module) -> ast.Module:
        """Extract code and create new function."""
        # First pass: identify what to extract and analyze dependencies
        self._analyze_extraction (tree)
        
        # Create the new function
        new_func = self._create_function()
        
        # Second pass: replace extracted code with function call
        new_tree = self._replace_with_call (tree, new_func)
        
        return new_tree
    
    def _analyze_extraction (self, tree: ast.Module):
        """Analyze code to extract and determine dependencies."""
        for node in ast.walk (tree):
            if isinstance (node, ast.stmt):
                if hasattr (node, 'lineno'):
                    if self.start_line <= node.lineno <= self.end_line:
                        self.extracted_stmts.append (node)
        
        # Find variables used in extracted code
        used_vars = set()
        defined_vars = set()
        
        for stmt in self.extracted_stmts:
            for node in ast.walk (stmt):
                if isinstance (node, ast.Name):
                    if isinstance (node.ctx, ast.Load):
                        used_vars.add (node.id)
                    elif isinstance (node.ctx, ast.Store):
                        defined_vars.add (node.id)
        
        # Parameters are variables used but not defined in extracted code
        self.required_params = used_vars - defined_vars
        
        # Return values are variables defined in extracted code and used after
        # For simplicity, we'll return all defined variables
        self.return_vars = defined_vars
    
    def _create_function (self) -> ast.FunctionDef:
        """Create the new function with extracted code."""
        # Create parameters
        args = ast.arguments(
            posonlyargs=[],
            args=[ast.arg (arg=param, annotation=None) for param in sorted (self.required_params)],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[]
        )
        
        # Create function body
        body = self.extracted_stmts.copy()
        
        # Add return statement if needed
        if self.return_vars:
            if len (self.return_vars) == 1:
                return_value = ast.Name (id=list (self.return_vars)[0], ctx=ast.Load())
            else:
                return_value = ast.Tuple(
                    elts=[ast.Name (id=var, ctx=ast.Load()) for var in sorted (self.return_vars)],
                    ctx=ast.Load()
                )
            body.append (ast.Return (value=return_value))
        
        # Create function
        func = ast.FunctionDef(
            name=self.new_func_name,
            args=args,
            body=body,
            decorator_list=[],
            returns=None,
            lineno=self.start_line,
            col_offset=0
        )
        
        return func
    
    def _replace_with_call (self, tree: ast.Module, new_func: ast.FunctionDef) -> ast.Module:
        """Replace extracted code with function call."""
        new_body = []
        
        # Add the new function definition at the start
        new_body.append (new_func)
        
        extracted_lines = set (range (self.start_line, self.end_line + 1))
        replaced = False
        
        for stmt in tree.body:
            if hasattr (stmt, 'lineno') and stmt.lineno in extracted_lines:
                if not replaced:
                    # Replace first extracted statement with function call
                    call = ast.Call(
                        func=ast.Name (id=self.new_func_name, ctx=ast.Load()),
                        args=[ast.Name (id=param, ctx=ast.Load()) for param in sorted (self.required_params)],
                        keywords=[]
                    )
                    
                    if self.return_vars:
                        # Assign return value
                        if len (self.return_vars) == 1:
                            target = ast.Name (id=list (self.return_vars)[0], ctx=ast.Store())
                        else:
                            target = ast.Tuple(
                                elts=[ast.Name (id=var, ctx=ast.Store()) for var in sorted (self.return_vars)],
                                ctx=ast.Store()
                            )
                        call_stmt = ast.Assign (targets=[target], value=call)
                    else:
                        call_stmt = ast.Expr (value=call)
                    
                    new_body.append (call_stmt)
                    replaced = True
                # Skip other extracted statements
            else:
                new_body.append (stmt)
        
        new_tree = ast.Module (body=new_body, type_ignores=[])
        ast.fix_missing_locations (new_tree)
        return new_tree

# Example usage
code = """
def process_data (items):
    # Lines 3-6 will be extracted
    total = 0
    for item in items:
        if item > 0:
            total += item
    
    print(f"Total: {total}")
    return total
"""

tree = ast.parse (code)

# Extract lines 3-6 into new function
extractor = FunctionExtractor(
    start_line=3,
    end_line=6,
    new_func_name='calculate_sum'
)

new_tree = extractor.extract (tree)
new_code = ast.unparse (new_tree)

print("=== After Extraction ===")
print(new_code)
\`\`\`

### Adding Error Handling

\`\`\`python
import ast
from typing import List

class ErrorHandlingAdder (ast.NodeTransformer):
    """
    Wrap function calls in try/except blocks.
    This is how Cursor adds error handling automatically.
    """
    
    def __init__(self, functions_to_wrap: List[str]):
        self.functions_to_wrap = set (functions_to_wrap)
        self.wrap_count = 0
    
    def visit_Expr (self, node: ast.Expr) -> ast.stmt:
        """Wrap expression statements that are function calls."""
        if isinstance (node.value, ast.Call):
            if self._should_wrap (node.value):
                return self._wrap_in_try_except (node)
        return node
    
    def visit_Assign (self, node: ast.Assign) -> ast.stmt:
        """Wrap assignments from function calls."""
        if isinstance (node.value, ast.Call):
            if self._should_wrap (node.value):
                return self._wrap_in_try_except (node)
        return node
    
    def _should_wrap (self, call: ast.Call) -> bool:
        """Check if this call should be wrapped."""
        if isinstance (call.func, ast.Name):
            return call.func.id in self.functions_to_wrap
        elif isinstance (call.func, ast.Attribute):
            return call.func.attr in self.functions_to_wrap
        return False
    
    def _wrap_in_try_except (self, stmt: ast.stmt) -> ast.Try:
        """Wrap a statement in try/except."""
        self.wrap_count += 1
        
        # Create exception handler
        handler = ast.ExceptHandler(
            type=ast.Name (id='Exception', ctx=ast.Load()),
            name='e',
            body=[
                ast.Expr (value=ast.Call(
                    func=ast.Name (id='print', ctx=ast.Load()),
                    args=[
                        ast.JoinedStr (values=[
                            ast.Constant (value="Error: "),
                            ast.FormattedValue(
                                value=ast.Name (id='e', ctx=ast.Load()),
                                conversion=-1
                            )
                        ])
                    ],
                    keywords=[]
                ))
            ]
        )
        
        # Create try block
        try_node = ast.Try(
            body=[stmt],
            handlers=[handler],
            orelse=[],
            finalbody=[]
        )
        
        return try_node

# Example usage
code = """
def main():
    data = fetch_data()
    result = process (data)
    save_result (result)
    
    normal_operation()
"""

tree = ast.parse (code)

# Add error handling to specific functions
adder = ErrorHandlingAdder(['fetch_data', 'process', 'save_result'])
new_tree = adder.visit (tree)
ast.fix_missing_locations (new_tree)

new_code = ast.unparse (new_tree)
print("=== With Error Handling ===")
print(new_code)
print(f"\\nAdded error handling to {adder.wrap_count} calls")
\`\`\`

### Code Style Normalizer

\`\`\`python
import ast

class StyleNormalizer (ast.NodeTransformer):
    """
    Normalize code style to match conventions.
    Makes generated code consistent with existing style.
    """
    
    def __init__(self):
        self.changes = []
    
    def visit_FunctionDef (self, node: ast.FunctionDef) -> ast.FunctionDef:
        """Ensure function names are snake_case."""
        original_name = node.name
        
        # Convert to snake_case if needed
        new_name = self._to_snake_case (original_name)
        
        if new_name != original_name:
            self.changes.append (f"Function '{original_name}' → '{new_name}'")
            node.name = new_name
        
        self.generic_visit (node)
        return node
    
    def visit_ClassDef (self, node: ast.ClassDef) -> ast.ClassDef:
        """Ensure class names are PascalCase."""
        original_name = node.name
        
        # Convert to PascalCase if needed
        new_name = self._to_pascal_case (original_name)
        
        if new_name != original_name:
            self.changes.append (f"Class '{original_name}' → '{new_name}'")
            node.name = new_name
        
        self.generic_visit (node)
        return node
    
    def visit_Name (self, node: ast.Name) -> ast.Name:
        """Ensure constant names are UPPER_CASE."""
        # Only process if it's a global assignment target
        if isinstance (node.ctx, ast.Store):
            if node.id.isupper() or '_' in node.id:
                # Likely a constant, ensure UPPER_CASE
                original = node.id
                new_name = node.id.upper()
                if new_name != original:
                    self.changes.append (f"Constant '{original}' → '{new_name}'")
                    node.id = new_name
        
        return node
    
    def _to_snake_case (self, name: str) -> str:
        """Convert to snake_case."""
        import re
        # Insert underscore before capitals
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\\1_\\2', name)
        # Insert underscore before capital followed by lowercase
        s2 = re.sub('([a-z0-9])([A-Z])', r'\\1_\\2', s1)
        return s2.lower()
    
    def _to_pascal_case (self, name: str) -> str:
        """Convert to PascalCase."""
        words = name.replace('_', ' ').replace('-', ' ').split()
        return '.join (word.capitalize() for word in words)

# Example usage
code = """
class user_service:
    def GetUser (self, userID):
        max_retries = 3
        return self.FetchData (userID)
    
    def FetchData (self, id):
        return database.get (id)

def ProcessData (input_data):
    return transform (input_data)

MAX_SIZE = 100
tempValue = 50
"""

tree = ast.parse (code)

normalizer = StyleNormalizer()
new_tree = normalizer.visit (tree)
ast.fix_missing_locations (new_tree)

new_code = ast.unparse (new_tree)

print("=== Normalized Code ===")
print(new_code)

if normalizer.changes:
    print("\\n=== Changes Made ===")
    for change in normalizer.changes:
        print(f"  {change}")
\`\`\`

### Preserving Formatting with LibCST

\`\`\`python
# LibCST example (requires: pip install libcst)
import libcst as cst

class PreservingRenamer (cst.CSTTransformer):
    """
    Rename variables while preserving formatting and comments.
    This is production-quality refactoring.
    """
    
    def __init__(self, old_name: str, new_name: str):
        self.old_name = old_name
        self.new_name = new_name
    
    def leave_Name (self, original_node: cst.Name, updated_node: cst.Name) -> cst.Name:
        """Rename Name nodes."""
        if updated_node.value == self.old_name:
            return updated_node.with_changes (value=self.new_name)
        return updated_node
    
    def leave_Param (self, original_node: cst.Param, updated_node: cst.Param) -> cst.Param:
        """Rename function parameters."""
        if isinstance (updated_node.name, cst.Name):
            if updated_node.name.value == self.old_name:
                return updated_node.with_changes(
                    name=cst.Name (self.new_name)
                )
        return updated_node

# Example with formatting preservation
code = """
def calculate (value):  # Calculate result
    result = value * 2  # Double the value
    
    # Return the result
    return result
"""

# Parse with LibCST
module = cst.parse_module (code)

# Transform
renamer = PreservingRenamer('value', 'input_value')
new_module = module.visit (renamer)

# Code is identical except for renamed variable
# Comments and formatting are preserved!
print("=== Formatting Preserved ===")
print(new_module.code)
\`\`\`

### Complete Refactoring Tool

\`\`\`python
import ast
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

@dataclass
class RefactoringOperation:
    type: str  # 'rename', 'extract', 'inline', 'add_error_handling'
    target: str
    new_value: Optional[str] = None
    metadata: Dict[str, Any] = None

class CodeRefactorer:
    """
    Complete refactoring system.
    Supports multiple refactoring operations.
    """
    
    def __init__(self, code: str):
        self.original_code = code
        self.tree = ast.parse (code)
        self.operations: List[RefactoringOperation] = []
    
    def rename_symbol (self, old_name: str, new_name: str):
        """Add rename operation."""
        self.operations.append(RefactoringOperation(
            type='rename',
            target=old_name,
            new_value=new_name
        ))
        return self
    
    def extract_function (self, start_line: int, end_line: int, func_name: str):
        """Add extract function operation."""
        self.operations.append(RefactoringOperation(
            type='extract',
            target=f"{start_line}-{end_line}",
            new_value=func_name,
            metadata={'start_line': start_line, 'end_line': end_line}
        ))
        return self
    
    def add_error_handling (self, function_names: List[str]):
        """Add error handling operation."""
        self.operations.append(RefactoringOperation(
            type='add_error_handling',
            target=','.join (function_names),
            metadata={'functions': function_names}
        ))
        return self
    
    def apply (self) -> str:
        """Apply all refactoring operations."""
        current_tree = self.tree
        
        for operation in self.operations:
            if operation.type == 'rename':
                transformer = VariableRenamer (operation.target, operation.new_value)
                current_tree = transformer.visit (current_tree)
                ast.fix_missing_locations (current_tree)
            
            elif operation.type == 'extract':
                extractor = FunctionExtractor(
                    operation.metadata['start_line'],
                    operation.metadata['end_line'],
                    operation.new_value
                )
                current_tree = extractor.extract (current_tree)
            
            elif operation.type == 'add_error_handling':
                adder = ErrorHandlingAdder (operation.metadata['functions'])
                current_tree = adder.visit (current_tree)
                ast.fix_missing_locations (current_tree)
        
        return ast.unparse (current_tree)
    
    def preview (self) -> str:
        """Preview operations without applying."""
        lines = ["Planned Refactoring Operations:\\n"]
        for i, op in enumerate (self.operations, 1):
            lines.append (f"{i}. {op.type.replace('_', ' ').title()}")
            lines.append (f"   Target: {op.target}")
            if op.new_value:
                lines.append (f"   New value: {op.new_value}")
        return "\\n".join (lines)

# Example usage
code = """
def process (data):
    items = data.split(',')
    total = 0
    for item in items:
        total += int (item)
    result = total
    return result
"""

# Chain multiple refactoring operations
refactorer = CodeRefactorer (code)
refactorer.rename_symbol('data', 'input_str').rename_symbol('result', 'final_result')

print("=== Preview ===")
print(refactorer.preview())

print("\\n=== Refactored Code ===")
refactored = refactorer.apply()
print(refactored)
\`\`\`

## Real-World Case Study: How Cursor Uses AST Modification

Cursor performs sophisticated AST modifications:

**1. Auto-Fix Suggestions:**
\`\`\`python
# Cursor detects unused import
import os  # Not used

# Offers quick fix: "Remove unused import"
# Internally: AST modification removes Import node
\`\`\`

**2. Refactoring:**
\`\`\`python
# User selects code and chooses "Extract Method"
selected = """
    total = 0
    for item in items:
        total += item
"""

# Cursor:
# 1. Analyzes selected AST nodes
# 2. Determines parameters needed
# 3. Creates new function AST
# 4. Replaces selected code with call
# 5. Preserves formatting with LibCST
\`\`\`

**3. Code Generation:**
\`\`\`python
# User: "Add a method to save to database"
# Cursor generates new AST nodes
new_method = ast.FunctionDef(
    name='save_to_db',
    args=...,
    body=[...]
)

# Inserts into class AST
# Unparsed to code with proper indentation
\`\`\`

**4. Pattern Application:**
\`\`\`python
# User: "Add logging to all API calls"
# Cursor:
# 1. Finds all API call nodes
# 2. Wraps each in logging context
# 3. Maintains code structure
# 4. Applies consistently
\`\`\`

## Hands-On Exercise

Build a production refactoring engine:

\`\`\`python
class ProductionRefactoringEngine:
    """
    Production-grade refactoring with validation and rollback.
    """
    
    def __init__(self, code: str):
        self.original_code = code
        self.current_code = code
        self.history: List[str] = [code]
        self.operations_applied: List[str] = []
    
    def rename_variable (self, old_name: str, new_name: str) -> bool:
        """Safely rename a variable."""
        try:
            tree = ast.parse (self.current_code)
            
            # Validate names
            if not self._is_valid_identifier (new_name):
                raise ValueError (f"Invalid identifier: {new_name}")
            
            # Apply transformation
            renamer = VariableRenamer (old_name, new_name)
            new_tree = renamer.visit (tree)
            ast.fix_missing_locations (new_tree)
            
            # Validate result
            if not self._validate_syntax (new_tree):
                raise SyntaxError("Transformation produced invalid syntax")
            
            # Apply change
            self.current_code = ast.unparse (new_tree)
            self.history.append (self.current_code)
            self.operations_applied.append (f"Renamed '{old_name}' to '{new_name}'")
            
            return True
        
        except Exception as e:
            print(f"Refactoring failed: {e}")
            return False
    
    def _is_valid_identifier (self, name: str) -> bool:
        """Check if name is a valid Python identifier."""
        return name.isidentifier() and not __import__('keyword').iskeyword (name)
    
    def _validate_syntax (self, tree: ast.AST) -> bool:
        """Validate that AST is syntactically correct."""
        try:
            compile (tree, '<string>', 'exec')
            return True
        except:
            return False
    
    def undo (self) -> bool:
        """Undo last operation."""
        if len (self.history) > 1:
            self.history.pop()
            self.current_code = self.history[-1]
            if self.operations_applied:
                self.operations_applied.pop()
            return True
        return False
    
    def get_diff (self) -> str:
        """Show differences from original."""
        import difflib
        diff = difflib.unified_diff(
            self.original_code.splitlines (keepends=True),
            self.current_code.splitlines (keepends=True),
            fromfile='original',
            tofile='refactored'
        )
        return '.join (diff)
    
    def get_current_code (self) -> str:
        """Get current code state."""
        return self.current_code
    
    def get_history (self) -> List[str]:
        """Get operation history."""
        return self.operations_applied.copy()

# Test the engine
code = """
def calculate (value):
    result = value * 2
    return result

output = calculate(10)
"""

engine = ProductionRefactoringEngine (code)

# Apply refactorings
engine.rename_variable('value', 'input_value')
engine.rename_variable('result', 'output_value')

print("=== Refactored Code ===")
print(engine.get_current_code())

print("\\n=== Operations Applied ===")
for op in engine.get_history():
    print(f"  - {op}")

print("\\n=== Diff ===")
print(engine.get_diff())
\`\`\`

**Exercise Tasks:**1. Add validation to ensure refactoring doesn't break code
2. Implement undo/redo functionality
3. Add preview mode showing changes before applying
4. Preserve comments and docstrings
5. Handle scope-aware renaming (don't rename shadowed variables incorrectly)

## Common Pitfalls

### 1. Not Fixing Locations

\`\`\`python
# ❌ Wrong: Missing locations cause issues
new_tree = transformer.visit (tree)
code = ast.unparse (new_tree)  # May have wrong line numbers

# ✅ Correct: Always fix locations
new_tree = transformer.visit (tree)
ast.fix_missing_locations (new_tree)
code = ast.unparse (new_tree)
\`\`\`

### 2. Forgetting to Return Nodes

\`\`\`python
# ❌ Wrong: Doesn't return node
class MyTransformer (ast.NodeTransformer):
    def visit_Name (self, node):
        node.id = "new_name"
        # Forgot to return!

# ✅ Correct: Always return the node
class MyTransformer (ast.NodeTransformer):
    def visit_Name (self, node):
        node.id = "new_name"
        return node  # Important!
\`\`\`

### 3. Not Calling generic_visit

\`\`\`python
# ❌ Wrong: Won't visit children
class MyTransformer (ast.NodeTransformer):
    def visit_FunctionDef (self, node):
        print(node.name)
        return node  # Children not visited!

# ✅ Correct: Visit children
class MyTransformer (ast.NodeTransformer):
    def visit_FunctionDef (self, node):
        print(node.name)
        self.generic_visit (node)  # Visit children
        return node
\`\`\`

### 4. Losing Type Information

\`\`\`python
# ❌ Wrong: Loses annotations
def transform_function (node):
    return ast.FunctionDef(
        name=node.name,
        args=node.args,
        body=node.body
    )  # Lost returns, decorators, etc.

# ✅ Correct: Preserve all attributes
def transform_function (node):
    return ast.FunctionDef(
        name=node.name,
        args=node.args,
        body=node.body,
        decorator_list=node.decorator_list,
        returns=node.returns,
        type_comment=node.type_comment
    )
\`\`\`

## Production Checklist

### Transformation Safety
- [ ] Validate AST before and after transformation
- [ ] Test with edge cases
- [ ] Handle errors gracefully
- [ ] Provide rollback capability
- [ ] Verify semantic correctness

### Code Quality
- [ ] Fix missing locations after modifications
- [ ] Preserve formatting where possible (use LibCST)
- [ ] Maintain code style consistency
- [ ] Keep comments and docstrings
- [ ] Handle indentation correctly

### Performance
- [ ] Minimize tree traversals
- [ ] Cache parsed ASTs
- [ ] Batch multiple transformations
- [ ] Profile transformation time
- [ ] Optimize for large files

### User Experience
- [ ] Show preview before applying
- [ ] Provide clear error messages
- [ ] Enable undo/redo
- [ ] Track what changed
- [ ] Support partial application

### Testing
- [ ] Unit test each transformation
- [ ] Test with real codebases
- [ ] Verify no syntax errors
- [ ] Check semantic preservation
- [ ] Validate edge cases

## Summary

AST modification enables powerful code transformations:

- **Refactoring**: Rename, extract, inline safely
- **Code Generation**: Create syntactically correct code
- **Pattern Application**: Apply changes consistently
- **Auto-Fixes**: Correct issues automatically
- **Style Normalization**: Ensure consistency

These capabilities are essential for AI coding tools like Cursor to provide intelligent refactoring, code generation, and automated fixes while maintaining code correctness and style.

In the next section, we'll explore static analysis and code quality—detecting issues and patterns in code systematically.
`,
};

export default codeModificationAst;
