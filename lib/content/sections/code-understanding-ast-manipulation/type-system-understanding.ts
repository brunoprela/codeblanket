const typeSystemUnderstanding = {
  id: 'type-system-understanding',
  title: 'Type System Understanding',
  content: `
# Type System Understanding

## Introduction

Python's type hints system provides valuable information for code analysis. When Cursor suggests \`user.email: str\` instead of just \`user.email\`, it's using type information. Understanding types enables better auto-completion, error detection, and code generation.

**Why Type Understanding Matters:**

For AI coding tools like Cursor to be maximally helpful, they need to:
- Parse and understand type annotations
- Infer types even without annotations
- Validate type usage
- Suggest type-aware completions
- Generate type-correct code
- Detect type-related bugs

This section teaches you to build these capabilities.

## Deep Technical Explanation

### Python Type System Levels

**1. Runtime Types:**
\`\`\`python
x = 5  # Runtime type: int
isinstance(x, int)  # True
\`\`\`

**2. Type Hints (Annotations):**
\`\`\`python
def add(x: int, y: int) -> int:
    return x + y
\`\`\`

**3. Type Checking (mypy, pyright):**
- Static analysis tools
- Verify type consistency
- Catch type errors before runtime

### Type Annotation AST Nodes

\`\`\`python
# Function with type hints
def process(data: List[str]) -> Dict[str, int]:
    pass

# AST representation:
FunctionDef(
    name='process',
    args=arguments(
        args=[
            arg(
                arg='data',
                annotation=Subscript(  # List[str]
                    value=Name(id='List'),
                    slice=Name(id='str')
                )
            )
        ]
    ),
    returns=Subscript(  # Dict[str, int]
        value=Name(id='Dict'),
        slice=Tuple(elts=[Name(id='str'), Name(id='int')])
    )
)
\`\`\`

### Generic Types and Complex Annotations

\`\`\`python
from typing import Optional, Union, List, Dict, Callable

# Optional (Union with None)
def get_user(id: int) -> Optional[User]:
    pass

# Union types
def process(data: Union[str, bytes]) -> int:
    pass

# Nested generics
def transform(items: List[Dict[str, Any]]) -> Dict[str, List[int]]:
    pass

# Callable types
def apply(func: Callable[[int, int], int], x: int, y: int) -> int:
    pass
\`\`\`

## Code Implementation

### Type Annotation Extractor

\`\`\`python
import ast
from dataclasses import dataclass
from typing import Optional, List, Dict

@dataclass
class TypeInfo:
    name: str
    type_annotation: str
    line: int
    kind: str  # 'parameter', 'return', 'variable'

class TypeAnnotationExtractor(ast.NodeVisitor):
    """
    Extract all type annotations from code.
    This builds a type map for the codebase.
    """
    
    def __init__(self):
        self.types: Dict[str, TypeInfo] = {}
        self.current_function: Optional[str] = None
    
    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Extract function type annotations."""
        func_name = node.name
        old_func = self.current_function
        self.current_function = func_name
        
        # Extract parameter types
        for arg in node.args.args:
            if arg.annotation:
                type_str = ast.unparse(arg.annotation)
                key = f"{func_name}.{arg.arg}"
                self.types[key] = TypeInfo(
                    name=arg.arg,
                    type_annotation=type_str,
                    line=node.lineno,
                    kind='parameter'
                )
        
        # Extract return type
        if node.returns:
            return_type = ast.unparse(node.returns)
            key = f"{func_name}.return"
            self.types[key] = TypeInfo(
                name=func_name,
                type_annotation=return_type,
                line=node.lineno,
                kind='return'
            )
        
        self.generic_visit(node)
        self.current_function = old_func
    
    def visit_AnnAssign(self, node: ast.AnnAssign):
        """Extract variable annotations."""
        if isinstance(node.target, ast.Name):
            var_name = node.target.id
            type_str = ast.unparse(node.annotation)
            
            scope = self.current_function or '<module>'
            key = f"{scope}.{var_name}"
            
            self.types[key] = TypeInfo(
                name=var_name,
                type_annotation=type_str,
                line=node.lineno,
                kind='variable'
            )
        
        self.generic_visit(node)
    
    def get_type_for(self, name: str, scope: Optional[str] = None) -> Optional[str]:
        """Get type annotation for a symbol."""
        if scope:
            key = f"{scope}.{name}"
            if key in self.types:
                return self.types[key].type_annotation
        
        # Try without scope
        for key, type_info in self.types.items():
            if key.endswith(f".{name}"):
                return type_info.type_annotation
        
        return None
    
    def visualize_types(self) -> str:
        """Create visualization of type annotations."""
        lines = ["=== Type Annotations ===\\n"]
        
        # Group by kind
        by_kind = {}
        for type_info in self.types.values():
            by_kind.setdefault(type_info.kind, []).append(type_info)
        
        for kind, infos in sorted(by_kind.items()):
            lines.append(f"{kind.title()}s:")
            for info in sorted(infos, key=lambda i: i.line):
                lines.append(f"  {info.name}: {info.type_annotation} (line {info.line})")
            lines.append("")
        
        return "\\n".join(lines)

# Example usage
code = """
from typing import List, Dict, Optional

def process_users(users: List[Dict[str, str]], strict: bool = False) -> Optional[List[str]]:
    names: List[str] = []
    
    for user in users:
        name: str = user.get('name', ')
        if name:
            names.append(name)
    
    return names if names else None

class UserService:
    def __init__(self, db_url: str):
        self.db_url: str = db_url
        self.cache: Dict[int, str] = {}
    
    def get_user(self, user_id: int) -> Optional[str]:
        return self.cache.get(user_id)
"""

extractor = TypeAnnotationExtractor()
tree = ast.parse(code)
extractor.visit(tree)

print(extractor.visualize_types())

# Query specific types
print("\\n=== Type Queries ===")
user_id_type = extractor.get_type_for('user_id', 'get_user')
print(f"user_id parameter type: {user_id_type}")

return_type = extractor.get_type_for('return', 'process_users')
print(f"process_users return type: {return_type}")
\`\`\`

### Type Inference Engine

\`\`\`python
import ast
from typing import Dict, Optional, Set

class TypeInferenceEngine(ast.NodeVisitor):
    """
    Infer types even without annotations.
    This is how IDEs provide type information for untyped code.
    """
    
    def __init__(self):
        self.inferred_types: Dict[str, str] = {}
        self.current_scope: str = '<module>'
    
    def visit_Assign(self, node: ast.Assign):
        """Infer types from assignments."""
        # Infer from literal values
        if isinstance(node.value, ast.Constant):
            type_name = type(node.value.value).__name__
            for target in node.targets:
                if isinstance(target, ast.Name):
                    key = f"{self.current_scope}.{target.id}"
                    self.inferred_types[key] = type_name
        
        # Infer from list/dict/set literals
        elif isinstance(node.value, ast.List):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    key = f"{self.current_scope}.{target.id}"
                    # Try to infer element type
                    if node.value.elts:
                        elem_type = self._infer_from_node(node.value.elts[0])
                        self.inferred_types[key] = f"list[{elem_type}]"
                    else:
                        self.inferred_types[key] = "list"
        
        elif isinstance(node.value, ast.Dict):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    key = f"{self.current_scope}.{target.id}"
                    self.inferred_types[key] = "dict"
        
        # Infer from function calls
        elif isinstance(node.value, ast.Call):
            call_type = self._infer_return_type(node.value)
            if call_type:
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        key = f"{self.current_scope}.{target.id}"
                        self.inferred_types[key] = call_type
        
        self.generic_visit(node)
    
    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Enter function scope."""
        old_scope = self.current_scope
        self.current_scope = node.name
        
        # Infer parameter types if not annotated
        for arg in node.args.args:
            if not arg.annotation:
                # Look for usage patterns to infer type
                inferred = self._infer_param_type(node, arg.arg)
                if inferred:
                    key = f"{node.name}.{arg.arg}"
                    self.inferred_types[key] = inferred
        
        self.generic_visit(node)
        self.current_scope = old_scope
    
    def _infer_from_node(self, node: ast.AST) -> str:
        """Infer type from AST node."""
        if isinstance(node, ast.Constant):
            return type(node.value).__name__
        elif isinstance(node, ast.List):
            return "list"
        elif isinstance(node, ast.Dict):
            return "dict"
        elif isinstance(node, ast.Set):
            return "set"
        elif isinstance(node, ast.Tuple):
            return "tuple"
        elif isinstance(node, ast.Name):
            # Look up in inferred types
            key = f"{self.current_scope}.{node.id}"
            return self.inferred_types.get(key, "Any")
        return "Any"
    
    def _infer_return_type(self, call: ast.Call) -> Optional[str]:
        """Infer return type of function call."""
        # Known built-in functions
        builtins = {
            'len': 'int',
            'str': 'str',
            'int': 'int',
            'float': 'float',
            'list': 'list',
            'dict': 'dict',
            'set': 'set',
            'tuple': 'tuple',
            'open': 'TextIOWrapper',
            'range': 'range',
        }
        
        if isinstance(call.func, ast.Name):
            return builtins.get(call.func.id)
        
        return None
    
    def _infer_param_type(self, func: ast.FunctionDef, param_name: str) -> Optional[str]:
        """Infer parameter type from usage."""
        # Look for operations that reveal type
        for node in ast.walk(func):
            if isinstance(node, ast.BinOp):
                # Check if parameter is used in arithmetic
                if isinstance(node.left, ast.Name) and node.left.id == param_name:
                    if isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
                        return "int | float"  # Numeric
            
            elif isinstance(node, ast.Subscript):
                # Used as sequence/mapping
                if isinstance(node.value, ast.Name) and node.value.id == param_name:
                    return "list | dict"  # Subscriptable
            
            elif isinstance(node, ast.Compare):
                # String operations
                if isinstance(node.left, ast.Name) and node.left.id == param_name:
                    for op in node.ops:
                        if isinstance(op, ast.In):
                            return "str | list"  # Iterable
        
        return None
    
    def get_type_summary(self) -> str:
        """Generate summary of inferred types."""
        lines = ["=== Inferred Types ===\\n"]
        
        for key, type_str in sorted(self.inferred_types.items()):
            lines.append(f"{key}: {type_str}")
        
        return "\\n".join(lines)

# Example usage
code = """
def calculate(x, y):
    # Infer from usage: x and y are numeric
    result = x + y
    total = result * 2
    return total

def process_items(items):
    # Infer from usage: items is subscriptable
    results = []
    for item in items:
        if item in cache:
            results.append(item)
    return results

# Simple assignments
name = "Alice"
age = 30
scores = [95, 87, 92]
user_data = {"name": "Bob", "age": 25}
"""

inferencer = TypeInferenceEngine()
tree = ast.parse(code)
inferencer.visit(tree)

print(inferencer.get_type_summary())
\`\`\`

### Type Checker

\`\`\`python
import ast
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class TypeError:
    line: int
    message: str
    expected: str
    actual: str

class SimpleTypeChecker(ast.NodeVisitor):
    """
    Basic type checking for Python code.
    Detects type mismatches and invalid operations.
    """
    
    def __init__(self, type_annotations: Dict[str, TypeInfo]):
        self.annotations = type_annotations
        self.errors: List[TypeError] = []
        self.variable_types: Dict[str, str] = {}
    
    def visit_Assign(self, node: ast.Assign):
        """Check assignment type consistency."""
        # Get inferred type of value
        value_type = self._get_type(node.value)
        
        # Check against declared types
        for target in node.targets:
            if isinstance(target, ast.Name):
                var_name = target.id
                
                # Store type for later checks
                self.variable_types[var_name] = value_type
        
        self.generic_visit(node)
    
    def visit_Call(self, node: ast.Call):
        """Check function call type consistency."""
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            
            # Check argument types against parameter types
            for i, arg in enumerate(node.args):
                param_key = f"{func_name}.param{i}"
                if param_key in self.annotations:
                    expected_type = self.annotations[param_key].type_annotation
                    actual_type = self._get_type(arg)
                    
                    if not self._is_compatible(expected_type, actual_type):
                        self.errors.append(TypeError(
                            line=node.lineno,
                            message=f"Argument {i} type mismatch in call to '{func_name}'",
                            expected=expected_type,
                            actual=actual_type
                        ))
        
        self.generic_visit(node)
    
    def visit_Return(self, node: ast.Return):
        """Check return type matches function signature."""
        # Would need function context to check return type
        self.generic_visit(node)
    
    def _get_type(self, node: ast.AST) -> str:
        """Get type of an expression."""
        if isinstance(node, ast.Constant):
            return type(node.value).__name__
        
        elif isinstance(node, ast.Name):
            return self.variable_types.get(node.id, "Any")
        
        elif isinstance(node, ast.List):
            if node.elts:
                elem_type = self._get_type(node.elts[0])
                return f"list[{elem_type}]"
            return "list"
        
        elif isinstance(node, ast.Dict):
            return "dict"
        
        elif isinstance(node, ast.BinOp):
            left_type = self._get_type(node.left)
            right_type = self._get_type(node.right)
            
            if left_type == right_type == "int":
                return "int"
            elif left_type in ["int", "float"] and right_type in ["int", "float"]:
                return "float"
            
            return "Any"
        
        return "Any"
    
    def _is_compatible(self, expected: str, actual: str) -> bool:
        """Check if types are compatible."""
        if expected == actual:
            return True
        
        # Any is compatible with everything
        if expected == "Any" or actual == "Any":
            return True
        
        # int is compatible with float
        if expected == "float" and actual == "int":
            return True
        
        # Optional[T] accepts None
        if expected.startswith("Optional["):
            if actual == "None":
                return True
            inner_type = expected[9:-1]  # Extract T from Optional[T]
            return self._is_compatible(inner_type, actual)
        
        return False
    
    def generate_report(self) -> str:
        """Generate type error report."""
        if not self.errors:
            return "✅ No type errors found!"
        
        lines = [f"=== Found {len(self.errors)} Type Errors ===\\n"]
        
        for error in sorted(self.errors, key=lambda e: e.line):
            lines.append(f"Line {error.line}: {error.message}")
            lines.append(f"  Expected: {error.expected}")
            lines.append(f"  Actual: {error.actual}")
            lines.append("")
        
        return "\\n".join(lines)

# Example usage
code = """
def add(x: int, y: int) -> int:
    return x + y

def process(data: str) -> int:
    return len(data)

# Type errors
result1 = add("hello", "world")  # Error: strings instead of ints
result2 = process(123)  # Error: int instead of str

# Correct usage
result3 = add(5, 10)  # OK
result4 = process("test")  # OK
"""

# First extract types
extractor = TypeAnnotationExtractor()
tree = ast.parse(code)
extractor.visit(tree)

# Then check types
checker = SimpleTypeChecker(extractor.types)
checker.visit(tree)

print(checker.generate_report())
\`\`\`

### Type-Aware Code Completion

\`\`\`python
import ast
from typing import List, Dict

class TypeAwareCompleter:
    """
    Provide type-aware code completions.
    This is how Cursor knows what attributes/methods are available.
    """
    
    def __init__(self, code: str):
        self.tree = ast.parse(code)
        self.type_extractor = TypeAnnotationExtractor()
        self.type_extractor.visit(self.tree)
    
    def get_completions(self, variable_name: str, scope: str = None) -> List[str]:
        """
        Get possible completions for a variable.
        
        Args:
            variable_name: Variable to complete
            scope: Function/class scope
            
        Returns:
            List of possible completions
        """
        # Get type of variable
        var_type = self.type_extractor.get_type_for(variable_name, scope)
        
        if not var_type:
            return []
        
        # Return attributes/methods based on type
        completions = self._get_completions_for_type(var_type)
        return completions
    
    def _get_completions_for_type(self, type_str: str) -> List[str]:
        """Get completions for a type."""
        # Built-in type completions
        type_methods = {
            'str': [
                'upper', 'lower', 'strip', 'split', 'join', 'replace',
                'startswith', 'endswith', 'find', 'format', 'capitalize'
            ],
            'list': [
                'append', 'extend', 'insert', 'remove', 'pop', 'clear',
                'index', 'count', 'sort', 'reverse', 'copy'
            ],
            'dict': [
                'get', 'keys', 'values', 'items', 'pop', 'update',
                'clear', 'setdefault', 'popitem', 'copy'
            ],
            'int': [
                'bit_length', 'to_bytes', 'from_bytes'
            ],
            'float': [
                'is_integer', 'hex', 'fromhex'
            ],
        }
        
        # Handle generic types like List[str], Dict[str, int]
        if '[' in type_str:
            base_type = type_str.split('[')[0].lower()
            return type_methods.get(base_type, [])
        
        return type_methods.get(type_str.lower(), [])
    
    def suggest_function_call(self, func_name: str) -> Dict:
        """
        Suggest how to call a function with types.
        
        Returns:
            Dict with parameter names and types
        """
        # Find function in AST
        for node in ast.walk(self.tree):
            if isinstance(node, ast.FunctionDef) and node.name == func_name:
                params = []
                
                for arg in node.args.args:
                    param_info = {
                        'name': arg.arg,
                        'type': None,
                        'default': None
                    }
                    
                    # Get type annotation
                    if arg.annotation:
                        param_info['type'] = ast.unparse(arg.annotation)
                    
                    params.append(param_info)
                
                # Get defaults
                defaults = node.args.defaults
                if defaults:
                    # Defaults align with last N parameters
                    offset = len(params) - len(defaults)
                    for i, default in enumerate(defaults):
                        params[offset + i]['default'] = ast.unparse(default)
                
                # Get return type
                return_type = None
                if node.returns:
                    return_type = ast.unparse(node.returns)
                
                return {
                    'name': func_name,
                    'parameters': params,
                    'returns': return_type
                }
        
        return {}

# Example usage
code = """
from typing import List, Dict

def process_users(users: List[Dict[str, str]], strict: bool = False) -> List[str]:
    ''Process user data.''
    names: List[str] = []
    
    for user in users:
        name: str = user.get('name', ')
        if name:
            names.append(name)
    
    return names

def main():
    data = "Hello World"
    items = [1, 2, 3, 4, 5]
    config = {"key": "value"}
"""

completer = TypeAwareCompleter(code)

# Test completions
print("=== Completions for 'data' (str) ===")
completions = completer.get_completions('data', 'main')
print("  ", ", ".join(completions[:5]))

print("\\n=== Completions for 'items' (list) ===")
completions = completer.get_completions('items', 'main')
print("  ", ", ".join(completions[:5]))

print("\\n=== Completions for 'config' (dict) ===")
completions = completer.get_completions('config', 'main')
print("  ", ", ".join(completions[:5]))

# Function call suggestion
print("\\n=== Function Call Suggestion ===")
suggestion = completer.suggest_function_call('process_users')
if suggestion:
    params_str = ", ".join(
        f"{p['name']}: {p['type']}" + (f" = {p['default']}" if p['default'] else "")
        for p in suggestion['parameters']
    )
    print(f"def {suggestion['name']}({params_str}) -> {suggestion['returns']}")
\`\`\`

## Real-World Case Study: How Cursor Uses Type Information

Cursor leverages type understanding extensively:

**1. Smart Auto-Completion:**
\`\`\`python
def process(data: pd.DataFrame):
    # When you type 'data.' Cursor shows DataFrame methods
    data.  # <-- Shows: head(), tail(), describe(), etc.
\`\`\`

**2. Type-Based Suggestions:**
\`\`\`python
def calculate(x: int, y: int) -> int:
    # When generating return, Cursor knows it must be int
    return  # <-- Suggests int operations, not string operations
\`\`\`

**3. Error Prevention:**
\`\`\`python
def get_user(user_id: int) -> Optional[User]:
    pass

result = get_user("123")  # Cursor warns: expected int, got str
\`\`\`

**4. Type Inference:**
\`\`\`python
items = [1, 2, 3]  # Cursor infers: List[int]
# When you type 'items.' it shows list methods
# When you do 'items[0].' it shows int methods
\`\`\`

## Common Pitfalls

### 1. Not Handling Optional

\`\`\`python
# ❌ Wrong: Doesn't check for None
def get_name(user: Optional[User]) -> str:
    return user.name  # Type error if user is None!

# ✅ Correct: Check for None
def get_name(user: Optional[User]) -> str:
    if user is None:
        return ""
    return user.name
\`\`\`

### 2. Ignoring Generic Type Parameters

\`\`\`python
# ❌ Wrong: Loses element type info
def process(items: list):  # What type are elements?
    return items[0]

# ✅ Correct: Specify element type
def process(items: List[int]) -> int:
    return items[0]
\`\`\`

### 3. Over-Strict Type Checking

\`\`\`python
# ❌ Wrong: Too strict
def add(x: int, y: int) -> int:
    return x + y

# Can't call with floats even though it works

# ✅ Better: Use Union or Protocol
def add(x: Union[int, float], y: Union[int, float]) -> Union[int, float]:
    return x + y
\`\`\`

## Production Checklist

### Type Extraction
- [ ] Parse function annotations
- [ ] Extract variable annotations
- [ ] Handle generic types
- [ ] Support Union/Optional
- [ ] Track type aliases

### Type Inference
- [ ] Infer from literals
- [ ] Infer from operations
- [ ] Infer from built-ins
- [ ] Handle comprehensions
- [ ] Track through assignments

### Type Checking
- [ ] Validate function calls
- [ ] Check return types
- [ ] Verify operations
- [ ] Handle type narrowing
- [ ] Support gradual typing

### Completions
- [ ] Type-aware suggestions
- [ ] Method completions
- [ ] Parameter hints
- [ ] Return type display
- [ ] Documentation lookup

### Integration
- [ ] Support mypy/pyright
- [ ] Handle stub files (.pyi)
- [ ] Process type aliases
- [ ] Support TypedDict
- [ ] Handle Protocols

## Summary

Type system understanding enables intelligent code assistance:

- **Type Extraction**: Parse annotations from AST
- **Type Inference**: Deduce types without annotations
- **Type Checking**: Validate type consistency
- **Smart Completions**: Suggest based on types
- **Error Prevention**: Catch type errors early

These capabilities allow AI coding tools like Cursor to provide context-aware suggestions, catch errors before runtime, and generate type-correct code that integrates seamlessly with your existing codebase.

In the next section, we'll explore documentation and comment extraction—understanding the human-written context around code.
`,
};

export default typeSystemUnderstanding;
