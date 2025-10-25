const pythonCodeAnalysis = {
  id: 'python-code-analysis',
  title: 'Python Code Analysis with AST',
  content: `
# Python Code Analysis with AST

## Introduction

Python\'s \`ast\` module provides powerful capabilities for analyzing Python code at a structural level. This is exactly how tools like Cursor, PyLint, and MyPy understand your codebase—they parse Python code into ASTs and extract meaningful information about functions, classes, variables, types, and relationships.

**Why Python AST Analysis Matters:**

When you use Cursor on a Python project, it needs to understand:
- What functions exist and what they do
- Which classes are defined and their inheritance hierarchies
- What variables are in scope at any point
- Which modules are imported and what they provide
- Type hints and annotations for intelligent suggestions
- Code patterns and potential issues

All of this is possible through systematic AST analysis, which we'll master in this section.

## Deep Technical Explanation

### Python's AST Module Architecture

Python\'s \`ast\` module provides a complete toolkit for working with Python ASTs:

**Core Components:**

1. **\`ast.parse()\`**: Converts source code string to AST
2. **\`ast.NodeVisitor\`**: Base class for traversing ASTs (read-only)
3. **\`ast.NodeTransformer\`**: Base class for modifying ASTs
4. **\`ast.walk()\`**: Generator for traversing all nodes
5. **\`ast.unparse()\`**: Converts AST back to source code (Python 3.9+)

### Python AST Node Types

Python's AST has ~80 different node types organized into categories:

**Statements (stmt):**
- \`FunctionDef\`, \`AsyncFunctionDef\`: Function definitions
- \`ClassDef\`: Class definitions
- \`Return\`, \`Delete\`, \`Assign\`: Control flow and operations
- \`For\`, \`While\`, \`If\`: Loops and conditionals
- \`With\`, \`AsyncWith\`: Context managers
- \`Raise\`, \`Try\`, \`ExceptHandler\`: Exception handling
- \`Import\`, \`ImportFrom\`: Module imports

**Expressions (expr):**
- \`BinOp\`, \`UnaryOp\`, \`Compare\`: Operators
- \`Call\`, \`Attribute\`, \`Subscript\`: Access patterns
- \`Lambda\`, \`IfExp\`: Inline expressions
- \`ListComp\`, \`DictComp\`, \`GeneratorExp\`: Comprehensions
- \`Name\`, \`Constant\`: Values and identifiers

**Type Annotations:**
- \`arg\`: Function arguments with annotations
- \`arguments\`: Complete argument specification
- Type expressions for annotations

### Scope Analysis

Understanding variable scope is crucial for code analysis:

\`\`\`python
global_var = 1

def outer():
    outer_var = 2
    
    def inner():
        inner_var = 3
        return global_var + outer_var + inner_var
    
    return inner
\`\`\`

Each function creates a new scope, and the AST preserves this structure:
- Global scope: \`global_var\`
- \`outer\` scope: \`outer_var\`
- \`inner\` scope: \`inner_var\`

## Code Implementation

### Comprehensive Function Analysis

\`\`\`python
import ast
from dataclasses import dataclass
from typing import List, Optional, Set

@dataclass
class ParameterInfo:
    name: str
    annotation: Optional[str]
    default: Optional[str]
    kind: str  # 'positional', 'keyword', 'vararg', 'kwarg'

@dataclass
class FunctionAnalysis:
    name: str
    lineno: int
    col_offset: int
    parameters: List[ParameterInfo]
    return_annotation: Optional[str]
    docstring: Optional[str]
    decorators: List[str]
    is_async: bool
    is_method: bool
    is_static: bool
    is_classmethod: bool
    local_variables: Set[str]
    calls_made: List[str]
    complexity: int

class FunctionAnalyzer (ast.NodeVisitor):
    """
    Comprehensive function analysis.
    Similar to what Cursor does when understanding your functions.
    """
    
    def __init__(self):
        self.functions: List[FunctionAnalysis] = []
        self.current_class: Optional[str] = None
    
    def visit_ClassDef (self, node: ast.ClassDef):
        """Track when we're inside a class."""
        old_class = self.current_class
        self.current_class = node.name
        self.generic_visit (node)
        self.current_class = old_class
    
    def visit_FunctionDef (self, node: ast.FunctionDef):
        """Analyze regular functions."""
        self._analyze_function (node, is_async=False)
        self.generic_visit (node)
    
    def visit_AsyncFunctionDef (self, node: ast.AsyncFunctionDef):
        """Analyze async functions."""
        self._analyze_function (node, is_async=True)
        self.generic_visit (node)
    
    def _analyze_function (self, node, is_async: bool):
        """Core function analysis logic."""
        
        # Parse parameters
        parameters = []
        args = node.args
        
        # Regular positional arguments
        for i, arg in enumerate (args.args):
            default_offset = len (args.args) - len (args.defaults)
            default = None
            if i >= default_offset:
                default_node = args.defaults[i - default_offset]
                default = ast.unparse (default_node)
            
            parameters.append(ParameterInfo(
                name=arg.arg,
                annotation=ast.unparse (arg.annotation) if arg.annotation else None,
                default=default,
                kind='positional'
            ))
        
        # *args
        if args.vararg:
            parameters.append(ParameterInfo(
                name=args.vararg.arg,
                annotation=ast.unparse (args.vararg.annotation) if args.vararg.annotation else None,
                default=None,
                kind='vararg'
            ))
        
        # Keyword-only arguments
        for i, arg in enumerate (args.kwonlyargs):
            default = None
            if args.kw_defaults[i]:
                default = ast.unparse (args.kw_defaults[i])
            
            parameters.append(ParameterInfo(
                name=arg.arg,
                annotation=ast.unparse (arg.annotation) if arg.annotation else None,
                default=default,
                kind='keyword'
            ))
        
        # **kwargs
        if args.kwarg:
            parameters.append(ParameterInfo(
                name=args.kwarg.arg,
                annotation=ast.unparse (args.kwarg.annotation) if args.kwarg.annotation else None,
                default=None,
                kind='kwarg'
            ))
        
        # Return annotation
        return_annotation = None
        if node.returns:
            return_annotation = ast.unparse (node.returns)
        
        # Decorators
        decorators = [ast.unparse (dec) for dec in node.decorator_list]
        
        # Check if method
        is_method = self.current_class is not None
        is_static = 'staticmethod' in decorators
        is_classmethod = 'classmethod' in decorators
        
        # Analyze function body
        local_vars = self._find_local_variables (node.body)
        calls = self._find_function_calls (node.body)
        complexity = self._calculate_complexity (node.body)
        
        # Create analysis
        analysis = FunctionAnalysis(
            name=node.name,
            lineno=node.lineno,
            col_offset=node.col_offset,
            parameters=parameters,
            return_annotation=return_annotation,
            docstring=ast.get_docstring (node),
            decorators=decorators,
            is_async=is_async,
            is_method=is_method,
            is_static=is_static,
            is_classmethod=is_classmethod,
            local_variables=local_vars,
            calls_made=calls,
            complexity=complexity
        )
        
        self.functions.append (analysis)
    
    def _find_local_variables (self, body: List[ast.stmt]) -> Set[str]:
        """Find all variables assigned in function body."""
        variables = set()
        
        for node in ast.walk (ast.Module (body=body)):
            if isinstance (node, ast.Assign):
                for target in node.targets:
                    if isinstance (target, ast.Name):
                        variables.add (target.id)
            elif isinstance (node, ast.AnnAssign):
                if isinstance (node.target, ast.Name):
                    variables.add (node.target.id)
            elif isinstance (node, (ast.For, ast.AsyncFor)):
                if isinstance (node.target, ast.Name):
                    variables.add (node.target.id)
        
        return variables
    
    def _find_function_calls (self, body: List[ast.stmt]) -> List[str]:
        """Find all function calls in function body."""
        calls = []
        
        for node in ast.walk (ast.Module (body=body)):
            if isinstance (node, ast.Call):
                if isinstance (node.func, ast.Name):
                    calls.append (node.func.id)
                elif isinstance (node.func, ast.Attribute):
                    calls.append (node.func.attr)
        
        return calls
    
    def _calculate_complexity (self, body: List[ast.stmt]) -> int:
        """Calculate cyclomatic complexity."""
        complexity = 1  # Base complexity
        
        for node in ast.walk (ast.Module (body=body)):
            if isinstance (node, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance (node, ast.ExceptHandler):
                complexity += 1
            elif isinstance (node, ast.BoolOp):
                complexity += len (node.values) - 1
        
        return complexity

# Example usage
code = """
from typing import List, Optional

class DataProcessor:
    ''Process data with various methods.''
    
    def __init__(self, config: dict):
        self.config = config
        self.cache = {}
    
    @staticmethod
    def validate_input (data: List[int]) -> bool:
        ''Validate that input data is correct.''
        return all (x > 0 for x in data)
    
    @classmethod
    def from_file (cls, filename: str) -> 'DataProcessor':
        ''Create processor from config file.''
        config = load_config (filename)
        return cls (config)
    
    async def process (self, data: List[int], *, strict: bool = False) -> Optional[dict]:
        ''
        Process data asynchronously.
        
        Args:
            data: Input data to process
            strict: Whether to use strict validation
            
        Returns:
            Processed results or None
        ''
        if self.validate_input (data):
            result = await self._process_internal (data)
            return result
        return None
    
    async def _process_internal (self, data: List[int]) -> dict:
        cleaned = [x for x in data if x < 1000]
        total = sum (cleaned)
        return {'data': cleaned, 'total': total}
"""

analyzer = FunctionAnalyzer()
tree = ast.parse (code)
analyzer.visit (tree)

print("=== Function Analysis ===\\n")
for func in analyzer.functions:
    print(f"Function: {func.name} (line {func.lineno})")
    print(f"  Type: {'Async' if func.is_async else 'Sync'}", end=')
    if func.is_method:
        if func.is_static:
            print(" Static Method")
        elif func.is_classmethod:
            print(" Class Method")
        else:
            print(" Instance Method")
    else:
        print(" Function")
    
    print(f"  Parameters:")
    for param in func.parameters:
        param_str = f"    {param.name}"
        if param.annotation:
            param_str += f": {param.annotation}"
        if param.default:
            param_str += f" = {param.default}"
        param_str += f" [{param.kind}]"
        print(param_str)
    
    if func.return_annotation:
        print(f"  Returns: {func.return_annotation}")
    
    if func.decorators:
        print(f"  Decorators: {', '.join (func.decorators)}")
    
    if func.docstring:
        print(f"  Docstring: {func.docstring.split (chr(10))[0]}...")
    
    print(f"  Local Variables: {', '.join (sorted (func.local_variables))}")
    print(f"  Calls: {', '.join (func.calls_made)}")
    print(f"  Complexity: {func.complexity}")
    print()
\`\`\`

### Class Hierarchy Analysis

\`\`\`python
import ast
from dataclasses import dataclass
from typing import List, Dict, Set

@dataclass
class ClassAnalysis:
    name: str
    lineno: int
    bases: List[str]
    decorators: List[str]
    docstring: Optional[str]
    methods: List[str]
    class_methods: List[str]
    static_methods: List[str]
    properties: List[str]
    instance_attributes: Set[str]
    class_attributes: Set[str]

class ClassAnalyzer (ast.NodeVisitor):
    """
    Analyze Python classes in detail.
    Understands inheritance, methods, and attributes.
    """
    
    def __init__(self):
        self.classes: Dict[str, ClassAnalysis] = {}
        self.current_class: Optional[str] = None
    
    def visit_ClassDef (self, node: ast.ClassDef):
        """Analyze a class definition."""
        
        # Extract base classes
        bases = []
        for base in node.bases:
            bases.append (self._expr_to_str (base))
        
        # Extract decorators
        decorators = [ast.unparse (dec) for dec in node.decorator_list]
        
        # Analyze methods
        methods = []
        class_methods = []
        static_methods = []
        properties = []
        
        for item in node.body:
            if isinstance (item, ast.FunctionDef):
                method_name = item.name
                decorators_strs = [ast.unparse (d) for d in item.decorator_list]
                
                if 'classmethod' in decorators_strs:
                    class_methods.append (method_name)
                elif 'staticmethod' in decorators_strs:
                    static_methods.append (method_name)
                elif 'property' in decorators_strs:
                    properties.append (method_name)
                else:
                    methods.append (method_name)
        
        # Find class-level attributes
        class_attributes = set()
        for item in node.body:
            if isinstance (item, ast.Assign):
                for target in item.targets:
                    if isinstance (target, ast.Name):
                        class_attributes.add (target.id)
            elif isinstance (item, ast.AnnAssign):
                if isinstance (item.target, ast.Name):
                    class_attributes.add (item.target.id)
        
        # Find instance attributes (look in __init__)
        instance_attributes = self._find_instance_attributes (node)
        
        # Create analysis
        analysis = ClassAnalysis(
            name=node.name,
            lineno=node.lineno,
            bases=bases,
            decorators=decorators,
            docstring=ast.get_docstring (node),
            methods=methods,
            class_methods=class_methods,
            static_methods=static_methods,
            properties=properties,
            instance_attributes=instance_attributes,
            class_attributes=class_attributes
        )
        
        self.classes[node.name] = analysis
        
        # Visit nested classes
        old_class = self.current_class
        self.current_class = node.name
        self.generic_visit (node)
        self.current_class = old_class
    
    def _expr_to_str (self, node: ast.expr) -> str:
        """Convert expression to string representation."""
        if isinstance (node, ast.Name):
            return node.id
        elif isinstance (node, ast.Attribute):
            return f"{self._expr_to_str (node.value)}.{node.attr}"
        else:
            return ast.unparse (node)
    
    def _find_instance_attributes (self, class_node: ast.ClassDef) -> Set[str]:
        """Find instance attributes assigned in __init__."""
        attributes = set()
        
        # Find __init__ method
        init_method = None
        for item in class_node.body:
            if isinstance (item, ast.FunctionDef) and item.name == '__init__':
                init_method = item
                break
        
        if not init_method:
            return attributes
        
        # Find self.x assignments
        for node in ast.walk (init_method):
            if isinstance (node, ast.Assign):
                for target in node.targets:
                    if isinstance (target, ast.Attribute):
                        if isinstance (target.value, ast.Name) and target.value.id == 'self':
                            attributes.add (target.attr)
            elif isinstance (node, ast.AnnAssign):
                if isinstance (node.target, ast.Attribute):
                    if isinstance (node.target.value, ast.Name) and node.target.value.id == 'self':
                        attributes.add (node.target.attr)
        
        return attributes
    
    def build_inheritance_tree (self) -> Dict[str, List[str]]:
        """Build a tree showing which classes inherit from which."""
        tree = {}
        for class_name, analysis in self.classes.items():
            if analysis.bases:
                tree[class_name] = analysis.bases
            else:
                tree[class_name] = []
        return tree

# Example usage
code = """
from abc import ABC, abstractmethod

class Animal(ABC):
    ''Base animal class.''
    species_count = 0
    
    def __init__(self, name: str, age: int):
        self.name = name
        self.age = age
        Animal.species_count += 1
    
    @abstractmethod
    def make_sound (self) -> str:
        pass
    
    @classmethod
    def get_count (cls) -> int:
        return cls.species_count
    
    @staticmethod
    def is_valid_age (age: int) -> bool:
        return 0 < age < 100
    
    @property
    def info (self) -> str:
        return f"{self.name} ({self.age} years)"

class Dog(Animal):
    ''Dog class inheriting from Animal.''
    breed: str = "Unknown"
    
    def __init__(self, name: str, age: int, breed: str):
        super().__init__(name, age)
        self.breed = breed
        self.tricks = []
    
    def make_sound (self) -> str:
        return "Woof!"
    
    def add_trick (self, trick: str):
        self.tricks.append (trick)

class ServiceDog(Dog):
    ''Service dog with additional capabilities.''
    
    def __init__(self, name: str, age: int, breed: str, service_type: str):
        super().__init__(name, age, breed)
        self.service_type = service_type
        self.certified = False
"""

analyzer = ClassAnalyzer()
tree = ast.parse (code)
analyzer.visit (tree)

print("=== Class Analysis ===\\n")
for class_name, analysis in analyzer.classes.items():
    print(f"Class: {analysis.name} (line {analysis.lineno})")
    
    if analysis.bases:
        print(f"  Inherits from: {', '.join (analysis.bases)}")
    
    if analysis.docstring:
        print(f"  Docstring: {analysis.docstring}")
    
    if analysis.class_attributes:
        print(f"  Class Attributes: {', '.join (sorted (analysis.class_attributes))}")
    
    if analysis.instance_attributes:
        print(f"  Instance Attributes: {', '.join (sorted (analysis.instance_attributes))}")
    
    if analysis.methods:
        print(f"  Methods: {', '.join (analysis.methods)}")
    
    if analysis.class_methods:
        print(f"  Class Methods: {', '.join (analysis.class_methods)}")
    
    if analysis.static_methods:
        print(f"  Static Methods: {', '.join (analysis.static_methods)}")
    
    if analysis.properties:
        print(f"  Properties: {', '.join (analysis.properties)}")
    
    print()

print("=== Inheritance Tree ===")
tree = analyzer.build_inheritance_tree()
for cls, parents in tree.items():
    if parents:
        print(f"{cls} -> {', '.join (parents)}")
    else:
        print(f"{cls} (base class)")
\`\`\`

### Import Analysis

\`\`\`python
import ast
from dataclasses import dataclass
from typing import List, Set

@dataclass
class ImportInfo:
    module: str
    names: List[str]
    asnames: List[str]
    lineno: int
    is_from_import: bool

class ImportAnalyzer (ast.NodeVisitor):
    """
    Analyze all imports in a Python file.
    Critical for understanding dependencies and available symbols.
    """
    
    def __init__(self):
        self.imports: List[ImportInfo] = []
        self.imported_names: Set[str] = set()
    
    def visit_Import (self, node: ast.Import):
        """Handle 'import x' statements."""
        for alias in node.names:
            module = alias.name
            asname = alias.asname if alias.asname else alias.name
            
            self.imports.append(ImportInfo(
                module=module,
                names=[module],
                asnames=[asname],
                lineno=node.lineno,
                is_from_import=False
            ))
            
            self.imported_names.add (asname)
        
        self.generic_visit (node)
    
    def visit_ImportFrom (self, node: ast.ImportFrom):
        """Handle 'from x import y' statements."""
        module = node.module if node.module else '
        names = []
        asnames = []
        
        for alias in node.names:
            name = alias.name
            asname = alias.asname if alias.asname else alias.name
            names.append (name)
            asnames.append (asname)
            self.imported_names.add (asname)
        
        self.imports.append(ImportInfo(
            module=module,
            names=names,
            asnames=asnames,
            lineno=node.lineno,
            is_from_import=True
        ))
        
        self.generic_visit (node)
    
    def get_stdlib_imports (self) -> List[str]:
        """Identify standard library imports."""
        import sys
        stdlib_modules = set (sys.stdlib_module_names)
        
        return [
            imp.module for imp in self.imports 
            if imp.module.split('.')[0] in stdlib_modules
        ]
    
    def get_third_party_imports (self) -> List[str]:
        """Identify third-party imports (rough heuristic)."""
        import sys
        stdlib_modules = set (sys.stdlib_module_names)
        
        third_party = []
        for imp in self.imports:
            root_module = imp.module.split('.')[0]
            if root_module not in stdlib_modules and root_module not in ['.', '..']:
                third_party.append (imp.module)
        
        return third_party
    
    def get_local_imports (self) -> List[ImportInfo]:
        """Identify relative imports."""
        return [imp for imp in self.imports if imp.module.startswith('.')]

# Example usage
code = """
import os
import sys
from pathlib import Path
from typing import List, Dict, Optional
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score

from .utils import helper_function
from ..models import UserModel
from ...config import settings
"""

analyzer = ImportAnalyzer()
tree = ast.parse (code)
analyzer.visit (tree)

print("=== Import Analysis ===\\n")
print("Standard Library Imports:")
for module in analyzer.get_stdlib_imports():
    print(f"  {module}")

print("\\nThird-Party Imports:")
for module in analyzer.get_third_party_imports():
    print(f"  {module}")

print("\\nLocal Imports:")
for imp in analyzer.get_local_imports():
    print(f"  {imp.module} -> {', '.join (imp.names)}")

print(f"\\nTotal imported names available: {len (analyzer.imported_names)}")
print(f"Names: {', '.join (sorted (analyzer.imported_names))}")
\`\`\`

### Variable Scope Tracking

\`\`\`python
import ast
from typing import Dict, Set, List
from dataclasses import dataclass, field

@dataclass
class Scope:
    name: str
    type: str  # 'module', 'class', 'function'
    parent: Optional['Scope']
    variables: Set[str] = field (default_factory=set)
    reads: Set[str] = field (default_factory=set)
    writes: Set[str] = field (default_factory=set)

class ScopeAnalyzer (ast.NodeVisitor):
    """
    Track variable scopes throughout code.
    Essential for understanding variable visibility and shadowing.
    """
    
    def __init__(self):
        self.current_scope: Scope = Scope('module', 'module', None)
        self.all_scopes: List[Scope] = [self.current_scope]
    
    def visit_FunctionDef (self, node: ast.FunctionDef):
        """Enter function scope."""
        # Create new scope
        new_scope = Scope (node.name, 'function', self.current_scope)
        self.all_scopes.append (new_scope)
        
        # Add parameters to function scope
        for arg in node.args.args:
            new_scope.variables.add (arg.arg)
        
        # Visit body in new scope
        old_scope = self.current_scope
        self.current_scope = new_scope
        self.generic_visit (node)
        self.current_scope = old_scope
    
    def visit_ClassDef (self, node: ast.ClassDef):
        """Enter class scope."""
        new_scope = Scope (node.name, 'class', self.current_scope)
        self.all_scopes.append (new_scope)
        
        old_scope = self.current_scope
        self.current_scope = new_scope
        self.generic_visit (node)
        self.current_scope = old_scope
    
    def visit_Assign (self, node: ast.Assign):
        """Track variable assignments."""
        for target in node.targets:
            if isinstance (target, ast.Name):
                self.current_scope.variables.add (target.id)
                self.current_scope.writes.add (target.id)
        self.generic_visit (node)
    
    def visit_AnnAssign (self, node: ast.AnnAssign):
        """Track annotated assignments."""
        if isinstance (node.target, ast.Name):
            self.current_scope.variables.add (node.target.id)
            if node.value:  # Only count as write if assigned
                self.current_scope.writes.add (node.target.id)
        self.generic_visit (node)
    
    def visit_Name (self, node: ast.Name):
        """Track variable reads."""
        if isinstance (node.ctx, ast.Load):
            self.current_scope.reads.add (node.id)
        self.generic_visit (node)
    
    def find_variable_scope (self, var_name: str, from_scope: Scope) -> Optional[Scope]:
        """Find which scope a variable is defined in."""
        scope = from_scope
        while scope:
            if var_name in scope.variables:
                return scope
            scope = scope.parent
        return None
    
    def find_undefined_variables (self, scope: Scope) -> Set[str]:
        """Find variables read but not defined in any parent scope."""
        undefined = set()
        for var in scope.reads:
            if not self.find_variable_scope (var, scope):
                # Check if it's a builtin
                if var not in dir(__builtins__):
                    undefined.add (var)
        return undefined

# Example usage
code = """
# Global scope
global_var = 100

def outer_function (param1):
    # outer_function scope
    outer_var = 200
    
    def inner_function (param2):
        # inner_function scope
        inner_var = 300
        # Read from all scopes
        result = global_var + outer_var + param1 + param2 + inner_var
        return result
    
    return inner_function(10)

class MyClass:
    # class scope
    class_var = 400
    
    def method (self, param3):
        # method scope
        method_var = 500
        return class_var + param3 + method_var  # class_var is undefined!

result = outer_function(5)
"""

analyzer = ScopeAnalyzer()
tree = ast.parse (code)
analyzer.visit (tree)

print("=== Scope Analysis ===\\n")
for scope in analyzer.all_scopes:
    print(f"Scope: {scope.name} ({scope.type})")
    if scope.parent:
        print(f"  Parent: {scope.parent.name}")
    if scope.variables:
        print(f"  Variables defined: {', '.join (sorted (scope.variables))}")
    if scope.reads:
        print(f"  Variables read: {', '.join (sorted (scope.reads))}")
    if scope.writes:
        print(f"  Variables written: {', '.join (sorted (scope.writes))}")
    
    # Check for undefined variables
    undefined = analyzer.find_undefined_variables (scope)
    if undefined:
        print(f"  ⚠️  Undefined variables: {', '.join (sorted (undefined))}")
    
    print()
\`\`\`

## Real-World Case Study: How Cursor Uses Python AST Analysis

Cursor\'s intelligent Python support relies heavily on AST analysis:

**1. Context-Aware Completions:**
\`\`\`python
class DataProcessor:
    def __init__(self, config):
        self.config = config
        self.cache = {}
    
    def process (self, data):
        # When typing here, Cursor knows:
        # - self.config exists (from AST analysis of __init__)
        # - self.cache exists (from AST analysis)
        # - data is a parameter (from function AST)
        # It can suggest: self.config, self.cache, data
        |  # <-- cursor here
\`\`\`

Cursor's AST analysis extracts:
- Instance attributes from \`__init__\`
- Method signatures and parameters
- Available names in current scope

**2. Intelligent Refactoring:**

When you ask Cursor to "extract this into a method," it:
1. Parses the selection into an AST
2. Analyzes variable usage (which are local, which are from self)
3. Determines required parameters
4. Generates proper method signature
5. Updates all references

**3. Type-Aware Generation:**

\`\`\`python
def calculate_stats (data: List[dict]) -> dict:
    # Cursor knows from AST:
    # - data is a List of dicts (from annotation)
    # - Must return a dict (from annotation)
    # - Can suggest: sum, mean, data[0], etc.
\`\`\`

**4. Import Management:**

Cursor tracks imports through AST analysis:
- Knows what's already imported
- Can suggest adding imports
- Understands aliased imports (import pandas as pd)
- Recognizes from imports (from pathlib import Path)

## Hands-On Exercise

Build a Python code analyzer that creates an LLM-friendly summary:

\`\`\`python
import ast
from dataclasses import dataclass
from typing import List, Dict

class PythonCodeSummarizer:
    """
    Create comprehensive summaries of Python code for LLM context.
    This is similar to what Cursor does when building context.
    """
    
    def __init__(self, code: str):
        self.code = code
        self.tree = ast.parse (code)
        self.summary = {}
    
    def analyze (self) -> Dict:
        """Run all analyses and build summary."""
        self.summary['imports'] = self._analyze_imports()
        self.summary['classes'] = self._analyze_classes()
        self.summary['functions'] = self._analyze_functions()
        self.summary['globals'] = self._analyze_globals()
        self.summary['complexity'] = self._calculate_overall_complexity()
        return self.summary
    
    def _analyze_imports (self) -> List[Dict]:
        """Extract import information."""
        imports = []
        for node in ast.walk (self.tree):
            if isinstance (node, ast.Import):
                for alias in node.names:
                    imports.append({
                        'type': 'import',
                        'module': alias.name,
                        'as': alias.asname
                    })
            elif isinstance (node, ast.ImportFrom):
                imports.append({
                    'type': 'from',
                    'module': node.module,
                    'names': [alias.name for alias in node.names]
                })
        return imports
    
    def _analyze_classes (self) -> List[Dict]:
        """Extract class information."""
        classes = []
        for node in ast.walk (self.tree):
            if isinstance (node, ast.ClassDef):
                methods = [
                    n.name for n in node.body 
                    if isinstance (n, ast.FunctionDef)
                ]
                classes.append({
                    'name': node.name,
                    'bases': [ast.unparse (b) for b in node.bases],
                    'methods': methods,
                    'docstring': ast.get_docstring (node)
                })
        return classes
    
    def _analyze_functions (self) -> List[Dict]:
        """Extract function information."""
        functions = []
        for node in ast.walk (self.tree):
            if isinstance (node, ast.FunctionDef):
                functions.append({
                    'name': node.name,
                    'params': [arg.arg for arg in node.args.args],
                    'returns': ast.unparse (node.returns) if node.returns else None,
                    'docstring': ast.get_docstring (node)
                })
        return functions
    
    def _analyze_globals (self) -> List[str]:
        """Find global variables."""
        globals_vars = []
        for node in self.tree.body:
            if isinstance (node, ast.Assign):
                for target in node.targets:
                    if isinstance (target, ast.Name):
                        globals_vars.append (target.id)
        return globals_vars
    
    def _calculate_overall_complexity (self) -> int:
        """Calculate total cyclomatic complexity."""
        complexity = 0
        for node in ast.walk (self.tree):
            if isinstance (node, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
        return complexity
    
    def to_text_summary (self) -> str:
        """Generate text summary for LLM context."""
        lines = ["# Python Code Summary\\n"]
        
        if self.summary.get('imports'):
            lines.append("## Imports")
            for imp in self.summary['imports']:
                if imp['type'] == 'import':
                    lines.append (f"- {imp['module']}")
                else:
                    lines.append (f"- from {imp['module']} import {', '.join (imp['names'])}")
            lines.append("")
        
        if self.summary.get('classes'):
            lines.append("## Classes")
            for cls in self.summary['classes']:
                lines.append (f"### {cls['name']}")
                if cls['bases']:
                    lines.append (f"Inherits: {', '.join (cls['bases'])}")
                if cls['methods']:
                    lines.append (f"Methods: {', '.join (cls['methods'])}")
                if cls['docstring']:
                    lines.append (f"Purpose: {cls['docstring'].split (chr(10))[0]}")
                lines.append("")
        
        if self.summary.get('functions'):
            lines.append("## Functions")
            for func in self.summary['functions']:
                sig = f"{func['name']}({', '.join (func['params'])})"
                if func['returns']:
                    sig += f" -> {func['returns']}"
                lines.append (f"### {sig}")
                if func['docstring']:
                    lines.append (f"{func['docstring'].split (chr(10))[0]}")
                lines.append("")
        
        lines.append (f"## Metrics")
        lines.append (f"- Complexity: {self.summary.get('complexity', 0)}")
        lines.append (f"- Classes: {len (self.summary.get('classes', []))}")
        lines.append (f"- Functions: {len (self.summary.get('functions', []))}")
        
        return "\\n".join (lines)

# Test it
code = """
from typing import List, Optional
import pandas as pd

class DataAnalyzer:
    ''Analyze data with pandas.''
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
    
    def compute_stats (self) -> dict:
        ''Compute basic statistics.''
        return {
            'mean': self.data.mean(),
            'std': self.data.std()
        }
    
    def filter_data (self, threshold: float) -> pd.DataFrame:
        ''Filter data above threshold.''
        return self.data[self.data > threshold]

def load_data (filename: str) -> pd.DataFrame:
    ''Load data from CSV file.''
    return pd.read_csv (filename)

def main():
    df = load_data('data.csv')
    analyzer = DataAnalyzer (df)
    stats = analyzer.compute_stats()
    print(stats)
"""

summarizer = PythonCodeSummarizer (code)
summarizer.analyze()
print(summarizer.to_text_summary())
\`\`\`

## Common Pitfalls

### 1. Not Handling Async Functions

\`\`\`python
# ❌ Wrong: Only handles regular functions
def find_functions (tree):
    return [n for n in ast.walk (tree) if isinstance (n, ast.FunctionDef)]

# ✅ Correct: Handle both sync and async
def find_functions (tree):
    return [
        n for n in ast.walk (tree) 
        if isinstance (n, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
\`\`\`

### 2. Forgetting to Check Node Types

\`\`\`python
# ❌ Wrong: Assumes structure
def get_first_func_name (tree):
    return tree.body[0].name  # Crashes if first item isn't a function

# ✅ Correct: Verify types
def get_first_func_name (tree):
    for node in tree.body:
        if isinstance (node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return node.name
    return None
\`\`\`

### 3. Missing call to generic_visit()

\`\`\`python
# ❌ Wrong: Won't visit children
class MyVisitor (ast.NodeVisitor):
    def visit_FunctionDef (self, node):
        print(node.name)
        # Forgot generic_visit! Won't see nested functions

# ✅ Correct: Always visit children
class MyVisitor (ast.NodeVisitor):
    def visit_FunctionDef (self, node):
        print(node.name)
        self.generic_visit (node)  # Important!
\`\`\`

### 4. Ignoring Different Import Types

\`\`\`python
# ❌ Wrong: Only handles one import type
def get_imports (tree):
    return [node.names[0].name for node in ast.walk (tree) if isinstance (node, ast.Import)]

# ✅ Correct: Handle both import types
def get_imports (tree):
    imports = []
    for node in ast.walk (tree):
        if isinstance (node, ast.Import):
            imports.extend([alias.name for alias in node.names])
        elif isinstance (node, ast.ImportFrom):
            imports.append (node.module)
    return imports
\`\`\`

## Production Checklist

### Analysis Quality
- [ ] Handle all function types (sync, async, lambda)
- [ ] Track both class and instance methods
- [ ] Identify static methods and class methods
- [ ] Extract type annotations accurately
- [ ] Parse decorators correctly
- [ ] Handle property decorators

### Scope Handling
- [ ] Track variable scope correctly
- [ ] Identify variable shadowing
- [ ] Handle global and nonlocal keywords
- [ ] Track closure variables
- [ ] Identify undefined variables

### Performance
- [ ] Cache parsed ASTs for unchanged files
- [ ] Use ast.walk() for simple traversals
- [ ] Use NodeVisitor for complex analysis
- [ ] Avoid re-parsing unnecessarily
- [ ] Profile analysis for bottlenecks

### Error Handling
- [ ] Handle syntax errors gracefully
- [ ] Check for None values before access
- [ ] Verify node types before casting
- [ ] Handle incomplete or malformed ASTs
- [ ] Provide meaningful error messages

### Integration
- [ ] Export analysis results in useful formats
- [ ] Provide both detailed and summary views
- [ ] Support incremental analysis
- [ ] Enable filtering and querying results
- [ ] Document expected input/output

## Summary

Python AST analysis provides powerful code understanding capabilities:

- **Function Analysis**: Extract signatures, parameters, types, and complexity
- **Class Analysis**: Understand inheritance, methods, and attributes
- **Import Tracking**: Know what modules and symbols are available
- **Scope Analysis**: Track variable definitions and usage
- **Code Summaries**: Generate LLM-friendly code descriptions

These techniques form the foundation for building intelligent Python tools like Cursor, enabling context-aware suggestions, accurate refactoring, and smart code generation.

In the next section, we'll explore tree-sitter for multi-language parsing, allowing us to analyze code in any programming language with the same systematic approach.
`,
};

export default pythonCodeAnalysis;
