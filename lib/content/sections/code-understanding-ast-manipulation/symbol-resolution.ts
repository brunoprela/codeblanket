const symbolResolution = {
  id: 'symbol-resolution',
  title: 'Symbol Resolution & References',
  content: `
# Symbol Resolution & References

## Introduction

When you use Cursor and it suggests \`user.email\` after you type \`user.\`, it's performing symbol resolution—figuring out what \`user\` refers to and what attributes it has. When you "go to definition" in your IDE, that's reference tracking. These capabilities are fundamental to intelligent code understanding.

**Why Symbol Resolution Matters:**

For AI coding tools like Cursor to be truly helpful, they need to:
- Understand what names refer to (variables, functions, classes)
- Track where symbols are defined and used
- Resolve imports and dependencies
- Understand scope and shadowing
- Provide accurate "go to definition" functionality
- Find all references to a symbol

This section teaches you to build these capabilities.

## Deep Technical Explanation

### What is Symbol Resolution?

Symbol resolution is the process of determining what a name (symbol) refers to in code:

\`\`\`python
from models import User  # User is now a symbol

def create_user(name):   # create_user is a symbol; name is a symbol
    user = User(name=name)  # user is a symbol
    return user
    # When we see 'User', we resolve it to the class from models
    # When we see 'name', we resolve it to the parameter
    # When we see 'user', we resolve it to the local variable
\`\`\`

### Symbol Tables

A symbol table maps names to their definitions:

\`\`\`
Symbol Table for create_user function:
{
    'name': Parameter(line=2, type='parameter'),
    'user': LocalVariable(line=3, type=User),
    'User': ImportedClass(from='models', line=1),
    'create_user': Function(line=2),
}
\`\`\`

### Scope Chains

Symbol resolution follows scope chains from inner to outer:

\`\`\`python
x = 1  # Global scope

def outer():
    x = 2  # outer scope (shadows global)
    
    def inner():
        x = 3  # inner scope (shadows outer and global)
        print(x)  # Resolves to inner's x (3)
    
    print(x)  # Resolves to outer's x (2)

print(x)  # Resolves to global x (1)

# Resolution chain: inner scope → outer scope → global scope → builtins
\`\`\`

### Import Resolution

Understanding what imports make available:

\`\`\`python
import os                    # os module available
from pathlib import Path     # Path class available
import numpy as np           # numpy available as np
from typing import List, Dict # List and Dict available

# Resolution:
# os → module os
# Path → class Path from pathlib
# np → module numpy (aliased)
# List → type List from typing
\`\`\`

### Attribute Resolution

Resolving object attributes and method calls:

\`\`\`python
class User:
    def __init__(self, name):
        self.name = name  # self.name is an attribute
    
    def greet(self):
        return f"Hello, {self.name}"  # Resolves self.name to attribute

user = User("Alice")
print(user.name)      # Resolves user → User instance → name attribute
print(user.greet())   # Resolves user → User instance → greet method
\`\`\`

## Code Implementation

### Symbol Table Builder

\`\`\`python
import ast
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
from enum import Enum

class SymbolType(Enum):
    MODULE = "module"
    CLASS = "class"
    FUNCTION = "function"
    METHOD = "method"
    PARAMETER = "parameter"
    LOCAL_VAR = "local_var"
    GLOBAL_VAR = "global_var"
    IMPORT = "import"
    ATTRIBUTE = "attribute"

@dataclass
class Symbol:
    name: str
    type: SymbolType
    defined_at: int
    scope: str
    references: List[int] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)

class SymbolTableBuilder(ast.NodeVisitor):
    """
    Build comprehensive symbol tables for code.
    This is the foundation of IDE features like "go to definition".
    """
    
    def __init__(self):
        self.symbols: Dict[str, Symbol] = {}
        self.scopes: List[str] = ['<module>']
        self.current_class: Optional[str] = None
    
    @property
    def current_scope(self) -> str:
        return self.scopes[-1]
    
    def _add_symbol(self, name: str, symbol_type: SymbolType, lineno: int, **metadata):
        """Add a symbol to the table."""
        full_name = f"{self.current_scope}.{name}"
        
        if full_name not in self.symbols:
            self.symbols[full_name] = Symbol(
                name=name,
                type=symbol_type,
                defined_at=lineno,
                scope=self.current_scope,
                metadata=metadata
            )
    
    def _add_reference(self, name: str, lineno: int):
        """Add a reference to a symbol."""
        # Try to find symbol in current and parent scopes
        symbol = self._resolve_symbol(name)
        if symbol:
            symbol.references.append(lineno)
    
    def _resolve_symbol(self, name: str) -> Optional[Symbol]:
        """Resolve a name to a symbol following scope chain."""
        # Try current scope and all parent scopes
        for i in range(len(self.scopes) - 1, -1, -1):
            scope = '.'.join(self.scopes[:i+1])
            full_name = f"{scope}.{name}"
            if full_name in self.symbols:
                return self.symbols[full_name]
        return None
    
    def visit_Import(self, node: ast.Import):
        """Track import statements."""
        for alias in node.names:
            name = alias.asname if alias.asname else alias.name
            self._add_symbol(
                name,
                SymbolType.IMPORT,
                node.lineno,
                module=alias.name,
                asname=alias.asname
            )
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node: ast.ImportFrom):
        """Track from...import statements."""
        for alias in node.names:
            name = alias.asname if alias.asname else alias.name
            self._add_symbol(
                name,
                SymbolType.IMPORT,
                node.lineno,
                module=node.module,
                imported_name=alias.name,
                asname=alias.asname
            )
        self.generic_visit(node)
    
    def visit_ClassDef(self, node: ast.ClassDef):
        """Track class definitions."""
        self._add_symbol(
            node.name,
            SymbolType.CLASS,
            node.lineno,
            bases=[ast.unparse(b) for b in node.bases],
            decorators=[ast.unparse(d) for d in node.decorator_list]
        )
        
        # Enter class scope
        old_class = self.current_class
        self.current_class = node.name
        self.scopes.append(node.name)
        
        self.generic_visit(node)
        
        # Exit class scope
        self.scopes.pop()
        self.current_class = old_class
    
    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Track function definitions."""
        symbol_type = SymbolType.METHOD if self.current_class else SymbolType.FUNCTION
        
        self._add_symbol(
            node.name,
            symbol_type,
            node.lineno,
            params=[arg.arg for arg in node.args.args],
            decorators=[ast.unparse(d) for d in node.decorator_list]
        )
        
        # Enter function scope
        self.scopes.append(node.name)
        
        # Add parameters to symbol table
        for arg in node.args.args:
            self._add_symbol(
                arg.arg,
                SymbolType.PARAMETER,
                node.lineno
            )
        
        self.generic_visit(node)
        
        # Exit function scope
        self.scopes.pop()
    
    def visit_Assign(self, node: ast.Assign):
        """Track variable assignments."""
        for target in node.targets:
            if isinstance(target, ast.Name):
                # Determine if global or local
                symbol_type = (
                    SymbolType.GLOBAL_VAR if len(self.scopes) == 1 
                    else SymbolType.LOCAL_VAR
                )
                self._add_symbol(
                    target.id,
                    symbol_type,
                    node.lineno
                )
            elif isinstance(target, ast.Attribute):
                # Track attribute assignment (like self.x = ...)
                if isinstance(target.value, ast.Name) and target.value.id == 'self':
                    self._add_symbol(
                        target.attr,
                        SymbolType.ATTRIBUTE,
                        node.lineno,
                        class_name=self.current_class
                    )
        
        self.generic_visit(node)
    
    def visit_Name(self, node: ast.Name):
        """Track name references."""
        if isinstance(node.ctx, ast.Load):
            # Name is being read, add as reference
            self._add_reference(node.id, node.lineno)
        self.generic_visit(node)
    
    def find_definition(self, name: str, scope: Optional[str] = None) -> Optional[Symbol]:
        """Find the definition of a symbol."""
        if scope:
            full_name = f"{scope}.{name}"
            return self.symbols.get(full_name)
        else:
            # Search in all scopes
            return self._resolve_symbol(name)
    
    def find_all_references(self, name: str, scope: Optional[str] = None) -> List[int]:
        """Find all references to a symbol."""
        symbol = self.find_definition(name, scope)
        if symbol:
            return symbol.references
        return []
    
    def get_symbols_in_scope(self, scope: str) -> List[Symbol]:
        """Get all symbols defined in a specific scope."""
        return [
            symbol for symbol in self.symbols.values()
            if symbol.scope == scope
        ]
    
    def visualize_symbols(self) -> str:
        """Create a text visualization of symbol table."""
        lines = ["=== Symbol Table ===\\n"]
        
        # Group by scope
        scopes = {}
        for symbol in self.symbols.values():
            if symbol.scope not in scopes:
                scopes[symbol.scope] = []
            scopes[symbol.scope].append(symbol)
        
        for scope, symbols in sorted(scopes.items()):
            lines.append(f"Scope: {scope}")
            for symbol in sorted(symbols, key=lambda s: s.defined_at):
                ref_count = len(symbol.references)
                lines.append(
                    f"  {symbol.name} [{symbol.type.value}] "
                    f"(line {symbol.defined_at}, {ref_count} refs)"
                )
                if symbol.references:
                    ref_lines = ', '.join(str(r) for r in sorted(symbol.references)[:5])
                    if len(symbol.references) > 5:
                        ref_lines += "..."
                    lines.append(f"    Referenced at: {ref_lines}")
            lines.append("")
        
        return "\\n".join(lines)

# Example usage
code = """
import os
from typing import List

class UserService:
    def __init__(self, db):
        self.db = db
        self.cache = {}
    
    def get_user(self, user_id: int) -> dict:
        if user_id in self.cache:
            return self.cache[user_id]
        
        user = self.db.query(user_id)
        self.cache[user_id] = user
        return user
    
    def clear_cache(self):
        self.cache = {}

def create_service():
    return UserService(database)

database = None
"""

builder = SymbolTableBuilder()
tree = ast.parse(code)
builder.visit(tree)

print(builder.visualize_symbols())

# Test definition finding
print("\\n=== Symbol Resolution Examples ===")
symbol = builder.find_definition('get_user', '<module>.UserService')
if symbol:
    print(f"\\nget_user is defined at line {symbol.defined_at}")
    print(f"References: {len(symbol.references)}")

symbol = builder.find_definition('cache', '<module>.UserService')
if symbol:
    print(f"\\ncache is an {symbol.type.value} defined at line {symbol.defined_at}")
    print(f"Referenced at lines: {', '.join(map(str, symbol.references))}")
\`\`\`

### Cross-Reference Analyzer

\`\`\`python
import ast
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

@dataclass
class Reference:
    name: str
    line: int
    column: int
    ref_type: str  # 'read', 'write', 'call'
    context: str

class CrossReferenceAnalyzer(ast.NodeVisitor):
    """
    Build cross-references between symbols.
    Enables "find all references" functionality.
    """
    
    def __init__(self):
        self.definitions: Dict[str, List[Tuple[int, int]]] = {}
        self.references: Dict[str, List[Reference]] = {}
        self.current_function: Optional[str] = None
    
    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Track function definitions."""
        self.definitions.setdefault(node.name, []).append(
            (node.lineno, node.col_offset)
        )
        
        old_func = self.current_function
        self.current_function = node.name
        self.generic_visit(node)
        self.current_function = old_func
    
    def visit_ClassDef(self, node: ast.ClassDef):
        """Track class definitions."""
        self.definitions.setdefault(node.name, []).append(
            (node.lineno, node.col_offset)
        )
        self.generic_visit(node)
    
    def visit_Assign(self, node: ast.Assign):
        """Track assignments."""
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.references.setdefault(target.id, []).append(Reference(
                    name=target.id,
                    line=node.lineno,
                    column=node.col_offset,
                    ref_type='write',
                    context=self.current_function or '<module>'
                ))
        self.generic_visit(node)
    
    def visit_Name(self, node: ast.Name):
        """Track name usage."""
        if isinstance(node.ctx, ast.Load):
            self.references.setdefault(node.id, []).append(Reference(
                name=node.id,
                line=node.lineno,
                column=node.col_offset,
                ref_type='read',
                context=self.current_function or '<module>'
            ))
        self.generic_visit(node)
    
    def visit_Call(self, node: ast.Call):
        """Track function calls."""
        if isinstance(node.func, ast.Name):
            self.references.setdefault(node.func.id, []).append(Reference(
                name=node.func.id,
                line=node.lineno,
                column=node.col_offset,
                ref_type='call',
                context=self.current_function or '<module>'
            ))
        self.generic_visit(node)
    
    def find_all_references(self, name: str) -> List[Reference]:
        """Find all references to a name."""
        return self.references.get(name, [])
    
    def find_definition(self, name: str) -> Optional[Tuple[int, int]]:
        """Find where a name is defined."""
        definitions = self.definitions.get(name, [])
        return definitions[0] if definitions else None
    
    def get_reference_report(self, name: str) -> str:
        """Generate a report of all references to a name."""
        refs = self.find_all_references(name)
        definition = self.find_definition(name)
        
        lines = [f"=== References for '{name}' ===\\n"]
        
        if definition:
            lines.append(f"Defined at: line {definition[0]}, column {definition[1]}")
        else:
            lines.append("Definition: Not found")
        
        if refs:
            lines.append(f"\\nReferences ({len(refs)}):")
            
            # Group by type
            by_type = {}
            for ref in refs:
                by_type.setdefault(ref.ref_type, []).append(ref)
            
            for ref_type, refs_of_type in sorted(by_type.items()):
                lines.append(f"\\n  {ref_type.capitalize()}:")
                for ref in sorted(refs_of_type, key=lambda r: r.line):
                    lines.append(
                        f"    Line {ref.line}, col {ref.column} "
                        f"(in {ref.context})"
                    )
        else:
            lines.append("\\nNo references found")
        
        return "\\n".join(lines)

# Example usage
code = """
def calculate(x, y):
    result = x + y
    print(result)
    return result

def process():
    a = 10
    b = 20
    total = calculate(a, b)
    print(f"Total: {total}")
    return total

value = process()
print(value)
"""

analyzer = CrossReferenceAnalyzer()
tree = ast.parse(code)
analyzer.visit(tree)

# Show references for different symbols
for name in ['calculate', 'result', 'total']:
    print(analyzer.get_reference_report(name))
    print()
\`\`\`

### Import Resolver

\`\`\`python
import ast
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Set
import sys

@dataclass
class ImportedSymbol:
    name: str
    module: str
    original_name: Optional[str]  # For aliased imports
    line: int

class ImportResolver:
    """
    Resolve imports to understand what symbols are available.
    Critical for understanding external dependencies.
    """
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.imports: Dict[str, ImportedSymbol] = {}
        self.modules: Set[str] = set()
    
    def analyze_file(self, filepath: str):
        """Analyze imports in a file."""
        with open(filepath, 'r') as f:
            code = f.read()
        
        tree = ast.parse(code)
        self._extract_imports(tree, filepath)
    
    def _extract_imports(self, tree: ast.AST, filepath: str):
        """Extract all imports from AST."""
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    symbol_name = alias.asname if alias.asname else alias.name
                    self.imports[symbol_name] = ImportedSymbol(
                        name=symbol_name,
                        module=alias.name,
                        original_name=alias.name if alias.asname else None,
                        line=node.lineno
                    )
                    self.modules.add(alias.name)
            
            elif isinstance(node, ast.ImportFrom):
                module = node.module or '
                for alias in node.names:
                    symbol_name = alias.asname if alias.asname else alias.name
                    self.imports[symbol_name] = ImportedSymbol(
                        name=symbol_name,
                        module=module,
                        original_name=alias.name if alias.asname else None,
                        line=node.lineno
                    )
                    self.modules.add(module)
    
    def resolve_import(self, name: str) -> Optional[ImportedSymbol]:
        """Resolve a name to its import."""
        return self.imports.get(name)
    
    def is_stdlib(self, module: str) -> bool:
        """Check if a module is from standard library."""
        if hasattr(sys, 'stdlib_module_names'):
            return module.split('.')[0] in sys.stdlib_module_names
        # Fallback for older Python versions
        try:
            __import__(module)
            return True
        except ImportError:
            return False
    
    def categorize_imports(self) -> Dict[str, List[str]]:
        """Categorize imports into stdlib, third-party, and local."""
        categories = {
            'stdlib': [],
            'third_party': [],
            'local': []
        }
        
        for module in self.modules:
            if not module:
                continue
            
            root = module.split('.')[0]
            
            if self.is_stdlib(root):
                categories['stdlib'].append(module)
            elif module.startswith('.'):
                categories['local'].append(module)
            else:
                # Check if it's in project
                potential_path = self.project_root / (root.replace('.', '/') + '.py')
                if potential_path.exists():
                    categories['local'].append(module)
                else:
                    categories['third_party'].append(module)
        
        return categories
    
    def find_unused_imports(self, code: str) -> List[str]:
        """Find imported symbols that are never used."""
        tree = ast.parse(code)
        
        # Get all names used in code
        used_names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                used_names.add(node.id)
        
        # Find imports not in used names
        unused = []
        for name, imp in self.imports.items():
            if name not in used_names:
                unused.append(name)
        
        return unused
    
    def generate_import_graph(self) -> str:
        """Generate a visualization of imports."""
        categories = self.categorize_imports()
        
        lines = ["=== Import Graph ===\\n"]
        
        for category, modules in sorted(categories.items()):
            if modules:
                lines.append(f"{category.title()} Imports:")
                for module in sorted(modules):
                    # Find symbols imported from this module
                    symbols = [
                        sym.name for sym in self.imports.values()
                        if sym.module == module
                    ]
                    if symbols:
                        lines.append(f"  {module} → {', '.join(symbols)}")
                    else:
                        lines.append(f"  {module}")
                lines.append("")
        
        return "\\n".join(lines)

# Example usage
code = """
import os
import sys
from pathlib import Path
from typing import List, Dict
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def process_data(filename):
    # Uses Path, pd, np
    path = Path(filename)
    df = pd.read_csv(path)
    data = np.array(df.values)
    return train_test_split(data)

# Note: os, sys, List, Dict are imported but not used
"""

# Assuming current directory as project root
resolver = ImportResolver('.')
tree = ast.parse(code)
resolver._extract_imports(tree, 'example.py')

print(resolver.generate_import_graph())

# Find unused imports
unused = resolver.find_unused_imports(code)
if unused:
    print(f"\\n⚠️  Unused imports: {', '.join(unused)}")
\`\`\`

### Go-to-Definition Implementation

\`\`\`python
import ast
from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class Location:
    filepath: str
    line: int
    column: int
    end_line: Optional[int] = None
    end_column: Optional[int] = None

class GoToDefinitionResolver:
    """
    Implement "go to definition" functionality.
    This is what happens when you Cmd+Click in Cursor.
    """
    
    def __init__(self, code: str, filepath: str = '<stdin>'):
        self.code = code
        self.filepath = filepath
        self.tree = ast.parse(code)
        self.definitions: Dict[str, Location] = {}
        self._build_definitions()
    
    def _build_definitions(self):
        """Build index of all definitions."""
        for node in ast.walk(self.tree):
            if isinstance(node, ast.FunctionDef):
                self.definitions[node.name] = Location(
                    filepath=self.filepath,
                    line=node.lineno,
                    column=node.col_offset,
                    end_line=node.end_lineno,
                    end_column=node.end_col_offset
                )
            elif isinstance(node, ast.ClassDef):
                self.definitions[node.name] = Location(
                    filepath=self.filepath,
                    line=node.lineno,
                    column=node.col_offset,
                    end_line=node.end_lineno,
                    end_column=node.end_col_offset
                )
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        self.definitions[target.id] = Location(
                            filepath=self.filepath,
                            line=node.lineno,
                            column=node.col_offset
                        )
    
    def find_definition_at_position(self, line: int, column: int) -> Optional[Location]:
        """
        Find definition for symbol at given position.
        This is the core "go to definition" functionality.
        """
        # Find the node at this position
        target_node = self._find_node_at_position(line, column)
        
        if not target_node:
            return None
        
        # If it's a Name node, resolve to definition
        if isinstance(target_node, ast.Name):
            return self.definitions.get(target_node.id)
        
        # If it's an Attribute, try to resolve
        if isinstance(target_node, ast.Attribute):
            return self.definitions.get(target_node.attr)
        
        return None
    
    def _find_node_at_position(self, line: int, column: int) -> Optional[ast.AST]:
        """Find the AST node at a specific position."""
        for node in ast.walk(self.tree):
            if not hasattr(node, 'lineno'):
                continue
            
            # Check if position is within node
            if node.lineno == line:
                if hasattr(node, 'col_offset'):
                    if node.col_offset <= column:
                        if hasattr(node, 'end_col_offset'):
                            if column <= node.end_col_offset:
                                return node
        
        return None
    
    def get_hover_info(self, line: int, column: int) -> Optional[str]:
        """
        Get information to show on hover.
        This is what IDEs show when you hover over a symbol.
        """
        node = self._find_node_at_position(line, column)
        
        if isinstance(node, ast.Name):
            definition = self.definitions.get(node.id)
            if definition:
                return f"{node.id} defined at line {definition.line}"
        
        elif isinstance(node, ast.FunctionDef):
            params = [arg.arg for arg in node.args.args]
            return f"def {node.name}({', '.join(params)})"
        
        elif isinstance(node, ast.ClassDef):
            bases = [ast.unparse(b) for b in node.bases]
            base_str = f"({', '.join(bases)})" if bases else ""
            return f"class {node.name}{base_str}"
        
        return None

# Example usage
code = """
class User:
    def __init__(self, name):
        self.name = name
    
    def greet(self):
        return f"Hello, {self.name}"

def create_user(name):
    user = User(name)
    return user

result = create_user("Alice")
message = result.greet()
"""

resolver = GoToDefinitionResolver(code, 'example.py')

# Test go-to-definition for various symbols
test_positions = [
    (9, 11),  # 'user' variable
    (9, 16),  # 'User' class
    (12, 9),  # 'create_user' function
    (13, 18),  # 'greet' method
]

print("=== Go-to-Definition Tests ===\\n")
lines = code.split('\\n')
for line, col in test_positions:
    location = resolver.find_definition_at_position(line, col)
    code_snippet = lines[line - 1][max(0, col-5):col+15].strip()
    
    print(f"Position ({line}, {col}): '{code_snippet}'")
    if location:
        print(f"  → Definition at line {location.line}, column {location.column}")
    else:
        print(f"  → No definition found")
    print()
\`\`\`

## Real-World Case Study: How Cursor Uses Symbol Resolution

Cursor leverages symbol resolution for intelligent features:

**1. Auto-completion:**
\`\`\`python
class UserService:
    def __init__(self, db):
        self.db = db
        self.cache = {}
    
    def get_user(self, user_id):
        # When you type 'self.' here, Cursor:
        # 1. Resolves 'self' to UserService instance
        # 2. Looks up UserService symbol in table
        # 3. Finds attributes: db, cache
        # 4. Suggests: self.db, self.cache
        self.  # <-- Shows: db, cache
\`\`\`

**2. Go-to-Definition:**
\`\`\`python
from models import User

def create_user(name):
    user = User(name=name)  # Cmd+click on 'User'
    # Cursor:
    # 1. Resolves 'User' to imported symbol
    # 2. Looks up import: from models import User
    # 3. Opens models.py
    # 4. Jumps to User class definition
    return user
\`\`\`

**3. Find All References:**
\`\`\`python
def calculate(x, y):  # Right-click → "Find All References"
    return x + y

result1 = calculate(1, 2)
result2 = calculate(3, 4)

# Cursor:
# 1. Builds cross-reference table
# 2. Finds all references to 'calculate'
# 3. Highlights: line 4, line 5
# 4. Shows count: "2 references"
\`\`\`

**4. Rename Symbol:**
\`\`\`python
def process_data(data):  # Right-click → "Rename Symbol"
    # Type new name: process_items
    # Cursor:
    # 1. Finds all references via symbol table
    # 2. Updates definition and all references
    # 3. Maintains correctness across file
    result = transform(data)
    return result

output = process_data(input)  # Also renamed automatically
\`\`\`

## Hands-On Exercise

Build a complete symbol resolution system:

\`\`\`python
class ComprehensiveSymbolResolver:
    """
    Complete symbol resolution system.
    Combines all techniques for production-ready functionality.
    """
    
    def __init__(self, code: str, filepath: str = '<stdin>'):
        self.code = code
        self.filepath = filepath
        self.tree = ast.parse(code)
        
        # Build all indexes
        self.symbol_table = SymbolTableBuilder()
        self.symbol_table.visit(self.tree)
        
        self.cross_refs = CrossReferenceAnalyzer()
        self.cross_refs.visit(self.tree)
        
        self.goto_def = GoToDefinitionResolver(code, filepath)
    
    def resolve_at_cursor(self, line: int, column: int) -> Dict:
        """
        Complete resolution for position.
        Returns all available information.
        """
        # Find what's at this position
        node = self.goto_def._find_node_at_position(line, column)
        
        if not node:
            return {'error': 'No symbol at position'}
        
        result = {
            'node_type': type(node).__name__,
            'line': line,
            'column': column,
        }
        
        # If it's a name, get full details
        if isinstance(node, ast.Name):
            name = node.id
            result['name'] = name
            
            # Get definition
            definition = self.goto_def.find_definition_at_position(line, column)
            if definition:
                result['definition'] = {
                    'line': definition.line,
                    'column': definition.column,
                    'file': definition.filepath
                }
            
            # Get all references
            refs = self.cross_refs.find_all_references(name)
            result['references'] = [
                {'line': ref.line, 'type': ref.ref_type, 'context': ref.context}
                for ref in refs
            ]
            
            # Get symbol info
            symbol = self.symbol_table.find_definition(name)
            if symbol:
                result['symbol'] = {
                    'type': symbol.type.value,
                    'scope': symbol.scope,
                    'defined_at': symbol.defined_at
                }
        
        # Get hover info
        hover = self.goto_def.get_hover_info(line, column)
        if hover:
            result['hover'] = hover
        
        return result
    
    def generate_context_for_llm(self, focus_line: int) -> str:
        """
        Generate rich context for LLM about code at a specific line.
        This is what Cursor sends to the LLM for intelligent suggestions.
        """
        lines = [f"# Context for line {focus_line}\\n"]
        
        # Add symbol table context
        lines.append("## Available Symbols")
        for scope_name, symbols in self.symbol_table.symbols.items():
            if symbols:
                lines.append(f"### {scope_name}")
                for symbol in symbols[:10]:  # Limit for context
                    lines.append(f"- {symbol.name} ({symbol.type.value})")
        
        # Add relevant definitions
        lines.append("\\n## Relevant Definitions")
        # Find definitions near the focus line
        nearby_symbols = [
            s for s in self.symbol_table.symbols.values()
            if abs(s.defined_at - focus_line) < 20
        ]
        for symbol in nearby_symbols:
            lines.append(f"- {symbol.name} at line {symbol.defined_at}")
        
        # Add import context
        lines.append("\\n## Available Imports")
        for symbol in self.symbol_table.symbols.values():
            if symbol.type == SymbolType.IMPORT:
                lines.append(f"- {symbol.name} from {symbol.metadata.get('module', 'unknown')}")
        
        return "\\n".join(lines)

# Test the complete system
code = """
from typing import List

class DataProcessor:
    def __init__(self, config: dict):
        self.config = config
        self.results = []
    
    def process(self, items: List[int]) -> List[int]:
        processed = []
        for item in items:
            if self.validate(item):
                result = self.transform(item)
                processed.append(result)
                self.results.append(result)
        return processed
    
    def validate(self, item: int) -> bool:
        return item > 0
    
    def transform(self, item: int) -> int:
        return item * 2

processor = DataProcessor({'strict': True})
output = processor.process([1, 2, 3, 4, 5])
"""

resolver = ComprehensiveSymbolResolver(code, 'example.py')

# Test resolution at different positions
test_cases = [
    (11, 19),  # 'validate' method call
    (14, 27),  # 'result' variable
    (23, 12),  # 'DataProcessor' class usage
]

print("=== Symbol Resolution Examples ===\\n")
for line, col in test_cases:
    result = resolver.resolve_at_cursor(line, col)
    code_line = code.split('\\n')[line - 1]
    print(f"Position ({line}, {col}): {code_line.strip()}")
    print(f"Result: {result}")
    print()

# Generate LLM context for line 12
print("\\n=== LLM Context ===")
print(resolver.generate_context_for_llm(12))
\`\`\`

**Exercise Tasks:**
1. Add support for method resolution (obj.method)
2. Implement scope-aware resolution
3. Add type inference for better resolution
4. Build an index for faster lookups
5. Support cross-file symbol resolution

## Common Pitfalls

### 1. Not Following Scope Chain

\`\`\`python
# ❌ Wrong: Only looks in current scope
def resolve(name):
    return current_scope.get(name)

# ✅ Correct: Follow scope chain
def resolve(name):
    scope = current_scope
    while scope:
        if name in scope:
            return scope[name]
        scope = scope.parent
    return None
\`\`\`

### 2. Ignoring Import Aliases

\`\`\`python
# ❌ Wrong: Assumes original name
import pandas as pd
# Looking for 'pandas' won't find it

# ✅ Correct: Track aliases
imports = {'pd': 'pandas'}
# Now 'pd' resolves correctly
\`\`\`

### 3. Missing Attribute Resolution

\`\`\`python
# ❌ Wrong: Only resolves names
def resolve(node):
    if isinstance(node, ast.Name):
        return resolve_name(node.id)
    # Doesn't handle: obj.attr

# ✅ Correct: Handle attributes too
def resolve(node):
    if isinstance(node, ast.Name):
        return resolve_name(node.id)
    elif isinstance(node, ast.Attribute):
        return resolve_attribute(node)
\`\`\`

### 4. Not Tracking Symbol Scope Changes

\`\`\`python
# ❌ Wrong: Doesn't track scope properly
def analyze_function(func):
    for stmt in func.body:
        analyze(stmt)  # Still in same scope!

# ✅ Correct: Push/pop scopes
def analyze_function(func):
    push_scope(func.name)
    for stmt in func.body:
        analyze(stmt)
    pop_scope()
\`\`\`

## Production Checklist

### Symbol Resolution
- [ ] Build comprehensive symbol tables
- [ ] Follow scope chains correctly
- [ ] Handle all definition types
- [ ] Track import aliases
- [ ] Resolve attributes and methods
- [ ] Support nested scopes

### Cross-References
- [ ] Track all symbol definitions
- [ ] Record all symbol references
- [ ] Distinguish read/write/call
- [ ] Support position-based queries
- [ ] Handle shadowed names

### Performance
- [ ] Index symbols for fast lookup
- [ ] Cache resolution results
- [ ] Support incremental updates
- [ ] Optimize for large files
- [ ] Profile bottlenecks

### Accuracy
- [ ] Test with complex scoping
- [ ] Handle edge cases (closures, comprehensions)
- [ ] Validate resolution correctness
- [ ] Test with real codebases
- [ ] Compare with LSP servers

### Integration
- [ ] Provide clean API
- [ ] Support position queries
- [ ] Return structured results
- [ ] Enable batch operations
- [ ] Document behavior

## Summary

Symbol resolution and references are core to code understanding:

- **Symbol Tables**: Map names to definitions
- **Scope Resolution**: Follow scope chains correctly
- **Cross-References**: Track all symbol usage
- **Go-to-Definition**: Jump to symbol definitions
- **Find References**: Locate all symbol usage
- **Import Resolution**: Understand external dependencies

These capabilities enable Cursor to provide intelligent features like auto-completion, go-to-definition, find references, and rename symbol—all essential for productive coding with AI assistance.

In the next section, we'll explore code modification with AST—actually changing code programmatically while maintaining correctness.
`,
};

export default symbolResolution;
