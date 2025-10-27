const treeSitterParsing = {
  id: 'tree-sitter-parsing',
  title: 'Tree-sitter for Multi-Language Parsing',
  content: `
# Tree-sitter for Multi-Language Parsing

## Introduction

While Python\'s \`ast\` module is excellent for Python code, modern AI coding tools like Cursor need to understand code in dozens of languages—JavaScript, TypeScript, Go, Rust, Java, C++, and more. This is where tree-sitter becomes invaluable.

Tree-sitter is a parser generator and incremental parsing library that can parse source code in any language into a concrete syntax tree. It's used by GitHub, Atom, Neovim, and yes—tools like Cursor—to provide fast, accurate, multi-language code understanding.

**Why Tree-sitter for AI Code Tools:**

Cursor uses tree-sitter because it:
- Supports 40+ programming languages with consistent API
- Parses incrementally (only re-parses changed code)
- Recovers from syntax errors gracefully
- Provides precise syntax trees with location info
- Is extremely fast (written in C)
- Offers powerful query language for pattern matching

This allows Cursor to analyze any codebase regardless of programming language using a single, unified approach.

## Deep Technical Explanation

### Tree-sitter vs Language-Specific Parsers

**Language-Specific Parsers (like Python's ast):**
- One parser per language
- Deep language integration
- Abstract syntax trees (simplified)
- Language-specific APIs
- Best for single-language tools

**Tree-sitter:**
- Universal parsing system
- Consistent API across languages
- Concrete syntax trees (complete)
- Uniform query language
- Best for multi-language tools

### Tree-sitter Architecture

**1. Parser Generation:**
Tree-sitter uses grammar files (written in JavaScript) to generate parsers:

\`\`\`javascript
// Example: Simplified Python grammar
module.exports = grammar({
  name: 'python',
  rules: {
    module: $ => repeat($._statement),
    
    _statement: $ => choice(
      $.function_definition,
      $.class_definition,
      $.expression_statement,
      // ... more statement types
    ),
    
    function_definition: $ => seq(
      'def',
      field('name', $.identifier),
      field('parameters', $.parameters),
      ':',
      field('body', $.block)
    ),
    // ... more rules
  }
});
\`\`\`

**2. Incremental Parsing:**
Tree-sitter tracks changes and only re-parses affected code:

\`\`\`
Initial parse: 100ms for 10,000 lines
Edit one line: 2ms to update tree
Edit another: 1ms to update tree
\`\`\`

This makes it perfect for real-time code analysis in editors.

**3. Error Recovery:**
Unlike traditional parsers that fail on syntax errors, tree-sitter creates a best-effort tree with ERROR nodes:

\`\`\`python
def calculate (x, y):
    result = x + y  # Missing closing paren
    return result
\`\`\`

Tree-sitter will still parse this, marking the error but preserving structure around it.

### Concrete Syntax Tree vs Abstract Syntax Tree

**Concrete Syntax Tree (CST) - What tree-sitter produces:**
- Includes all tokens (parentheses, semicolons, commas)
- Preserves exact source structure
- Includes whitespace information
- Can perfectly reconstruct source code
- Larger and more detailed

**Abstract Syntax Tree (AST) - What language parsers produce:**
- Abstracts away syntactic details
- Focuses on semantic meaning
- Smaller and simpler
- Loses some formatting information

\`\`\`javascript
// For: const x = (1 + 2) * 3;

// CST (tree-sitter):
program
  ├── lexical_declaration
  │   ├── 'const'
  │   ├── variable_declarator
  │   │   ├── identifier: 'x'
  │   │   ├── '='
  │   │   └── binary_expression
  │   │       ├── parenthesized_expression
  │   │       │   ├── '('
  │   │       │   ├── binary_expression
  │   │       │   │   ├── number: '1'
  │   │       │   │   ├── '+'
  │   │       │   │   └── number: '2'
  │   │       │   └── ')'
  │   │       ├── '*'
  │   │       └── number: '3'
  │   └── ';'

// AST (simplified):
Program
  └── VariableDeclaration (const)
      └── VariableDeclarator
          ├── name: 'x'
          └── init: BinaryExpression (*)
              ├── left: BinaryExpression (+)
              │   ├── left: Literal (1)
              │   └── right: Literal (2)
              └── right: Literal (3)
\`\`\`

### Tree-sitter Query Language

Tree-sitter provides an S-expression-based query language for pattern matching:

\`\`\`scheme
;; Find all function definitions in Python
(function_definition
  name: (identifier) @function.name
  parameters: (parameters) @function.params
  body: (block) @function.body)

;; Find all class methods
(class_definition
  body: (block
    (function_definition
      name: (identifier) @method.name)))

;; Find async functions
(function_definition
  (async) @async.keyword
  name: (identifier) @async.function.name)
\`\`\`

This query language is used extensively in editors for syntax highlighting, code folding, and structural navigation.

## Code Implementation

### Basic Tree-sitter Setup and Parsing

\`\`\`python
from tree_sitter import Language, Parser
import tree_sitter_python as tspython
import tree_sitter_javascript as tsjavascript
import tree_sitter_typescript as tstypescript

# Build language objects
PY_LANGUAGE = Language (tspython.language())
JS_LANGUAGE = Language (tsjavascript.language())
TS_LANGUAGE = Language (tstypescript.language_typescript())

def parse_code (code: str, language: Language) -> 'Tree':
    """
    Parse code using tree-sitter.
    
    Args:
        code: Source code string
        language: Tree-sitter language object
        
    Returns:
        Parse tree
    """
    parser = Parser()
    parser.set_language (language)
    
    # Parse (returns Tree object)
    tree = parser.parse (bytes (code, 'utf8'))
    return tree

# Example: Parse Python code
python_code = """
def calculate_total (items, tax_rate=0.1):
    ''Calculate total with tax.''
    subtotal = sum (item['price'] for item in items)
    tax = subtotal * tax_rate
    return subtotal + tax

class ShoppingCart:
    def __init__(self):
        self.items = []
    
    def add_item (self, item):
        self.items.append (item)
"""

tree = parse_code (python_code, PY_LANGUAGE)
print(f"Root node: {tree.root_node.type}")
print(f"Children: {len (tree.root_node.children)}")
print(f"Has errors: {tree.root_node.has_error}")

# Example: Parse JavaScript code
javascript_code = """
function calculateTotal (items, taxRate = 0.1) {
    const subtotal = items.reduce((sum, item) => sum + item.price, 0);
    const tax = subtotal * taxRate;
    return subtotal + tax;
}

class ShoppingCart {
    constructor() {
        this.items = [];
    }
    
    addItem (item) {
        this.items.push (item);
    }
}
"""

js_tree = parse_code (javascript_code, JS_LANGUAGE)
print(f"\\nJavaScript root: {js_tree.root_node.type}")
print(f"Children: {len (js_tree.root_node.children)}")
\`\`\`

### Traversing Tree-sitter Trees

\`\`\`python
from tree_sitter import Node, Tree
from typing import List, Iterator

def traverse_tree (node: Node, depth: int = 0) -> Iterator[tuple]:
    """
    Traverse tree-sitter tree depth-first.
    Yields (node, depth) tuples.
    """
    yield (node, depth)
    for child in node.children:
        yield from traverse_tree (child, depth + 1)

def print_tree (node: Node, code_bytes: bytes, max_depth: int = 3):
    """
    Print a visual representation of parse tree.
    
    Args:
        node: Root node to print from
        code_bytes: Source code as bytes
        max_depth: Maximum depth to print
    """
    for node, depth in traverse_tree (node):
        if depth > max_depth:
            continue
            
        indent = "  " * depth
        node_text = ""
        
        # For leaf nodes, show the actual text
        if len (node.children) == 0:
            start = node.start_byte
            end = node.end_byte
            node_text = code_bytes[start:end].decode('utf8')
            # Truncate long text
            if len (node_text) > 30:
                node_text = node_text[:27] + "..."
            node_text = f" → '{node_text}'"
        
        print(f"{indent}{node.type}{node_text}")

# Example usage
code = """
def greet (name: str) -> str:
    return f"Hello, {name}!"
"""

tree = parse_code (code, PY_LANGUAGE)
code_bytes = bytes (code, 'utf8')

print("=== Tree Structure ===")
print_tree (tree.root_node, code_bytes)
\`\`\`

### Finding Specific Nodes

\`\`\`python
from tree_sitter import Node
from typing import List

def find_nodes_by_type (node: Node, node_type: str) -> List[Node]:
    """
    Find all nodes of a specific type.
    
    Args:
        node: Root node to search from
        node_type: Type of nodes to find
        
    Returns:
        List of matching nodes
    """
    results = []
    
    def walk (n: Node):
        if n.type == node_type:
            results.append (n)
        for child in n.children:
            walk (child)
    
    walk (node)
    return results

def get_node_text (node: Node, code_bytes: bytes) -> str:
    """Extract text for a node."""
    return code_bytes[node.start_byte:node.end_byte].decode('utf8')

# Example: Find all function definitions
python_code = """
def function_one():
    pass

class MyClass:
    def method_one (self):
        pass
    
    def method_two (self, x):
        return x * 2

def function_two (a, b):
    return a + b
"""

tree = parse_code (python_code, PY_LANGUAGE)
code_bytes = bytes (python_code, 'utf8')

# Find all function definitions
functions = find_nodes_by_type (tree.root_node, 'function_definition')

print(f"Found {len (functions)} functions:\\n")
for func in functions:
    # Get function name (first identifier child)
    name_node = func.child_by_field_name('name')
    if name_node:
        name = get_node_text (name_node, code_bytes)
        line = func.start_point[0] + 1
        print(f"  {name} at line {line}")
\`\`\`

### Multi-Language Function Extractor

\`\`\`python
from dataclasses import dataclass
from typing import List, Optional
from tree_sitter import Language, Parser, Node

@dataclass
class FunctionInfo:
    name: str
    start_line: int
    end_line: int
    language: str
    signature: str

class MultiLanguageFunctionExtractor:
    """
    Extract function definitions from multiple languages.
    This is how Cursor understands functions across your codebase.
    """
    
    # Map language to function definition node types
    FUNCTION_TYPES = {
        'python': 'function_definition',
        'javascript': 'function_declaration',
        'typescript': 'function_declaration',
        'java': 'method_declaration',
        'go': 'function_declaration',
        'rust': 'function_item',
        'cpp': 'function_definition',
    }
    
    def __init__(self):
        self.languages = {
            'python': Language (tspython.language()),
            'javascript': Language (tsjavascript.language()),
            'typescript': Language (tstypescript.language_typescript()),
        }
        self.parsers = {
            lang: self._create_parser (lang_obj)
            for lang, lang_obj in self.languages.items()
        }
    
    def _create_parser (self, language: Language) -> Parser:
        """Create parser for a language."""
        parser = Parser()
        parser.set_language (language)
        return parser
    
    def extract_functions (self, code: str, language: str) -> List[FunctionInfo]:
        """
        Extract all functions from code.
        
        Args:
            code: Source code
            language: Language name ('python', 'javascript', etc.)
            
        Returns:
            List of function information
        """
        if language not in self.parsers:
            raise ValueError (f"Unsupported language: {language}")
        
        parser = self.parsers[language]
        tree = parser.parse (bytes (code, 'utf8'))
        code_bytes = bytes (code, 'utf8')
        
        function_type = self.FUNCTION_TYPES.get (language)
        if not function_type:
            return []
        
        functions = []
        func_nodes = find_nodes_by_type (tree.root_node, function_type)
        
        for func_node in func_nodes:
            name = self._extract_function_name (func_node, language, code_bytes)
            signature = self._extract_signature (func_node, code_bytes)
            
            functions.append(FunctionInfo(
                name=name,
                start_line=func_node.start_point[0] + 1,
                end_line=func_node.end_point[0] + 1,
                language=language,
                signature=signature
            ))
        
        return functions
    
    def _extract_function_name (self, node: Node, language: str, code_bytes: bytes) -> str:
        """Extract function name from node."""
        # Most languages have a 'name' field
        name_node = node.child_by_field_name('name')
        if name_node:
            return get_node_text (name_node, code_bytes)
        
        # Fallback: look for identifier child
        for child in node.children:
            if 'identifier' in child.type:
                return get_node_text (child, code_bytes)
        
        return "anonymous"
    
    def _extract_signature (self, node: Node, code_bytes: bytes) -> str:
        """Extract full function signature."""
        # Get text from function keyword to body start
        for i, child in enumerate (node.children):
            if child.type == 'block' or child.type == 'statement_block':
                # Get everything before the body
                end_byte = child.start_byte
                signature_text = code_bytes[node.start_byte:end_byte].decode('utf8')
                return signature_text.strip()
        
        # Fallback: entire node text (first line)
        text = get_node_text (node, code_bytes)
        return text.split('\\n')[0]

# Example: Extract functions from multiple languages
extractor = MultiLanguageFunctionExtractor()

# Python code
py_code = """
def calculate (x, y):
    return x + y

async def fetch_data (url):
    return await request (url)

class Helper:
    def process (self, data):
        return data * 2
"""

# JavaScript code
js_code = """
function calculate (x, y) {
    return x + y;
}

async function fetchData (url) {
    return await fetch (url);
}

class Helper {
    process (data) {
        return data * 2;
    }
}
"""

print("=== Python Functions ===")
for func in extractor.extract_functions (py_code, 'python'):
    print(f"{func.name} (lines {func.start_line}-{func.end_line})")
    print(f"  Signature: {func.signature}")

print("\\n=== JavaScript Functions ===")
for func in extractor.extract_functions (js_code, 'javascript'):
    print(f"{func.name} (lines {func.start_line}-{func.end_line})")
    print(f"  Signature: {func.signature}")
\`\`\`

### Using Tree-sitter Queries

\`\`\`python
from tree_sitter import Language, Parser, Query

def query_code (code: str, language: Language, query_string: str) -> List:
    """
    Execute tree-sitter query on code.
    
    Args:
        code: Source code
        language: Tree-sitter language
        query_string: Query in S-expression format
        
    Returns:
        Query results
    """
    parser = Parser()
    parser.set_language (language)
    tree = parser.parse (bytes (code, 'utf8'))
    
    query = language.query (query_string)
    captures = query.captures (tree.root_node)
    
    code_bytes = bytes (code, 'utf8')
    results = []
    
    for node, capture_name in captures:
        text = get_node_text (node, code_bytes)
        results.append({
            'capture': capture_name,
            'text': text,
            'line': node.start_point[0] + 1
        })
    
    return results

# Example: Find all async functions in Python
python_code = """
async def fetch_user (user_id):
    return await db.get_user (user_id)

def process_data (data):
    return data * 2

async def save_user (user):
    await db.save (user)
"""

# Query for async functions
async_query = """
(function_definition
  (async) @async_keyword
  name: (identifier) @function_name)
"""

results = query_code (python_code, PY_LANGUAGE, async_query)

print("=== Async Functions ===")
for result in results:
    if result['capture'] == 'function_name':
        print(f"  {result['text']} at line {result['line']}")

# Example: Find all class methods
class_method_query = """
(class_definition
  body: (block
    (function_definition
      name: (identifier) @method_name)))
"""

code_with_classes = """
class UserService:
    def get_user (self, user_id):
        pass
    
    def save_user (self, user):
        pass

class ProductService:
    def get_product (self, product_id):
        pass
"""

results = query_code (code_with_classes, PY_LANGUAGE, class_method_query)

print("\\n=== Class Methods ===")
for result in results:
    print(f"  {result['text']} at line {result['line']}")
\`\`\`

### Error Recovery Example

\`\`\`python
def analyze_with_errors (code: str, language: Language):
    """
    Demonstrate tree-sitter's error recovery.
    It can still parse and analyze code with syntax errors.
    """
    parser = Parser()
    parser.set_language (language)
    tree = parser.parse (bytes (code, 'utf8'))
    
    code_bytes = bytes (code, 'utf8')
    
    print("=== Parse Result ===")
    print(f"Has errors: {tree.root_node.has_error}")
    
    # Find ERROR nodes
    def find_errors (node: Node, errors: List):
        if node.type == 'ERROR' or node.is_missing:
            errors.append({
                'line': node.start_point[0] + 1,
                'col': node.start_point[1],
                'text': get_node_text (node, code_bytes)
            })
        for child in node.children:
            find_errors (child, errors)
    
    errors = []
    find_errors (tree.root_node, errors)
    
    if errors:
        print("\\n=== Syntax Errors ===")
        for error in errors:
            print(f"  Line {error['line']}, Column {error['col']}")
            print(f"  Near: {error['text'][:50]}")
    
    # Still try to extract functions
    functions = find_nodes_by_type (tree.root_node, 'function_definition')
    print(f"\\n=== Functions Found (despite errors): {len (functions)} ===")
    for func in functions:
        name_node = func.child_by_field_name('name')
        if name_node:
            print(f"  {get_node_text (name_node, code_bytes)}")

# Code with syntax errors
buggy_code = """
def valid_function (x, y):
    return x + y

def missing_colon (x, y)  # Missing colon!
    return x + y

def unclosed_paren (x, y:
    return x + y

def another_valid():
    return 42
"""

analyze_with_errors (buggy_code, PY_LANGUAGE)
\`\`\`

### Incremental Parsing

\`\`\`python
from tree_sitter import Parser, Language

class IncrementalCodeAnalyzer:
    """
    Demonstrate incremental parsing.
    Only re-parses changed portions of code.
    """
    
    def __init__(self, language: Language):
        self.parser = Parser()
        self.parser.set_language (language)
        self.code = ""
        self.tree = None
    
    def set_code (self, code: str):
        """Set initial code and parse."""
        self.code = code
        self.tree = self.parser.parse (bytes (code, 'utf8'))
        print(f"Initial parse complete: {len (self.code)} bytes")
    
    def edit_code (self, start_byte: int, old_end_byte: int, new_text: str):
        """
        Edit code and re-parse incrementally.
        
        Args:
            start_byte: Where edit starts
            old_end_byte: Where edit ended (in old code)
            new_text: New text to insert
        """
        # Update code
        old_code = self.code
        self.code = (
            self.code[:start_byte] + 
            new_text + 
            self.code[old_end_byte:]
        )
        
        # Calculate position info
        start_point = self._byte_to_point (old_code, start_byte)
        old_end_point = self._byte_to_point (old_code, old_end_byte)
        new_end_byte = start_byte + len (new_text)
        new_end_point = self._byte_to_point (self.code, new_end_byte)
        
        # Edit tree
        self.tree.edit(
            start_byte=start_byte,
            old_end_byte=old_end_byte,
            new_end_byte=new_end_byte,
            start_point=start_point,
            old_end_point=old_end_point,
            new_end_point=new_end_point,
        )
        
        # Re-parse (only changed regions)
        import time
        start = time.time()
        self.tree = self.parser.parse (bytes (self.code, 'utf8'), self.tree)
        elapsed = (time.time() - start) * 1000
        print(f"Incremental re-parse: {elapsed:.2f}ms")
    
    def _byte_to_point (self, code: str, byte_offset: int) -> tuple:
        """Convert byte offset to (row, column) point."""
        text_before = code[:byte_offset]
        lines = text_before.split('\\n')
        row = len (lines) - 1
        col = len (lines[-1])
        return (row, col)

# Example usage
analyzer = IncrementalCodeAnalyzer(PY_LANGUAGE)

initial_code = """
def calculate (x, y):
    result = x + y
    return result

def process (data):
    return data * 2
"""

analyzer.set_code (initial_code)

# Edit: Change "x + y" to "x + y + 10"
# Find position of "x + y" in the first function
start = initial_code.index("x + y")
old_end = start + len("x + y")
new_text = "x + y + 10"

analyzer.edit_code (start, old_end, new_text)

print(f"\\nNew code:\\n{analyzer.code}")
\`\`\`

## Real-World Case Study: How Cursor Uses Tree-sitter

Cursor leverages tree-sitter for multi-language support:

**1. Universal Code Understanding:**
\`\`\`
When you open a file in Cursor:
1. Detect language from extension
2. Load appropriate tree-sitter parser
3. Parse entire file into syntax tree
4. Extract structure (functions, classes, imports)
5. Build context for AI suggestions
\`\`\`

**2. Real-Time Analysis:**
\`\`\`
As you type:
1. Tree-sitter incrementally updates parse tree
2. Only changed regions are re-parsed (< 5ms)
3. Cursor immediately sees new structure
4. AI suggestions update in real-time
\`\`\`

**3. Cross-Language Patterns:**
\`\`\`javascript
// Cursor can find similar patterns across languages

// JavaScript
function calculateTotal (items) {
    return items.reduce((sum, item) => sum + item.price, 0);
}

// Python equivalent Cursor can suggest
def calculate_total (items):
    return sum (item['price'] for item in items)
\`\`\`

**4. Syntax-Aware Editing:**

When Cursor generates code, it uses tree-sitter to:
- Ensure correct indentation
- Match existing code style
- Place code in correct locations (inside class, at module level, etc.)
- Preserve comments and structure

**5. Multi-File Context:**

Tree-sitter allows Cursor to quickly:
- Parse all files in your project
- Build a complete code graph
- Find function definitions across files
- Understand import relationships
- Provide relevant context to the LLM

## Hands-On Exercise

Build a multi-language code analyzer for AI context:

\`\`\`python
from dataclasses import dataclass
from typing import List, Dict, Optional
from tree_sitter import Language, Parser, Node
import tree_sitter_python as tspython
import tree_sitter_javascript as tsjavascript

@dataclass
class CodeSymbol:
    type: str  # 'function', 'class', 'method'
    name: str
    line: int
    language: str
    signature: Optional[str] = None
    parent: Optional[str] = None

class UniversalCodeAnalyzer:
    """
    Analyze code in any language to extract symbols.
    This is the foundation of tools like Cursor.
    """
    
    LANGUAGES = {
        'python': {
            'parser': Language (tspython.language()),
            'function': 'function_definition',
            'class': 'class_definition',
        },
        'javascript': {
            'parser': Language (tsjavascript.language()),
            'function': 'function_declaration',
            'class': 'class_declaration',
        },
    }
    
    def __init__(self):
        self.parsers = {}
        for lang, config in self.LANGUAGES.items():
            parser = Parser()
            parser.set_language (config['parser'])
            self.parsers[lang] = parser
    
    def analyze_file (self, code: str, language: str) -> List[CodeSymbol]:
        """
        Analyze a file and extract all symbols.
        
        Returns structure suitable for LLM context.
        """
        if language not in self.parsers:
            return []
        
        parser = self.parsers[language]
        tree = parser.parse (bytes (code, 'utf8'))
        code_bytes = bytes (code, 'utf8')
        
        symbols = []
        config = self.LANGUAGES[language]
        
        # Find classes
        class_nodes = find_nodes_by_type (tree.root_node, config['class'])
        for class_node in class_nodes:
            class_name = self._get_name (class_node, code_bytes)
            symbols.append(CodeSymbol(
                type='class',
                name=class_name,
                line=class_node.start_point[0] + 1,
                language=language
            ))
            
            # Find methods in class
            methods = find_nodes_by_type (class_node, config['function'])
            for method in methods:
                method_name = self._get_name (method, code_bytes)
                symbols.append(CodeSymbol(
                    type='method',
                    name=method_name,
                    line=method.start_point[0] + 1,
                    language=language,
                    parent=class_name
                ))
        
        # Find top-level functions
        func_nodes = find_nodes_by_type (tree.root_node, config['function'])
        for func_node in func_nodes:
            # Skip if it's a method (inside a class)
            if self._is_inside_class (func_node, class_nodes):
                continue
            
            func_name = self._get_name (func_node, code_bytes)
            symbols.append(CodeSymbol(
                type='function',
                name=func_name,
                line=func_node.start_point[0] + 1,
                language=language
            ))
        
        return symbols
    
    def _get_name (self, node: Node, code_bytes: bytes) -> str:
        """Extract name from function or class node."""
        name_node = node.child_by_field_name('name')
        if name_node:
            return get_node_text (name_node, code_bytes)
        return "anonymous"
    
    def _is_inside_class (self, func_node: Node, class_nodes: List[Node]) -> bool:
        """Check if function is inside any of the class nodes."""
        func_start = func_node.start_byte
        func_end = func_node.end_byte
        
        for class_node in class_nodes:
            if class_node.start_byte < func_start < func_end < class_node.end_byte:
                return True
        return False
    
    def generate_context_summary (self, symbols: List[CodeSymbol]) -> str:
        """
        Generate text summary for LLM context.
        This is what you'd send to an AI like Cursor\'s backend.
        """
        lines = ["# Code Structure\\n"]
        
        # Group by type
        classes = [s for s in symbols if s.type == 'class']
        functions = [s for s in symbols if s.type == 'function']
        methods = [s for s in symbols if s.type == 'method']
        
        if classes:
            lines.append("## Classes")
            for cls in classes:
                lines.append (f"- {cls.name} (line {cls.line})")
                class_methods = [m for m in methods if m.parent == cls.name]
                if class_methods:
                    for method in class_methods:
                        lines.append (f"  - {method.name}() (line {method.line})")
            lines.append("")
        
        if functions:
            lines.append("## Functions")
            for func in functions:
                lines.append (f"- {func.name}() (line {func.line})")
            lines.append("")
        
        return "\\n".join (lines)

# Test the analyzer
analyzer = UniversalCodeAnalyzer()

python_code = """
class UserService:
    def get_user (self, user_id):
        return db.query (user_id)
    
    def save_user (self, user):
        db.save (user)

def authenticate (username, password):
    user = UserService().get_user (username)
    return verify_password (user, password)
"""

javascript_code = """
class UserService {
    getUser (userId) {
        return db.query (userId);
    }
    
    saveUser (user) {
        db.save (user);
    }
}

function authenticate (username, password) {
    const user = new UserService().getUser (username);
    return verifyPassword (user, password);
}
"""

print("=== Python Analysis ===")
py_symbols = analyzer.analyze_file (python_code, 'python')
print(analyzer.generate_context_summary (py_symbols))

print("=== JavaScript Analysis ===")
js_symbols = analyzer.analyze_file (javascript_code, 'javascript')
print(analyzer.generate_context_summary (js_symbols))
\`\`\`

**Exercise Tasks:**1. Add support for more languages (TypeScript, Go, Java)
2. Extract function parameters and return types
3. Build a dependency graph showing which functions call which
4. Add query-based pattern matching for specific constructs
5. Implement incremental re-analysis for file edits

## Common Pitfalls

### 1. Not Handling Different Node Types Per Language

\`\`\`python
# ❌ Wrong: Assumes same node types
def find_functions (tree):
    return find_nodes_by_type (tree.root_node, 'function_definition')
    # Doesn't work for JavaScript!

# ✅ Correct: Language-specific node types
FUNCTION_TYPES = {
    'python': 'function_definition',
    'javascript': 'function_declaration',
    'java': 'method_declaration',
}

def find_functions (tree, language):
    node_type = FUNCTION_TYPES.get (language)
    if not node_type:
        return []
    return find_nodes_by_type (tree.root_node, node_type)
\`\`\`

### 2. Forgetting to Convert Bytes

\`\`\`python
# ❌ Wrong: Passing string to parse
tree = parser.parse (code)  # Error!

# ✅ Correct: Convert to bytes
tree = parser.parse (bytes (code, 'utf8'))
\`\`\`

### 3. Not Checking for None

\`\`\`python
# ❌ Wrong: Assumes field exists
name = node.child_by_field_name('name').text

# ✅ Correct: Check for None
name_node = node.child_by_field_name('name')
if name_node:
    name = get_node_text (name_node, code_bytes)
\`\`\`

### 4. Ignoring Error Recovery

\`\`\`python
# ❌ Wrong: Assuming no errors
tree = parser.parse (bytes (code, 'utf8'))
functions = find_nodes_by_type (tree.root_node, 'function_definition')
# May have ERROR nodes mixed in!

# ✅ Correct: Check for errors
tree = parser.parse (bytes (code, 'utf8'))
if tree.root_node.has_error:
    print("Warning: Parse errors detected")
# Filter out ERROR nodes
functions = [
    n for n in find_nodes_by_type (tree.root_node, 'function_definition')
    if not n.has_error
]
\`\`\`

### 5. Inefficient Tree Traversal

\`\`\`python
# ❌ Wrong: Re-walking tree multiple times
functions = find_nodes_by_type (root, 'function_definition')
classes = find_nodes_by_type (root, 'class_definition')
imports = find_nodes_by_type (root, 'import_statement')

# ✅ Correct: Walk once, collect all
def collect_nodes (root, types_wanted):
    found = {t: [] for t in types_wanted}
    for node, _ in traverse_tree (root):
        if node.type in types_wanted:
            found[node.type].append (node)
    return found

nodes = collect_nodes (root, ['function_definition', 'class_definition', 'import_statement'])
\`\`\`

## Production Checklist

### Language Support
- [ ] Install tree-sitter parsers for all target languages
- [ ] Map language file extensions to parsers
- [ ] Handle language-specific node types
- [ ] Test with real-world code samples per language
- [ ] Document supported languages

### Error Handling
- [ ] Handle parse errors gracefully
- [ ] Identify and report ERROR nodes
- [ ] Provide partial results even with errors
- [ ] Log parsing failures
- [ ] Include error context in reports

### Performance
- [ ] Use incremental parsing for edits
- [ ] Cache parsed trees for unchanged files
- [ ] Profile parsing performance per language
- [ ] Set timeouts for very large files
- [ ] Consider memory usage with many trees

### Query System
- [ ] Build library of useful queries per language
- [ ] Test queries with various code patterns
- [ ] Handle query errors
- [ ] Document query syntax
- [ ] Provide query examples

### Integration
- [ ] Provide consistent API across languages
- [ ] Export results in standard formats (JSON, dict)
- [ ] Support batch analysis of multiple files
- [ ] Enable language-agnostic consumers
- [ ] Version parser grammar updates

## Summary

Tree-sitter enables universal code understanding:

- **Multi-Language**: Parse 40+ languages with consistent API
- **Incremental**: Fast re-parsing of only changed code
- **Error Recovery**: Works even with syntax errors
- **Query Language**: Powerful pattern matching
- **Production Ready**: Used by GitHub, Atom, Neovim, and Cursor

Understanding tree-sitter is essential for building multi-language AI coding tools. It provides the foundation for Cursor\'s ability to understand any codebase, regardless of programming language, and offer intelligent suggestions across your entire project.

In the next section, we'll explore how to analyze code structure systematically to extract architectural patterns and relationships.
`,
};

export default treeSitterParsing;
