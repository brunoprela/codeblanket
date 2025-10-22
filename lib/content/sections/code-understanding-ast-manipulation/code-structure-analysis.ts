export default {
    id: 'code-structure-analysis',
    title: 'Code Structure Analysis',
    content: `
# Code Structure Analysis

## Introduction

Understanding code structure goes beyond parsing individual functions and classes. It involves analyzing how code is organized, identifying patterns, finding relationships between components, and building a comprehensive map of your codebase. This is exactly what Cursor does when it provides intelligent suggestions that understand your project's architecture.

**Why Structure Analysis Matters:**

When you ask Cursor "add error handling to all API calls," it needs to:
- Identify all functions that make API calls
- Understand the control flow in each function
- Find where exceptions should be caught
- Recognize existing error handling patterns
- Generate consistent error handling across the codebase

All of this requires sophisticated structure analysis that we'll master in this section.

## Deep Technical Explanation

### Levels of Code Structure

**1. Syntax Level (what we've covered):**
- Individual nodes (functions, classes, statements)
- Local scope and variables
- Immediate parent-child relationships

**2. Semantic Level (this section):**
- Code blocks and their purposes
- Control flow patterns (if/else, loops, try/except)
- Data flow (where values come from and go to)
- Call graphs (which functions call which)
- Dependency graphs (what depends on what)

**3. Architectural Level (next sections):**
- Module relationships
- Package structure
- Design patterns
- Cross-file dependencies

### Control Flow Analysis

Control flow analysis tracks the possible execution paths through code:

\`\`\`python
def process_payment(amount, card):
    if amount <= 0:
        return None  # Path 1: Early return
    
    if not validate_card(card):
        raise ValueError("Invalid card")  # Path 2: Exception
    
    try:
        charge(card, amount)  # Path 3a: Success
        return True
    except PaymentError:
        log_error()  # Path 3b: Error handling
        return False

# Control flow graph:
# START → amount check → card validation → try charge → SUCCESS
#           ↓               ↓                   ↓
#         return None    exception          exception → log → return False
\`\`\`

### Data Flow Analysis

Data flow tracks how data moves through code:

\`\`\`python
def calculate_price(base_price, tax_rate, discount):
    # Data flow:
    # base_price → subtotal (line 2)
    # tax_rate, subtotal → tax (line 3)
    # discount, subtotal → discount_amount (line 4)
    # subtotal, tax, discount_amount → total (line 5)
    
    subtotal = base_price
    tax = subtotal * tax_rate
    discount_amount = subtotal * discount
    total = subtotal + tax - discount_amount
    return total
\`\`\`

### Dependency Analysis

Understanding what code depends on what:

\`\`\`python
# File: services/user_service.py
from database import db
from models.user import User
from utils.validation import validate_email

class UserService:
    def create_user(self, email, name):
        # Depends on:
        # - validate_email (utils.validation)
        # - User (models.user)
        # - db (database)
        if not validate_email(email):
            raise ValueError("Invalid email")
        user = User(email=email, name=name)
        db.save(user)
        return user

# Dependency graph:
# UserService → validate_email
# UserService → User
# UserService → db
\`\`\`

## Code Implementation

### Control Flow Extractor

\`\`\`python
import ast
from dataclasses import dataclass
from typing import List, Dict, Set, Optional
from enum import Enum

class FlowType(Enum):
    SEQUENCE = "sequence"
    BRANCH = "branch"
    LOOP = "loop"
    TRY_EXCEPT = "try_except"
    RETURN = "return"
    RAISE = "raise"

@dataclass
class ControlFlowNode:
    type: FlowType
    lineno: int
    condition: Optional[str] = None
    children: List['ControlFlowNode'] = None
    
    def __post_init__(self):
        if self.children is None:
            self.children = []

class ControlFlowAnalyzer(ast.NodeVisitor):
    """
    Analyze control flow in Python code.
    Builds a graph showing possible execution paths.
    """
    
    def __init__(self):
        self.functions: Dict[str, List[ControlFlowNode]] = {}
        self.current_function: Optional[str] = None
        self.flow_stack: List[ControlFlowNode] = []
    
    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Analyze control flow in a function."""
        self.current_function = node.name
        self.functions[node.name] = []
        
        # Analyze function body
        for stmt in node.body:
            flow_node = self._analyze_statement(stmt)
            if flow_node:
                self.functions[node.name].append(flow_node)
        
        self.current_function = None
        self.generic_visit(node)
    
    def _analyze_statement(self, stmt: ast.stmt) -> Optional[ControlFlowNode]:
        """Analyze a single statement for control flow."""
        
        if isinstance(stmt, ast.If):
            # Branch: if/elif/else
            condition = ast.unparse(stmt.test)
            node = ControlFlowNode(
                type=FlowType.BRANCH,
                lineno=stmt.lineno,
                condition=condition
            )
            
            # Analyze if body
            for s in stmt.body:
                child = self._analyze_statement(s)
                if child:
                    node.children.append(child)
            
            # Analyze else body
            if stmt.orelse:
                else_node = ControlFlowNode(
                    type=FlowType.BRANCH,
                    lineno=stmt.orelse[0].lineno,
                    condition="else"
                )
                for s in stmt.orelse:
                    child = self._analyze_statement(s)
                    if child:
                        else_node.children.append(child)
                node.children.append(else_node)
            
            return node
        
        elif isinstance(stmt, (ast.For, ast.While)):
            # Loop
            condition = None
            if isinstance(stmt, ast.While):
                condition = ast.unparse(stmt.test)
            elif isinstance(stmt, ast.For):
                condition = f"for {ast.unparse(stmt.target)} in {ast.unparse(stmt.iter)}"
            
            node = ControlFlowNode(
                type=FlowType.LOOP,
                lineno=stmt.lineno,
                condition=condition
            )
            
            # Analyze loop body
            for s in stmt.body:
                child = self._analyze_statement(s)
                if child:
                    node.children.append(child)
            
            return node
        
        elif isinstance(stmt, ast.Try):
            # Try/except
            node = ControlFlowNode(
                type=FlowType.TRY_EXCEPT,
                lineno=stmt.lineno,
                condition="try"
            )
            
            # Analyze try body
            for s in stmt.body:
                child = self._analyze_statement(s)
                if child:
                    node.children.append(child)
            
            # Analyze except handlers
            for handler in stmt.handlers:
                exc_type = "Exception"
                if handler.type:
                    exc_type = ast.unparse(handler.type)
                
                except_node = ControlFlowNode(
                    type=FlowType.TRY_EXCEPT,
                    lineno=handler.lineno,
                    condition=f"except {exc_type}"
                )
                
                for s in handler.body:
                    child = self._analyze_statement(s)
                    if child:
                        except_node.children.append(child)
                
                node.children.append(except_node)
            
            # Analyze finally
            if stmt.finalbody:
                finally_node = ControlFlowNode(
                    type=FlowType.TRY_EXCEPT,
                    lineno=stmt.finalbody[0].lineno,
                    condition="finally"
                )
                for s in stmt.finalbody:
                    child = self._analyze_statement(s)
                    if child:
                        finally_node.children.append(child)
                node.children.append(finally_node)
            
            return node
        
        elif isinstance(stmt, ast.Return):
            # Return statement
            value = "None"
            if stmt.value:
                value = ast.unparse(stmt.value)
            return ControlFlowNode(
                type=FlowType.RETURN,
                lineno=stmt.lineno,
                condition=f"return {value}"
            )
        
        elif isinstance(stmt, ast.Raise):
            # Raise statement
            exc = "Exception"
            if stmt.exc:
                exc = ast.unparse(stmt.exc)
            return ControlFlowNode(
                type=FlowType.RAISE,
                lineno=stmt.lineno,
                condition=f"raise {exc}"
            )
        
        return None
    
    def visualize_flow(self, function_name: str, indent: int = 0) -> str:
        """Create a text visualization of control flow."""
        if function_name not in self.functions:
            return f"Function '{function_name}' not found"
        
        lines = [f"Control flow for {function_name}:\\n"]
        
        def visualize_node(node: ControlFlowNode, depth: int):
            prefix = "  " * depth
            symbol = {
                FlowType.BRANCH: "├─ if",
                FlowType.LOOP: "↻ loop",
                FlowType.TRY_EXCEPT: "⚡ try",
                FlowType.RETURN: "← return",
                FlowType.RAISE: "⚠ raise",
                FlowType.SEQUENCE: "→",
            }.get(node.type, "→")
            
            condition_str = ""
            if node.condition:
                condition_str = f": {node.condition}"
            
            lines.append(f"{prefix}{symbol}{condition_str} (line {node.lineno})")
            
            for child in node.children:
                visualize_node(child, depth + 1)
        
        for node in self.functions[function_name]:
            visualize_node(node, indent)
        
        return "\\n".join(lines)
    
    def calculate_complexity(self, function_name: str) -> int:
        """
        Calculate cyclomatic complexity from control flow.
        Complexity = number of decision points + 1
        """
        if function_name not in self.functions:
            return 0
        
        complexity = 1  # Base complexity
        
        def count_decisions(node: ControlFlowNode):
            nonlocal complexity
            if node.type in [FlowType.BRANCH, FlowType.LOOP]:
                complexity += 1
            for child in node.children:
                count_decisions(child)
        
        for node in self.functions[function_name]:
            count_decisions(node)
        
        return complexity

# Example usage
code = """
def process_order(order, inventory):
    '''Process an order with various checks.'''
    if not order or not order.items:
        return None
    
    total = 0
    for item in order.items:
        if item.quantity <= 0:
            continue
        
        if item.id not in inventory:
            raise ValueError(f"Item {item.id} not in inventory")
        
        if inventory[item.id] < item.quantity:
            print(f"Insufficient stock for {item.id}")
            continue
        
        total += item.price * item.quantity
    
    try:
        charge_customer(order.customer, total)
    except PaymentError as e:
        log_error(e)
        return False
    
    return True
"""

analyzer = ControlFlowAnalyzer()
tree = ast.parse(code)
analyzer.visit(tree)

print(analyzer.visualize_flow('process_order'))
print(f"\\nCyclomatic Complexity: {analyzer.calculate_complexity('process_order')}")
\`\`\`

### Data Flow Analyzer

\`\`\`python
import ast
from dataclasses import dataclass
from typing import Dict, Set, List, Tuple

@dataclass
class Variable:
    name: str
    defined_at: int
    used_at: List[int]
    assigned_from: List[str]  # What expressions assigned to it

class DataFlowAnalyzer(ast.NodeVisitor):
    """
    Track data flow through functions.
    Understand where variables come from and where they go.
    """
    
    def __init__(self):
        self.variables: Dict[str, Variable] = {}
        self.current_line: int = 0
    
    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Analyze data flow in a function."""
        # Track parameters as initial variables
        for arg in node.args.args:
            self.variables[arg.arg] = Variable(
                name=arg.arg,
                defined_at=node.lineno,
                used_at=[],
                assigned_from=["parameter"]
            )
        
        # Analyze body
        self.generic_visit(node)
    
    def visit_Assign(self, node: ast.Assign):
        """Track variable assignments."""
        # Get assigned value as string
        value_str = ast.unparse(node.value)
        
        # Track what variables are used in the value
        used_vars = self._extract_variables(node.value)
        
        # Update used_at for those variables
        for var in used_vars:
            if var in self.variables:
                self.variables[var].used_at.append(node.lineno)
        
        # Track target variables
        for target in node.targets:
            if isinstance(target, ast.Name):
                var_name = target.id
                if var_name not in self.variables:
                    self.variables[var_name] = Variable(
                        name=var_name,
                        defined_at=node.lineno,
                        used_at=[],
                        assigned_from=[]
                    )
                self.variables[var_name].assigned_from.append(value_str)
        
        self.generic_visit(node)
    
    def visit_Name(self, node: ast.Name):
        """Track variable usage."""
        if isinstance(node.ctx, ast.Load):
            # Variable is being read
            if node.id in self.variables:
                self.variables[node.id].used_at.append(node.lineno)
        self.generic_visit(node)
    
    def _extract_variables(self, node: ast.expr) -> Set[str]:
        """Extract all variable names from an expression."""
        variables = set()
        for n in ast.walk(node):
            if isinstance(n, ast.Name):
                variables.add(n.id)
        return variables
    
    def find_unused_variables(self) -> List[str]:
        """Find variables that are defined but never used."""
        return [
            var.name for var in self.variables.values()
            if not var.used_at
        ]
    
    def find_variable_dependencies(self, var_name: str) -> Set[str]:
        """Find what variables a given variable depends on."""
        if var_name not in self.variables:
            return set()
        
        dependencies = set()
        for assignment in self.variables[var_name].assigned_from:
            # Parse the assignment to find variables
            try:
                expr = ast.parse(assignment, mode='eval')
                for node in ast.walk(expr):
                    if isinstance(node, ast.Name):
                        if node.id != var_name and node.id in self.variables:
                            dependencies.add(node.id)
            except:
                pass
        
        return dependencies
    
    def build_dependency_graph(self) -> Dict[str, Set[str]]:
        """Build a complete dependency graph."""
        graph = {}
        for var_name in self.variables:
            graph[var_name] = self.find_variable_dependencies(var_name)
        return graph
    
    def trace_variable(self, var_name: str) -> str:
        """Create a human-readable trace of a variable."""
        if var_name not in self.variables:
            return f"Variable '{var_name}' not found"
        
        var = self.variables[var_name]
        lines = [f"Variable: {var_name}"]
        lines.append(f"  Defined at line {var.defined_at}")
        
        if var.assigned_from:
            lines.append("  Assigned from:")
            for assignment in var.assigned_from:
                lines.append(f"    - {assignment}")
        
        if var.used_at:
            lines.append(f"  Used at lines: {', '.join(map(str, sorted(set(var.used_at))))}")
        else:
            lines.append("  ⚠️  Never used!")
        
        # Show dependencies
        deps = self.find_variable_dependencies(var_name)
        if deps:
            lines.append(f"  Depends on: {', '.join(sorted(deps))}")
        
        return "\\n".join(lines)

# Example usage
code = """
def calculate_shipping(base_price, weight, distance, is_express):
    # Data flow example
    price_per_mile = 0.5
    price_per_pound = 0.3
    express_multiplier = 1.5
    
    distance_cost = distance * price_per_mile
    weight_cost = weight * price_per_pound
    
    subtotal = base_price + distance_cost + weight_cost
    
    if is_express:
        total = subtotal * express_multiplier
    else:
        total = subtotal
    
    unused_variable = 42  # Never used!
    
    return total
"""

analyzer = DataFlowAnalyzer()
tree = ast.parse(code)
analyzer.visit(tree)

print("=== Data Flow Analysis ===\\n")

# Show trace for key variables
for var in ['total', 'subtotal', 'distance_cost']:
    print(analyzer.trace_variable(var))
    print()

# Show unused variables
unused = analyzer.find_unused_variables()
if unused:
    print(f"⚠️  Unused variables: {', '.join(unused)}")

# Show dependency graph
print("\\n=== Dependency Graph ===")
graph = analyzer.build_dependency_graph()
for var, deps in sorted(graph.items()):
    if deps:
        print(f"{var} depends on: {', '.join(sorted(deps))}")
\`\`\`

### Call Graph Builder

\`\`\`python
import ast
from dataclasses import dataclass
from typing import Dict, Set, List
from collections import defaultdict

@dataclass
class FunctionCall:
    caller: str
    callee: str
    lineno: int
    args_count: int

class CallGraphBuilder(ast.NodeVisitor):
    """
    Build a call graph showing which functions call which.
    Essential for understanding code dependencies and impact analysis.
    """
    
    def __init__(self):
        self.current_function: Optional[str] = None
        self.calls: List[FunctionCall] = []
        self.functions: Set[str] = set()
    
    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Track function definition and calls within it."""
        self.functions.add(node.name)
        
        old_function = self.current_function
        self.current_function = node.name
        self.generic_visit(node)
        self.current_function = old_function
    
    def visit_Call(self, node: ast.Call):
        """Track function calls."""
        callee = None
        
        if isinstance(node.func, ast.Name):
            # Direct function call: foo()
            callee = node.func.id
        elif isinstance(node.func, ast.Attribute):
            # Method call: obj.method()
            callee = node.func.attr
        
        if callee and self.current_function:
            self.calls.append(FunctionCall(
                caller=self.current_function,
                callee=callee,
                lineno=node.lineno,
                args_count=len(node.args)
            ))
        
        self.generic_visit(node)
    
    def get_callers(self, function_name: str) -> List[str]:
        """Find all functions that call a given function."""
        return [
            call.caller for call in self.calls 
            if call.callee == function_name
        ]
    
    def get_callees(self, function_name: str) -> List[str]:
        """Find all functions called by a given function."""
        return [
            call.callee for call in self.calls 
            if call.caller == function_name
        ]
    
    def find_call_chain(self, from_func: str, to_func: str) -> List[List[str]]:
        """
        Find call chains from one function to another.
        Returns list of paths.
        """
        paths = []
        
        def dfs(current: str, target: str, path: List[str], visited: Set[str]):
            if current == target:
                paths.append(path + [current])
                return
            
            if current in visited:
                return
            
            visited.add(current)
            callees = self.get_callees(current)
            
            for callee in callees:
                dfs(callee, target, path + [current], visited.copy())
        
        dfs(from_func, to_func, [], set())
        return paths
    
    def find_orphan_functions(self) -> Set[str]:
        """Find functions that are never called."""
        called = set(call.callee for call in self.calls)
        return self.functions - called
    
    def calculate_fan_in(self, function_name: str) -> int:
        """Count how many functions call this function (fan-in)."""
        return len(set(self.get_callers(function_name)))
    
    def calculate_fan_out(self, function_name: str) -> int:
        """Count how many functions this function calls (fan-out)."""
        return len(set(self.get_callees(function_name)))
    
    def visualize_call_graph(self) -> str:
        """Create a text visualization of the call graph."""
        lines = ["=== Call Graph ===\\n"]
        
        # Group calls by caller
        calls_by_caller = defaultdict(list)
        for call in self.calls:
            calls_by_caller[call.caller].append(call)
        
        for caller in sorted(calls_by_caller.keys()):
            lines.append(f"{caller}:")
            for call in calls_by_caller[caller]:
                lines.append(f"  └─> {call.callee} (line {call.lineno})")
        
        # Show orphan functions
        orphans = self.find_orphan_functions()
        if orphans:
            lines.append("\\n⚠️  Functions never called:")
            for func in sorted(orphans):
                lines.append(f"  - {func}")
        
        return "\\n".join(lines)

# Example usage
code = """
def main():
    data = load_data()
    result = process(data)
    save_result(result)

def load_data():
    raw = fetch_from_api()
    return parse_data(raw)

def fetch_from_api():
    return api_call()

def parse_data(raw):
    cleaned = clean(raw)
    return transform(cleaned)

def clean(data):
    return data.strip()

def transform(data):
    return data.upper()

def process(data):
    validated = validate(data)
    return calculate(validated)

def validate(data):
    return data if data else None

def calculate(data):
    return len(data)

def save_result(result):
    write_to_db(result)
    send_notification(result)

def write_to_db(result):
    pass

def send_notification(result):
    pass

def orphan_function():
    '''This is never called.'''
    pass
"""

builder = CallGraphBuilder()
tree = ast.parse(code)
builder.visit(tree)

print(builder.visualize_call_graph())

# Analyze specific functions
print("\\n=== Function Analysis ===\\n")
print("main function:")
print(f"  Calls: {', '.join(builder.get_callees('main'))}")
print(f"  Fan-out: {builder.calculate_fan_out('main')}")

print("\\nload_data function:")
print(f"  Called by: {', '.join(builder.get_callers('load_data'))}")
print(f"  Calls: {', '.join(builder.get_callees('load_data'))}")
print(f"  Fan-in: {builder.calculate_fan_in('load_data')}")
print(f"  Fan-out: {builder.calculate_fan_out('load_data')}")

# Find call chains
print("\\n=== Call Chains ===")
chains = builder.find_call_chain('main', 'clean')
print(f"\\nPaths from 'main' to 'clean':")
for i, chain in enumerate(chains, 1):
    print(f"  Path {i}: {' → '.join(chain)}")
\`\`\`

### Code Block Identifier

\`\`\`python
import ast
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class CodeBlock:
    type: str  # 'function', 'class', 'if', 'loop', 'try', etc.
    start_line: int
    end_line: int
    description: str
    parent: Optional['CodeBlock'] = None
    children: List['CodeBlock'] = None
    
    def __post_init__(self):
        if self.children is None:
            self.children = []

class BlockIdentifier(ast.NodeVisitor):
    """
    Identify and categorize code blocks.
    Useful for understanding code organization and structure.
    """
    
    def __init__(self):
        self.blocks: List[CodeBlock] = []
        self.block_stack: List[CodeBlock] = []
    
    def _add_block(self, block: CodeBlock):
        """Add a block and establish parent-child relationships."""
        if self.block_stack:
            parent = self.block_stack[-1]
            parent.children.append(block)
            block.parent = parent
        
        self.blocks.append(block)
    
    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Identify function blocks."""
        params = [arg.arg for arg in node.args.args]
        desc = f"def {node.name}({', '.join(params)})"
        
        block = CodeBlock(
            type='function',
            start_line=node.lineno,
            end_line=node.end_lineno or node.lineno,
            description=desc
        )
        
        self._add_block(block)
        self.block_stack.append(block)
        self.generic_visit(node)
        self.block_stack.pop()
    
    def visit_ClassDef(self, node: ast.ClassDef):
        """Identify class blocks."""
        bases = [ast.unparse(b) for b in node.bases]
        base_str = f"({', '.join(bases)})" if bases else ""
        
        block = CodeBlock(
            type='class',
            start_line=node.lineno,
            end_line=node.end_lineno or node.lineno,
            description=f"class {node.name}{base_str}"
        )
        
        self._add_block(block)
        self.block_stack.append(block)
        self.generic_visit(node)
        self.block_stack.pop()
    
    def visit_If(self, node: ast.If):
        """Identify conditional blocks."""
        condition = ast.unparse(node.test)
        
        block = CodeBlock(
            type='if',
            start_line=node.lineno,
            end_line=node.end_lineno or node.lineno,
            description=f"if {condition}"
        )
        
        self._add_block(block)
        self.block_stack.append(block)
        
        # Visit if body
        for stmt in node.body:
            self.visit(stmt)
        
        self.block_stack.pop()
        
        # Visit else
        if node.orelse:
            else_block = CodeBlock(
                type='else',
                start_line=node.orelse[0].lineno,
                end_line=node.orelse[-1].end_lineno if node.orelse else node.orelse[0].lineno,
                description="else"
            )
            self._add_block(else_block)
            self.block_stack.append(else_block)
            
            for stmt in node.orelse:
                self.visit(stmt)
            
            self.block_stack.pop()
    
    def visit_For(self, node: ast.For):
        """Identify loop blocks."""
        target = ast.unparse(node.target)
        iter_expr = ast.unparse(node.iter)
        
        block = CodeBlock(
            type='for',
            start_line=node.lineno,
            end_line=node.end_lineno or node.lineno,
            description=f"for {target} in {iter_expr}"
        )
        
        self._add_block(block)
        self.block_stack.append(block)
        self.generic_visit(node)
        self.block_stack.pop()
    
    def visit_While(self, node: ast.While):
        """Identify while loop blocks."""
        condition = ast.unparse(node.test)
        
        block = CodeBlock(
            type='while',
            start_line=node.lineno,
            end_line=node.end_lineno or node.lineno,
            description=f"while {condition}"
        )
        
        self._add_block(block)
        self.block_stack.append(block)
        self.generic_visit(node)
        self.block_stack.pop()
    
    def visit_Try(self, node: ast.Try):
        """Identify try/except blocks."""
        block = CodeBlock(
            type='try',
            start_line=node.lineno,
            end_line=node.end_lineno or node.lineno,
            description="try"
        )
        
        self._add_block(block)
        self.block_stack.append(block)
        
        # Visit try body
        for stmt in node.body:
            self.visit(stmt)
        
        self.block_stack.pop()
        
        # Visit except handlers
        for handler in node.handlers:
            exc_type = "Exception"
            if handler.type:
                exc_type = ast.unparse(handler.type)
            
            except_block = CodeBlock(
                type='except',
                start_line=handler.lineno,
                end_line=handler.end_lineno or handler.lineno,
                description=f"except {exc_type}"
            )
            self._add_block(except_block)
            self.block_stack.append(except_block)
            
            for stmt in handler.body:
                self.visit(stmt)
            
            self.block_stack.pop()
    
    def visualize_blocks(self, indent: int = 0) -> str:
        """Create a tree visualization of code blocks."""
        lines = []
        
        def visualize(block: CodeBlock, depth: int):
            prefix = "  " * depth
            line_range = f"[{block.start_line}-{block.end_line}]"
            lines.append(f"{prefix}{block.type}: {block.description} {line_range}")
            
            for child in block.children:
                visualize(child, depth + 1)
        
        # Visualize top-level blocks
        top_level = [b for b in self.blocks if b.parent is None]
        for block in top_level:
            visualize(block, indent)
        
        return "\\n".join(lines)
    
    def find_deep_nesting(self, max_depth: int = 3) -> List[CodeBlock]:
        """Find blocks that are nested too deeply."""
        deep_blocks = []
        
        def get_depth(block: CodeBlock) -> int:
            depth = 0
            current = block.parent
            while current:
                depth += 1
                current = current.parent
            return depth
        
        for block in self.blocks:
            if get_depth(block) > max_depth:
                deep_blocks.append(block)
        
        return deep_blocks

# Example usage
code = """
class DataProcessor:
    def process(self, data):
        results = []
        
        for item in data:
            if item.is_valid():
                try:
                    result = self.transform(item)
                    if result:
                        for validation in self.validators:
                            if not validation(result):
                                break
                        else:
                            results.append(result)
                except ProcessError as e:
                    log_error(e)
                    continue
        
        return results
    
    def transform(self, item):
        return item.value * 2
"""

identifier = BlockIdentifier()
tree = ast.parse(code)
identifier.visit(tree)

print("=== Code Block Structure ===\\n")
print(identifier.visualize_blocks())

# Check for deep nesting
deep = identifier.find_deep_nesting(max_depth=3)
if deep:
    print("\\n⚠️  Deeply nested blocks (>3 levels):")
    for block in deep:
        depth = 0
        current = block.parent
        while current:
            depth += 1
            current = current.parent
        print(f"  Line {block.start_line}: {block.description} (depth {depth})")
\`\`\`

## Real-World Case Study: How Cursor Uses Structure Analysis

Cursor leverages structure analysis for intelligent code understanding:

**1. Context-Aware Suggestions:**
\`\`\`python
def process_payment(amount, card):
    if amount <= 0:
        return None
    
    # When typing here, Cursor knows:
    # - We're inside an if block (control flow)
    # - amount and card are available (data flow)
    # - We might want to call validate_card (call graph)
    # - Similar patterns in codebase do X here
    |
\`\`\`

**2. Impact Analysis:**

When you modify a function, Cursor uses call graph analysis to:
- Show which functions will be affected
- Suggest updating all callers
- Identify potential breaking changes
- Recommend adding tests for impacted code

**3. Refactoring Support:**

For "extract method" refactoring:
1. Data flow analysis determines required parameters
2. Control flow analysis ensures correct extraction
3. Call graph analysis finds all usage sites
4. Block analysis maintains proper nesting

**4. Pattern Recognition:**

Structure analysis helps Cursor identify:
- Common error handling patterns
- Repeated code blocks that could be refactored
- Similar logic that could be unified
- Architectural patterns in your codebase

## Hands-On Exercise

Build a comprehensive code structure analyzer:

\`\`\`python
class ComprehensiveStructureAnalyzer:
    """
    Complete structure analysis combining all techniques.
    This creates a rich understanding of code suitable for AI tools.
    """
    
    def __init__(self, code: str):
        self.code = code
        self.tree = ast.parse(code)
        
        # Run all analyzers
        self.control_flow = ControlFlowAnalyzer()
        self.control_flow.visit(self.tree)
        
        self.data_flow = DataFlowAnalyzer()
        self.data_flow.visit(self.tree)
        
        self.call_graph = CallGraphBuilder()
        self.call_graph.visit(self.tree)
        
        self.blocks = BlockIdentifier()
        self.blocks.visit(self.tree)
    
    def analyze_function(self, func_name: str) -> Dict:
        """Complete analysis of a single function."""
        analysis = {
            'name': func_name,
            'complexity': self.control_flow.calculate_complexity(func_name),
            'calls': self.call_graph.get_callees(func_name),
            'called_by': self.call_graph.get_callers(func_name),
            'fan_in': self.call_graph.calculate_fan_in(func_name),
            'fan_out': self.call_graph.calculate_fan_out(func_name),
        }
        
        return analysis
    
    def generate_summary(self) -> str:
        """Generate complete summary for LLM context."""
        lines = ["# Code Structure Analysis\\n"]
        
        # Call graph summary
        lines.append("## Call Graph")
        lines.append(self.call_graph.visualize_call_graph())
        
        # Complexity metrics
        lines.append("\\n## Complexity Metrics")
        for func_name in self.call_graph.functions:
            complexity = self.control_flow.calculate_complexity(func_name)
            fan_in = self.call_graph.calculate_fan_in(func_name)
            fan_out = self.call_graph.calculate_fan_out(func_name)
            lines.append(f"- {func_name}: complexity={complexity}, fan-in={fan_in}, fan-out={fan_out}")
        
        # Block structure
        lines.append("\\n## Block Structure")
        lines.append(self.blocks.visualize_blocks())
        
        # Warnings
        lines.append("\\n## Warnings")
        unused = self.data_flow.find_unused_variables()
        if unused:
            lines.append(f"- Unused variables: {', '.join(unused)}")
        
        orphans = self.call_graph.find_orphan_functions()
        if orphans:
            lines.append(f"- Orphan functions: {', '.join(orphans)}")
        
        deep_blocks = self.blocks.find_deep_nesting(max_depth=3)
        if deep_blocks:
            lines.append(f"- Deeply nested blocks: {len(deep_blocks)} found")
        
        return "\\n".join(lines)

# Test it
code = """
def main():
    data = fetch_data()
    processed = process(data)
    save(processed)

def fetch_data():
    return load_from_db()

def load_from_db():
    return [1, 2, 3, 4, 5]

def process(data):
    result = []
    for item in data:
        if item > 2:
            result.append(item * 2)
    return result

def save(data):
    write_to_file(data)
    log_completion()

def write_to_file(data):
    pass

def log_completion():
    pass
"""

analyzer = ComprehensiveStructureAnalyzer(code)
print(analyzer.generate_summary())
\`\`\`

## Common Pitfalls

### 1. Not Handling Nested Structures

\`\`\`python
# ❌ Wrong: Only finds top-level functions
def find_all_functions(tree):
    return [node for node in tree.body if isinstance(node, ast.FunctionDef)]

# ✅ Correct: Find all functions including nested
def find_all_functions(tree):
    return [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
\`\`\`

### 2. Ignoring Control Flow Complexity

\`\`\`python
# ❌ Wrong: Simple line count
def is_complex(func):
    return len(func.body) > 20

# ✅ Correct: Use cyclomatic complexity
def is_complex(func):
    analyzer = ControlFlowAnalyzer()
    analyzer.visit(func)
    return analyzer.calculate_complexity(func.name) > 10
\`\`\`

### 3. Missing Indirect Dependencies

\`\`\`python
# ❌ Wrong: Only direct calls
def get_dependencies(func):
    return find_direct_calls(func)

# ✅ Correct: Transitive dependencies
def get_dependencies(func):
    direct = find_direct_calls(func)
    indirect = set()
    for dep in direct:
        indirect.update(get_dependencies(dep))
    return direct | indirect
\`\`\`

## Production Checklist

### Analysis Coverage
- [ ] Analyze control flow (branches, loops, exceptions)
- [ ] Track data flow (variable definitions and usage)
- [ ] Build call graph (function dependencies)
- [ ] Identify code blocks and nesting
- [ ] Calculate complexity metrics
- [ ] Find unused and orphan code

### Performance
- [ ] Cache analysis results
- [ ] Support incremental analysis
- [ ] Profile for bottlenecks
- [ ] Handle large codebases efficiently
- [ ] Limit recursion depth

### Accuracy
- [ ] Handle edge cases (nested functions, closures)
- [ ] Track all variable scopes correctly
- [ ] Identify indirect dependencies
- [ ] Account for dynamic behavior
- [ ] Validate analysis results

### Output
- [ ] Provide structured data (JSON/dict)
- [ ] Generate human-readable summaries
- [ ] Create visualizations
- [ ] Support filtering and queries
- [ ] Document limitations

### Integration
- [ ] Design clean API
- [ ] Support multiple languages
- [ ] Enable batch processing
- [ ] Provide incremental updates
- [ ] Expose metrics and statistics

## Summary

Code structure analysis provides deep understanding of code organization:

- **Control Flow**: Understand execution paths and complexity
- **Data Flow**: Track how data moves through code
- **Call Graph**: Map function dependencies
- **Block Analysis**: Identify code organization patterns
- **Metrics**: Quantify code quality and complexity

These techniques enable AI coding tools like Cursor to provide intelligent, context-aware suggestions that understand not just individual functions, but the entire structure and flow of your codebase.

In the next section, we'll explore symbol resolution and references—understanding what names mean and where they're used throughout your code.
`
};
