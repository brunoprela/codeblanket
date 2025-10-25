const staticAnalysis = {
  id: 'static-analysis',
  title: 'Static Analysis & Code Quality',
  content: `
# Static Analysis & Code Quality

## Introduction

Static analysis examines code without executing it to find bugs, security issues, performance problems, and style violations. This is how tools like Cursor provide real-time feedback on code quality, suggest improvements, and catch errors before runtime.

**Why Static Analysis Matters:**

When Cursor shows you a warning that "this variable is never used" or suggests "this function is too complex," it's using static analysis. Modern AI coding tools need to:
- Detect potential bugs before runtime
- Identify security vulnerabilities
- Calculate complexity metrics
- Enforce code style
- Find code smells and anti-patterns
- Suggest improvements

This section teaches you to build these capabilities.

## Deep Technical Explanation

### Types of Static Analysis

**1. Syntactic Analysis:**
- Checks code structure
- Validates syntax rules
- Ensures well-formed AST
- Basic style checking

**2. Semantic Analysis:**
- Understands code meaning
- Tracks data flow
- Analyzes control flow
- Detects logical errors

**3. Data Flow Analysis:**
- Tracks how data moves
- Finds uninitialized variables
- Detects unused variables
- Identifies dead code

**4. Control Flow Analysis:**
- Maps execution paths
- Finds unreachable code
- Detects infinite loops
- Calculates complexity

**5. Security Analysis:**
- Identifies vulnerabilities
- Finds injection points
- Checks for unsafe patterns
- Validates input handling

### Code Metrics

**Cyclomatic Complexity:**
- Measures code complexity
- Counts decision points
- Formula: E - N + 2P (edges - nodes + 2*components)
- Simplified: 1 + number of decision points

**Maintainability Index:**
- Combines multiple metrics
- Range: 0-100 (higher is better)
- Considers complexity, length, and comments

**Code Churn:**
- Frequency of changes
- Indicates problematic areas
- Helps prioritize refactoring

## Code Implementation

### Complexity Analyzer

\`\`\`python
import ast
from dataclasses import dataclass
from typing import Dict, List

@dataclass
class ComplexityMetrics:
    cyclomatic: int
    cognitive: int  # Cognitive complexity (nesting-aware)
    lines: int
    statements: int

class ComplexityAnalyzer (ast.NodeVisitor):
    """
    Calculate various complexity metrics for code.
    This helps identify functions that need refactoring.
    """
    
    def __init__(self):
        self.functions: Dict[str, ComplexityMetrics] = {}
        self.current_function: Optional[str] = None
        self.nesting_depth: int = 0
    
    def visit_FunctionDef (self, node: ast.FunctionDef):
        """Analyze function complexity."""
        # Calculate cyclomatic complexity
        cyclomatic = self._calculate_cyclomatic (node)
        
        # Calculate cognitive complexity (nesting-aware)
        cognitive = self._calculate_cognitive (node)
        
        # Count lines and statements
        lines = (node.end_lineno or node.lineno) - node.lineno + 1
        statements = len (list (ast.walk (node)))
        
        self.functions[node.name] = ComplexityMetrics(
            cyclomatic=cyclomatic,
            cognitive=cognitive,
            lines=lines,
            statements=statements
        )
        
        old_function = self.current_function
        self.current_function = node.name
        self.generic_visit (node)
        self.current_function = old_function
    
    def _calculate_cyclomatic (self, node: ast.FunctionDef) -> int:
        """Calculate cyclomatic complexity."""
        complexity = 1  # Base complexity
        
        for child in ast.walk (node):
            # Each decision point adds 1
            if isinstance (child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance (child, ast.ExceptHandler):
                complexity += 1
            elif isinstance (child, ast.BoolOp):
                # And/Or operators add complexity
                complexity += len (child.values) - 1
            elif isinstance (child, (ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp)):
                # Comprehensions with conditions
                for generator in getattr (child, 'generators', []):
                    complexity += len (generator.ifs)
        
        return complexity
    
    def _calculate_cognitive (self, node: ast.FunctionDef) -> int:
        """
        Calculate cognitive complexity (considers nesting).
        Nested code is harder to understand.
        """
        complexity = 0
        
        def traverse (n: ast.AST, depth: int):
            nonlocal complexity
            
            if isinstance (n, (ast.If, ast.While, ast.For)):
                # Add base + nesting penalty
                complexity += 1 + depth
                depth += 1
            elif isinstance (n, ast.BoolOp):
                complexity += 1
            
            for child in ast.iter_child_nodes (n):
                traverse (child, depth)
        
        for stmt in node.body:
            traverse (stmt, 0)
        
        return complexity
    
    def get_complex_functions (self, threshold: int = 10) -> List[str]:
        """Find functions above complexity threshold."""
        return [
            name for name, metrics in self.functions.items()
            if metrics.cyclomatic > threshold
        ]
    
    def generate_report (self) -> str:
        """Generate complexity report."""
        lines = ["=== Complexity Report ===\\n"]
        
        for name, metrics in sorted (self.functions.items()):
            lines.append (f"Function: {name}")
            lines.append (f"  Cyclomatic Complexity: {metrics.cyclomatic}")
            lines.append (f"  Cognitive Complexity: {metrics.cognitive}")
            lines.append (f"  Lines: {metrics.lines}")
            lines.append (f"  Statements: {metrics.statements}")
            
            # Add warnings
            if metrics.cyclomatic > 10:
                lines.append("  ⚠️  High cyclomatic complexity (>10)")
            if metrics.cognitive > 15:
                lines.append("  ⚠️  High cognitive complexity (>15)")
            if metrics.lines > 50:
                lines.append("  ⚠️  Long function (>50 lines)")
            
            lines.append("")
        
        return "\\n".join (lines)

# Example usage
code = """
def complex_function (data, options):
    if not data:
        return None
    
    results = []
    for item in data:
        if item.valid:
            if options.get('strict'):
                if item.score > 0.8:
                    try:
                        processed = transform (item)
                        if processed:
                            for validator in options.get('validators', []):
                                if not validator (processed):
                                    break
                            else:
                                results.append (processed)
                    except ProcessError:
                        log_error (item)
                        continue
    
    return results if results else None

def simple_function (x, y):
    return x + y
"""

analyzer = ComplexityAnalyzer()
tree = ast.parse (code)
analyzer.visit (tree)

print(analyzer.generate_report())

# Find complex functions
complex_funcs = analyzer.get_complex_functions (threshold=5)
if complex_funcs:
    print(f"\\n⚠️  Functions needing refactoring: {', '.join (complex_funcs)}")
\`\`\`

### Bug Pattern Detector

\`\`\`python
import ast
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class BugReport:
    type: str
    line: int
    message: str
    severity: str  # 'error', 'warning', 'info'
    suggestion: str

class BugPatternDetector (ast.NodeVisitor):
    """
    Detect common bug patterns in code.
    This is how linters like pylint and flake8 work.
    """
    
    def __init__(self):
        self.issues: List[BugReport] = []
    
    def visit_Compare (self, node: ast.Compare):
        """Detect comparison issues."""
        # Check for comparison with True/False
        for op, comparator in zip (node.ops, node.comparators):
            if isinstance (comparator, ast.Constant):
                if comparator.value is True or comparator.value is False:
                    self.issues.append(BugReport(
                        type='compare_with_boolean',
                        line=node.lineno,
                        message=f"Comparing with {comparator.value} is unnecessary",
                        severity='warning',
                        suggestion="Use 'if variable:' instead of 'if variable == True:'"
                    ))
            
            # Check for is/is not with literals
            if isinstance (op, (ast.Is, ast.IsNot)):
                if isinstance (comparator, ast.Constant):
                    if not (comparator.value is None):
                        self.issues.append(BugReport(
                            type='is_with_literal',
                            line=node.lineno,
                            message="Using 'is' with literal (should use '==')",
                            severity='error',
                            suggestion="Use '==' for value comparison, 'is' only for None"
                        ))
        
        self.generic_visit (node)
    
    def visit_Try (self, node: ast.Try):
        """Detect exception handling issues."""
        for handler in node.handlers:
            # Check for bare except
            if handler.type is None:
                self.issues.append(BugReport(
                    type='bare_except',
                    line=handler.lineno,
                    message="Bare 'except:' catches all exceptions including system exit",
                    severity='error',
                    suggestion="Use 'except Exception:' to catch only regular exceptions"
                ))
            
            # Check for catching Exception and doing nothing
            if handler.body:
                if len (handler.body) == 1 and isinstance (handler.body[0], ast.Pass):
                    self.issues.append(BugReport(
                        type='silent_exception',
                        line=handler.lineno,
                        message="Exception caught but not handled (pass)",
                        severity='warning',
                        suggestion="At minimum, log the exception"
                    ))
        
        self.generic_visit (node)
    
    def visit_FunctionDef (self, node: ast.FunctionDef):
        """Detect function issues."""
        # Check for mutable default arguments
        for default in node.args.defaults:
            if isinstance (default, (ast.List, ast.Dict, ast.Set)):
                self.issues.append(BugReport(
                    type='mutable_default',
                    line=node.lineno,
                    message=f"Mutable default argument in function '{node.name}'",
                    severity='error',
                    suggestion="Use None as default and create mutable inside function"
                ))
        
        # Check for missing docstring
        if not ast.get_docstring (node):
            if not node.name.startswith('_'):  # Skip private functions
                self.issues.append(BugReport(
                    type='missing_docstring',
                    line=node.lineno,
                    message=f"Function '{node.name}' has no docstring",
                    severity='info',
                    suggestion="Add docstring describing parameters and return value"
                ))
        
        # Check for too many arguments
        num_args = len (node.args.args)
        if num_args > 5:
            self.issues.append(BugReport(
                type='too_many_parameters',
                line=node.lineno,
                message=f"Function '{node.name}' has {num_args} parameters (>5)",
                severity='warning',
                suggestion="Consider grouping parameters into a config object"
            ))
        
        self.generic_visit (node)
    
    def visit_Return (self, node: ast.Return):
        """Detect return statement issues."""
        # Check for returning multiple types
        # This is simplified - real analysis would track all returns
        if isinstance (node.value, ast.IfExp):
            self.issues.append(BugReport(
                type='conditional_return',
                line=node.lineno,
                message="Conditional expression in return",
                severity='info',
                suggestion="Consider making conditional logic more explicit"
            ))
        
        self.generic_visit (node)
    
    def visit_Call (self, node: ast.Call):
        """Detect function call issues."""
        # Check for dangerous eval/exec
        if isinstance (node.func, ast.Name):
            if node.func.id in ['eval', 'exec']:
                self.issues.append(BugReport(
                    type='dangerous_function',
                    line=node.lineno,
                    message=f"Use of dangerous function '{node.func.id}'",
                    severity='error',
                    suggestion="Avoid eval/exec - use safer alternatives like ast.literal_eval"
                ))
        
        self.generic_visit (node)
    
    def visit_Import (self, node: ast.Import):
        """Detect import issues."""
        for alias in node.names:
            # Check for import *
            if alias.name == '*':
                self.issues.append(BugReport(
                    type='wildcard_import',
                    line=node.lineno,
                    message="Wildcard import (from x import *)",
                    severity='warning',
                    suggestion="Import specific names or use qualified imports"
                ))
        
        self.generic_visit (node)
    
    def generate_report (self) -> str:
        """Generate bug report."""
        if not self.issues:
            return "✅ No issues found!"
        
        lines = [f"=== Found {len (self.issues)} Issues ===\\n"]
        
        # Sort by severity then line number
        severity_order = {'error': 0, 'warning': 1, 'info': 2}
        sorted_issues = sorted(
            self.issues,
            key=lambda x: (severity_order[x.severity], x.line)
        )
        
        for issue in sorted_issues:
            symbol = {'error': '❌', 'warning': '⚠️', 'info': 'ℹ️'}[issue.severity]
            lines.append (f"{symbol} Line {issue.line}: {issue.message}")
            lines.append (f"   Type: {issue.type}")
            lines.append (f"   Suggestion: {issue.suggestion}")
            lines.append("")
        
        return "\\n".join (lines)

# Example usage
code = """
def process_data (items, options, config, flags, mode, debug):  # Too many params
    # No docstring
    if items == True:  # Bad comparison
        return None
    
    try:
        result = eval (user_input)  # Dangerous!
    except:  # Bare except
        pass  # Silent exception
    
    if value is 5:  # Using 'is' with literal
        return True
    
    return None

from bad_module import *  # Wildcard import

def with_mutable_default (items=[]):  # Mutable default
    items.append(1)
    return items
"""

detector = BugPatternDetector()
tree = ast.parse (code)
detector.visit (tree)

print(detector.generate_report())

# Count by severity
error_count = sum(1 for i in detector.issues if i.severity == 'error')
warning_count = sum(1 for i in detector.issues if i.severity == 'warning')
print(f"\\nSummary: {error_count} errors, {warning_count} warnings")
\`\`\`

### Code Smell Detector

\`\`\`python
import ast
from dataclasses import dataclass
from typing import List, Set

@dataclass
class CodeSmell:
    name: str
    line: int
    description: str
    impact: str  # 'maintainability', 'performance', 'readability'
    refactoring_suggestion: str

class CodeSmellDetector (ast.NodeVisitor):
    """
    Detect code smells (indicators of deeper problems).
    This helps identify refactoring opportunities.
    """
    
    def __init__(self):
        self.smells: List[CodeSmell] = []
        self.function_lengths: Dict[str, int] = {}
        self.variable_uses: Dict[str, int] = {}
    
    def visit_FunctionDef (self, node: ast.FunctionDef):
        """Detect function-level smells."""
        func_name = node.name
        
        # Long method smell
        lines = (node.end_lineno or node.lineno) - node.lineno
        self.function_lengths[func_name] = lines
        
        if lines > 50:
            self.smells.append(CodeSmell(
                name='Long Method',
                line=node.lineno,
                description=f"Function '{func_name}' is {lines} lines long",
                impact='maintainability',
                refactoring_suggestion="Break into smaller, focused functions"
            ))
        
        # Feature envy (many calls to same object)
        call_targets = self._analyze_call_targets (node)
        if call_targets:
            most_common = max (call_targets.items(), key=lambda x: x[1])
            if most_common[1] > 3:
                self.smells.append(CodeSmell(
                    name='Feature Envy',
                    line=node.lineno,
                    description=f"Function calls '{most_common[0]}' {most_common[1]} times",
                    impact='maintainability',
                    refactoring_suggestion=f"Consider moving logic to {most_common[0]} class"
                ))
        
        # God function (too many responsibilities)
        responsibilities = self._count_responsibilities (node)
        if responsibilities > 5:
            self.smells.append(CodeSmell(
                name='God Function',
                line=node.lineno,
                description=f"Function has {responsibilities} distinct responsibilities",
                impact='maintainability',
                refactoring_suggestion="Split into multiple single-purpose functions"
            ))
        
        self.generic_visit (node)
    
    def visit_ClassDef (self, node: ast.ClassDef):
        """Detect class-level smells."""
        # Large class smell
        method_count = sum(1 for n in node.body if isinstance (n, ast.FunctionDef))
        
        if method_count > 15:
            self.smells.append(CodeSmell(
                name='Large Class',
                line=node.lineno,
                description=f"Class '{node.name}' has {method_count} methods",
                impact='maintainability',
                refactoring_suggestion="Split into multiple focused classes"
            ))
        
        # Data class smell (only getters/setters)
        if self._is_data_class (node):
            self.smells.append(CodeSmell(
                name='Data Class',
                line=node.lineno,
                description=f"Class '{node.name}' only contains data (no behavior)",
                impact='design',
                refactoring_suggestion="Consider using @dataclass or moving logic here"
            ))
        
        self.generic_visit (node)
    
    def visit_If (self, node: ast.If):
        """Detect conditional smells."""
        # Nested conditionals smell
        nesting_depth = self._calculate_nesting_depth (node)
        
        if nesting_depth > 3:
            self.smells.append(CodeSmell(
                name='Nested Conditionals',
                line=node.lineno,
                description=f"Conditional nested {nesting_depth} levels deep",
                impact='readability',
                refactoring_suggestion="Use early returns or extract to functions"
            ))
        
        self.generic_visit (node)
    
    def visit_For (self, node: ast.For):
        """Detect loop smells."""
        # Nested loops smell
        nested_loops = sum(1 for n in ast.walk (node) if isinstance (n, (ast.For, ast.While)))
        
        if nested_loops > 2:
            self.smells.append(CodeSmell(
                name='Nested Loops',
                line=node.lineno,
                description=f"{nested_loops} nested loops",
                impact='performance',
                refactoring_suggestion="Consider list comprehensions or extract to function"
            ))
        
        self.generic_visit (node)
    
    def _analyze_call_targets (self, node: ast.FunctionDef) -> Dict[str, int]:
        """Count calls to each target object."""
        targets = {}
        
        for child in ast.walk (node):
            if isinstance (child, ast.Call):
                if isinstance (child.func, ast.Attribute):
                    if isinstance (child.func.value, ast.Name):
                        target = child.func.value.id
                        targets[target] = targets.get (target, 0) + 1
        
        return targets
    
    def _count_responsibilities (self, node: ast.FunctionDef) -> int:
        """Count distinct responsibilities (simplified)."""
        # Count: assignments, calls, returns, try blocks
        responsibilities = 0
        
        for child in node.body:
            if isinstance (child, (ast.Assign, ast.AnnAssign)):
                responsibilities += 1
            elif isinstance (child, ast.Expr) and isinstance (child.value, ast.Call):
                responsibilities += 1
            elif isinstance (child, ast.Try):
                responsibilities += 1
        
        return responsibilities
    
    def _is_data_class (self, node: ast.ClassDef) -> bool:
        """Check if class is just a data holder."""
        methods = [n for n in node.body if isinstance (n, ast.FunctionDef)]
        
        # If only __init__ or simple getters/setters
        if len (methods) <= 1:
            return True
        
        # Check if all methods are just return self.x
        for method in methods:
            if method.name not in ['__init__', '__repr__', '__str__']:
                # If it has any logic beyond return self.x, not a pure data class
                if len (method.body) > 1:
                    return False
        
        return True
    
    def _calculate_nesting_depth (self, node: ast.AST, depth: int = 0) -> int:
        """Calculate maximum nesting depth."""
        max_depth = depth
        
        for child in ast.iter_child_nodes (node):
            if isinstance (child, (ast.If, ast.For, ast.While, ast.With)):
                child_depth = self._calculate_nesting_depth (child, depth + 1)
                max_depth = max (max_depth, child_depth)
        
        return max_depth
    
    def generate_report (self) -> str:
        """Generate code smell report."""
        if not self.smells:
            return "✅ No code smells detected!"
        
        lines = [f"=== Detected {len (self.smells)} Code Smells ===\\n"]
        
        # Group by impact
        by_impact = {}
        for smell in self.smells:
            by_impact.setdefault (smell.impact, []).append (smell)
        
        for impact, smells in sorted (by_impact.items()):
            lines.append (f"## {impact.title()} Issues\\n")
            
            for smell in sorted (smells, key=lambda s: s.line):
                lines.append (f"Line {smell.line}: {smell.name}")
                lines.append (f"  {smell.description}")
                lines.append (f"  Suggestion: {smell.refactoring_suggestion}")
                lines.append("")
        
        return "\\n".join (lines)

# Example usage
code = """
class DataHolder:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def get_x (self):
        return self.x

class MassiveService:
    def process_everything (self, data):
        # God function with many responsibilities
        validated = self.validate (data)
        cleaned = self.clean (data)
        transformed = self.transform (data)
        enriched = self.enrich (data)
        filtered = self.filter (data)
        sorted_data = self.sort (data)
        
        # Deep nesting
        if validated:
            if cleaned:
                if transformed:
                    if enriched:
                        if filtered:
                            return sorted_data
        
        return None

def complex_processing():
    # Nested loops
    results = []
    for i in range(10):
        for j in range(10):
            for k in range(10):
                results.append (i * j * k)
    return results
"""

detector = CodeSmellDetector()
tree = ast.parse (code)
detector.visit (tree)

print(detector.generate_report())
\`\`\`

### Security Vulnerability Scanner

\`\`\`python
import ast
from dataclasses import dataclass
from typing import List

@dataclass
class SecurityIssue:
    type: str
    line: int
    severity: str  # 'critical', 'high', 'medium', 'low'
    description: str
    cwe_id: Optional[str]  # Common Weakness Enumeration ID
    remediation: str

class SecurityScanner (ast.NodeVisitor):
    """
    Scan for security vulnerabilities in code.
    Detects common security anti-patterns.
    """
    
    def __init__(self):
        self.issues: List[SecurityIssue] = []
    
    def visit_Call (self, node: ast.Call):
        """Detect dangerous function calls."""
        if isinstance (node.func, ast.Name):
            func_name = node.func.id
            
            # Dangerous eval/exec
            if func_name in ['eval', 'exec', '__import__']:
                self.issues.append(SecurityIssue(
                    type='code_injection',
                    line=node.lineno,
                    severity='critical',
                    description=f"Use of {func_name}() with potentially untrusted input",
                    cwe_id='CWE-95',
                    remediation="Never use eval/exec on user input. Use ast.literal_eval for safe evaluation"
                ))
            
            # Unsafe deserialization
            elif func_name == 'pickle' or (isinstance (node.func, ast.Attribute) and node.func.attr in ['loads', 'load']):
                self.issues.append(SecurityIssue(
                    type='unsafe_deserialization',
                    line=node.lineno,
                    severity='high',
                    description="Pickle deserialization can execute arbitrary code",
                    cwe_id='CWE-502',
                    remediation="Use JSON or other safe serialization formats"
                ))
        
        # Check for SQL injection patterns
        if isinstance (node.func, ast.Attribute):
            if node.func.attr == 'execute':
                # Check if SQL is constructed with string formatting
                if node.args and isinstance (node.args[0], ast.JoinedStr):
                    self.issues.append(SecurityIssue(
                        type='sql_injection',
                        line=node.lineno,
                        severity='critical',
                        description="SQL query constructed with f-string (SQL injection risk)",
                        cwe_id='CWE-89',
                        remediation="Use parameterized queries with ? placeholders"
                    ))
        
        self.generic_visit (node)
    
    def visit_Assign (self, node: ast.Assign):
        """Detect security issues in assignments."""
        # Check for hardcoded credentials
        if isinstance (node.value, ast.Constant):
            if isinstance (node.value.value, str):
                value_lower = node.value.value.lower()
                
                for target in node.targets:
                    if isinstance (target, ast.Name):
                        var_name_lower = target.id.lower()
                        
                        if any (word in var_name_lower for word in ['password', 'secret', 'key', 'token', 'api_key']):
                            if len (node.value.value) > 8:  # Likely not a placeholder
                                self.issues.append(SecurityIssue(
                                    type='hardcoded_credentials',
                                    line=node.lineno,
                                    severity='high',
                                    description=f"Hardcoded credential in variable '{target.id}'",
                                    cwe_id='CWE-798',
                                    remediation="Store credentials in environment variables or secure vaults"
                                ))
        
        self.generic_visit (node)
    
    def visit_Try (self, node: ast.Try):
        """Detect security issues in exception handling."""
        for handler in node.handlers:
            # Catching Exception and ignoring
            if handler.body and len (handler.body) == 1:
                if isinstance (handler.body[0], ast.Pass):
                    self.issues.append(SecurityIssue(
                        type='exception_swallowing',
                        line=handler.lineno,
                        severity='medium',
                        description="Exception caught and ignored (may hide security issues)",
                        cwe_id='CWE-391',
                        remediation="Log exceptions and handle appropriately"
                    ))
        
        self.generic_visit (node)
    
    def visit_Import (self, node: ast.Import):
        """Detect insecure imports."""
        for alias in node.names:
            # Check for known vulnerable modules
            if alias.name in ['pickle', 'marshal']:
                self.issues.append(SecurityIssue(
                    type='insecure_module',
                    line=node.lineno,
                    severity='medium',
                    description=f"Import of {alias.name} (unsafe deserialization)",
                    cwe_id='CWE-502',
                    remediation="Consider safer alternatives like JSON"
                ))
        
        self.generic_visit (node)
    
    def generate_report (self) -> str:
        """Generate security report."""
        if not self.issues:
            return "✅ No security issues detected!"
        
        lines = [f"🔒 Security Scan Results: {len (self.issues)} issues found\\n"]
        
        # Group by severity
        by_severity = {'critical': [], 'high': [], 'medium': [], 'low': []}
        for issue in self.issues:
            by_severity[issue.severity].append (issue)
        
        severity_symbols = {
            'critical': '🚨',
            'high': '⚠️',
            'medium': '⚡',
            'low': 'ℹ️'
        }
        
        for severity in ['critical', 'high', 'medium', 'low']:
            issues = by_severity[severity]
            if not issues:
                continue
            
            lines.append (f"## {severity_symbols[severity]} {severity.upper()} Severity ({len (issues)})\\n")
            
            for issue in sorted (issues, key=lambda i: i.line):
                lines.append (f"Line {issue.line}: {issue.type.replace('_', ' ').title()}")
                lines.append (f"  {issue.description}")
                if issue.cwe_id:
                    lines.append (f"  CWE: {issue.cwe_id}")
                lines.append (f"  Fix: {issue.remediation}")
                lines.append("")
        
        return "\\n".join (lines)

# Example usage
code = """
import pickle

# Hardcoded credentials - BAD!
API_KEY = "sk-1234567890abcdef"
PASSWORD = "SuperSecret123!"

def process_data (user_input):
    # Code injection vulnerability
    result = eval (user_input)
    
    # SQL injection vulnerability
    user_id = request.args.get('id')
    query = f"SELECT * FROM users WHERE id = {user_id}"
    cursor.execute (query)
    
    # Unsafe deserialization
    data = pickle.loads (uploaded_file)
    
    try:
        risky_operation()
    except:
        pass  # Swallowing exceptions
    
    return result
"""

scanner = SecurityScanner()
tree = ast.parse (code)
scanner.visit (tree)

print(scanner.generate_report())

# Count by severity
critical = sum(1 for i in scanner.issues if i.severity == 'critical')
high = sum(1 for i in scanner.issues if i.severity == 'high')
print(f"\\n⚠️  {critical} critical, {high} high severity issues require immediate attention")
\`\`\`

## Real-World Case Study: How Cursor Uses Static Analysis

Cursor integrates static analysis for real-time feedback:

**1. Real-Time Warnings:**
\`\`\`python
def process (data=[]):  # Cursor immediately shows warning
    # "Mutable default argument"
    data.append(1)
    return data
\`\`\`

**2. Complexity Indicators:**
\`\`\`python
def complex_function():  # Cursor shows complexity: 15
    # Yellow/red indicator for high complexity
    # Suggests: "Consider refactoring"
\`\`\`

**3. Security Alerts:**
\`\`\`python
user_input = get_input()
eval (user_input)  # Cursor shows critical security warning
# "Code injection vulnerability - never use eval on user input"
\`\`\`

**4. Code Quality Metrics:**
\`\`\`
File: user_service.py
- Maintainability Index: 65 (Good)
- Average Complexity: 4.2
- Test Coverage: 85%
- Issues: 2 warnings, 0 errors
\`\`\`

## Hands-On Exercise

Build a comprehensive code quality analyzer:

\`\`\`python
class ComprehensiveCodeQualityAnalyzer:
    """
    Complete code quality analysis system.
    Combines all analysis techniques.
    """
    
    def __init__(self, code: str):
        self.code = code
        self.tree = ast.parse (code)
        
        # Run all analyzers
        self.complexity = ComplexityAnalyzer()
        self.complexity.visit (self.tree)
        
        self.bugs = BugPatternDetector()
        self.bugs.visit (self.tree)
        
        self.smells = CodeSmellDetector()
        self.smells.visit (self.tree)
        
        self.security = SecurityScanner()
        self.security.visit (self.tree)
    
    def calculate_maintainability_index (self) -> float:
        """
        Calculate maintainability index (0-100).
        Higher is better.
        """
        # Simplified calculation
        avg_complexity = sum(
            m.cyclomatic for m in self.complexity.functions.values()
        ) / max (len (self.complexity.functions), 1)
        
        num_lines = len (self.code.split('\\n'))
        
        # Formula: 171 - 5.2 * ln (volume) - 0.23 * complexity - 16.2 * ln (lines)
        import math
        volume = num_lines * 10  # Simplified
        
        mi = (
            171
            - 5.2 * math.log (volume)
            - 0.23 * avg_complexity
            - 16.2 * math.log (num_lines)
        )
        
        # Normalize to 0-100
        mi = max(0, min(100, mi))
        return round (mi, 2)
    
    def generate_comprehensive_report (self) -> str:
        """Generate complete quality report."""
        lines = ["=" * 60]
        lines.append("CODE QUALITY ANALYSIS REPORT")
        lines.append("=" * 60)
        
        # Overall score
        mi = self.calculate_maintainability_index()
        lines.append (f"\\nMaintainability Index: {mi}/100")
        
        if mi >= 80:
            lines.append("  ✅ Excellent maintainability")
        elif mi >= 60:
            lines.append("  👍 Good maintainability")
        elif mi >= 40:
            lines.append("  ⚠️  Moderate maintainability")
        else:
            lines.append("  🚨 Poor maintainability - needs refactoring")
        
        # Issue summary
        lines.append (f"\\n{'='*60}")
        lines.append("ISSUE SUMMARY")
        lines.append("=" * 60)
        
        critical_security = sum(
            1 for i in self.security.issues if i.severity == 'critical'
        )
        errors = sum(1 for i in self.bugs.issues if i.severity == 'error')
        warnings = sum(1 for i in self.bugs.issues if i.severity == 'warning')
        
        lines.append (f"  🔒 Security: {len (self.security.issues)} issues ({critical_security} critical)")
        lines.append (f"  🐛 Bugs: {len (self.bugs.issues)} issues ({errors} errors)")
        lines.append (f"  💭 Code Smells: {len (self.smells.smells)} smells")
        
        # Detailed reports
        if self.security.issues:
            lines.append (f"\\n{self.security.generate_report()}")
        
        if self.bugs.issues:
            lines.append (f"\\n{self.bugs.generate_report()}")
        
        lines.append (f"\\n{self.complexity.generate_report()}")
        
        if self.smells.smells:
            lines.append (f"\\n{self.smells.generate_report()}")
        
        return "\\n".join (lines)
    
    def get_actionable_items (self) -> List[str]:
        """Get prioritized list of actions to take."""
        actions = []
        
        # Priority 1: Critical security issues
        critical = [i for i in self.security.issues if i.severity == 'critical']
        for issue in critical:
            actions.append (f"🚨 URGENT: Fix {issue.type} at line {issue.line}")
        
        # Priority 2: Error-level bugs
        errors = [i for i in self.bugs.issues if i.severity == 'error']
        for issue in errors:
            actions.append (f"❌ Fix {issue.type} at line {issue.line}")
        
        # Priority 3: High complexity functions
        complex_funcs = self.complexity.get_complex_functions (threshold=10)
        for func in complex_funcs:
            actions.append (f"🔧 Refactor complex function: {func}")
        
        # Priority 4: Code smells
        for smell in self.smells.smells[:3]:  # Top 3
            actions.append (f"💭 Address {smell.name} at line {smell.line}")
        
        return actions

# Test the comprehensive analyzer
code = """
import pickle

API_KEY = "sk-abc123def456"

def process_user_data (data, options, config, mode, flags, debug_level):
    if not data:
        return None
    
    results = []
    for item in data:
        if item.valid:
            if options.get('strict'):
                if item.score > 0.8:
                    try:
                        processed = eval (str (item))
                        if processed:
                            results.append (processed)
                    except:
                        pass
    
    query = f"SELECT * FROM users WHERE id = {data['id']}"
    cursor.execute (query)
    
    return results if results else None

def transform (value=[]):
    value.append(1)
    return value
"""

analyzer = ComprehensiveCodeQualityAnalyzer (code)
print(analyzer.generate_comprehensive_report())

print("\\n" + "=" * 60)
print("ACTIONABLE ITEMS (PRIORITY ORDER)")
print("=" * 60)
for i, action in enumerate (analyzer.get_actionable_items(), 1):
    print(f"{i}. {action}")
\`\`\`

**Exercise Tasks:**
1. Add more bug patterns (type errors, logic errors)
2. Implement code metrics (coupling, cohesion)
3. Build a custom rule engine for project-specific checks
4. Add auto-fix suggestions for common issues
5. Create visual reports (HTML output with charts)

## Common Pitfalls

### 1. False Positives

\`\`\`python
# ❌ Wrong: Too strict, many false positives
def check_complexity (func):
    if calculate_complexity (func) > 5:
        raise Error("Too complex!")  # Too strict!

# ✅ Correct: Reasonable thresholds with warnings
def check_complexity (func):
    complexity = calculate_complexity (func)
    if complexity > 15:
        return "error"
    elif complexity > 10:
        return "warning"
    return "ok"
\`\`\`

### 2. Missing Context

\`\`\`python
# ❌ Wrong: Doesn't consider context
def is_dangerous (func_name):
    return func_name in ['eval', 'exec']
    # Misses: safe_eval might be okay

# ✅ Correct: Consider context and usage
def is_dangerous (call_node):
    if is_eval_like (call_node):
        # Check if input is trusted
        if is_user_input (call_node.args[0]):
            return True
    return False
\`\`\`

### 3. Over-Complicating

\`\`\`python
# ❌ Wrong: Too complex analysis
def analyze_everything (code):
    # 100 different checks
    # Takes minutes to run
    # Overwhelming output

# ✅ Correct: Focus on actionable items
def analyze_key_issues (code):
    # Security, bugs, major smells
    # Fast execution
    # Prioritized output
\`\`\`

## Production Checklist

### Analysis Coverage
- [ ] Complexity metrics (cyclomatic, cognitive)
- [ ] Bug pattern detection
- [ ] Security vulnerability scanning
- [ ] Code smell identification
- [ ] Style checking

### Quality
- [ ] Minimize false positives
- [ ] Provide clear explanations
- [ ] Include remediation suggestions
- [ ] Prioritize by severity
- [ ] Consider context

### Performance
- [ ] Cache analysis results
- [ ] Support incremental analysis
- [ ] Parallel analysis of files
- [ ] Timeout for large files
- [ ] Profile analyzer performance

### Integration
- [ ] Provide multiple output formats
- [ ] Support custom rules
- [ ] Enable/disable specific checks
- [ ] Integration with CI/CD
- [ ] IDE plugin support

### Reporting
- [ ] Clear, actionable messages
- [ ] Severity levels
- [ ] Line/column numbers
- [ ] Code snippets
- [ ] Fix suggestions

## Summary

Static analysis enables comprehensive code quality checking:

- **Complexity Analysis**: Measure code complexity
- **Bug Detection**: Find common errors
- **Security Scanning**: Identify vulnerabilities
- **Code Smells**: Detect design issues
- **Quality Metrics**: Quantify maintainability

These capabilities are essential for AI coding tools like Cursor to provide real-time feedback, suggest improvements, and help developers write better code. Combined with AST understanding, static analysis enables intelligent, context-aware assistance that goes far beyond simple syntax checking.

In the next section, we'll explore type system understanding—how to work with Python's type hints for even more intelligent analysis.
`,
};

export default staticAnalysis;
