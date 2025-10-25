const codeSimilarityDetection = {
  id: 'code-similarity-detection',
  title: 'Code Similarity & Clone Detection',
  content: `
# Code Similarity & Clone Detection

## Introduction

Duplicate and similar code (code clones) indicate refactoring opportunities and can hide bugs. When Cursor suggests "this code is similar to X" or offers to extract common patterns, it's using clone detection. Modern AI tools need to identify similar code to suggest consolidation, find examples, and maintain DRY (Don't Repeat Yourself) principles.

**Why Clone Detection Matters:**

AI coding tools need to:
- Find duplicate code for refactoring
- Locate similar examples for context
- Detect copy-paste errors
- Suggest code reuse opportunities
- Identify pattern candidates
- Maintain codebase quality

This section teaches you to build these capabilities.

## Deep Technical Explanation

### Types of Code Clones

**Type I: Exact Clones**
Identical code except for whitespace and comments
\`\`\`python
# Clone 1
def process_a (data):
    result = transform (data)
    return result

# Clone 2  
def process_b (data):
    result = transform (data)
    return result
\`\`\`

**Type II: Renamed Clones**
Identical structure with different names
\`\`\`python
# Clone 1
def calculate_total (items):
    sum = 0
    for item in items:
        sum += item
    return sum

# Clone 2
def compute_sum (values):
    total = 0
    for value in values:
        total += value
    return total
\`\`\`

**Type III: Near-Miss Clones**
Similar with minor modifications
\`\`\`python
# Clone 1
def validate_user (user):
    if not user.name:
        return False
    if not user.email:
        return False
    return True

# Clone 2
def validate_product (product):
    if not product.name:
        return False
    if not product.price:
        return False
    if not product.category:
        return False
    return True
\`\`\`

**Type IV: Semantic Clones**
Different syntax, same functionality
\`\`\`python
# Clone 1
def sum_list (items):
    total = 0
    for item in items:
        total += item
    return total

# Clone 2
def sum_list (items):
    return sum (items)
\`\`\`

### Similarity Metrics

**Token-Based Similarity:**
- Compare token sequences
- Ignore identifiers
- Fast but less accurate

**AST-Based Similarity:**
- Compare tree structures
- More accurate
- Handles Type II clones

**Semantic Similarity:**
- Compare behavior
- Most accurate
- Computationally expensive

## Code Implementation

### Token-Based Clone Detector

\`\`\`python
import ast
import hashlib
from dataclasses import dataclass
from typing import List, Set, Dict
from collections import defaultdict

@dataclass
class CodeFragment:
    function_name: str
    start_line: int
    end_line: int
    token_hash: str
    tokens: List[str]

class TokenBasedCloneDetector (ast.NodeVisitor):
    """
    Detect code clones using token-based comparison.
    Fast and effective for Type I clones.
    """
    
    def __init__(self, min_tokens: int = 20):
        self.min_tokens = min_tokens
        self.fragments: List[CodeFragment] = []
        self.clones: List[tuple] = []
    
    def visit_FunctionDef (self, node: ast.FunctionDef):
        """Extract and hash function tokens."""
        # Get function tokens
        tokens = self._extract_tokens (node)
        
        if len (tokens) >= self.min_tokens:
            # Normalize tokens (remove identifiers for Type II detection)
            normalized_tokens = self._normalize_tokens (tokens)
            
            # Create hash
            token_str = ' '.join (normalized_tokens)
            token_hash = hashlib.md5(token_str.encode()).hexdigest()
            
            fragment = CodeFragment(
                function_name=node.name,
                start_line=node.lineno,
                end_line=node.end_lineno or node.lineno,
                token_hash=token_hash,
                tokens=tokens
            )
            
            self.fragments.append (fragment)
        
        self.generic_visit (node)
    
    def _extract_tokens (self, node: ast.FunctionDef) -> List[str]:
        """Extract tokens from function AST."""
        tokens = []
        
        for child in ast.walk (node):
            # Add node type as token
            tokens.append (type (child).__name__)
            
            # Add operator types
            if isinstance (child, ast.operator):
                tokens.append (type (child).__name__)
            
            # Add constant values (but not names)
            if isinstance (child, ast.Constant):
                if isinstance (child.value, (int, float, str, bool)):
                    tokens.append (str (type (child.value).__name__))
        
        return tokens
    
    def _normalize_tokens (self, tokens: List[str]) -> List[str]:
        """Normalize tokens to detect renamed clones."""
        # Remove specific identifiers, keep structure
        normalized = []
        for token in tokens:
            if token in ['Name', 'arg']:
                normalized.append('IDENTIFIER')
            else:
                normalized.append (token)
        return normalized
    
    def find_clones (self):
        """Find all clones by matching hashes."""
        # Group by hash
        by_hash = defaultdict (list)
        for fragment in self.fragments:
            by_hash[fragment.token_hash].append (fragment)
        
        # Find groups with multiple fragments
        for hash_val, fragments in by_hash.items():
            if len (fragments) > 1:
                # These are clones
                for i in range (len (fragments)):
                    for j in range (i + 1, len (fragments)):
                        self.clones.append((fragments[i], fragments[j]))
    
    def calculate_similarity (self, frag1: CodeFragment, frag2: CodeFragment) -> float:
        """Calculate similarity between two fragments (0-1)."""
        if frag1.token_hash == frag2.token_hash:
            return 1.0
        
        # Calculate token overlap
        tokens1 = set (frag1.tokens)
        tokens2 = set (frag2.tokens)
        
        intersection = len (tokens1 & tokens2)
        union = len (tokens1 | tokens2)
        
        return intersection / union if union > 0 else 0.0
    
    def generate_report (self) -> str:
        """Generate clone detection report."""
        if not self.clones:
            return "✅ No code clones detected!"
        
        lines = [f"=== Found {len (self.clones)} Code Clone Pairs ===\\n"]
        
        for i, (frag1, frag2) in enumerate (self.clones, 1):
            lines.append (f"Clone Pair #{i}:")
            lines.append (f"  Function 1: {frag1.function_name} (lines {frag1.start_line}-{frag1.end_line})")
            lines.append (f"  Function 2: {frag2.function_name} (lines {frag2.start_line}-{frag2.end_line})")
            similarity = self.calculate_similarity (frag1, frag2)
            lines.append (f"  Similarity: {similarity*100:.1f}%")
            lines.append("")
        
        return "\\n".join (lines)

# Example usage
code = """
def calculate_user_total (users):
    total = 0
    for user in users:
        total += user.points
    return total

def calculate_product_total (products):
    total = 0
    for product in products:
        total += product.price
    return total

def process_data (items):
    result = []
    for item in items:
        if item.valid:
            result.append (item)
    return result

def filter_items (elements):
    output = []
    for element in elements:
        if element.valid:
            output.append (element)
    return output
"""

detector = TokenBasedCloneDetector()
tree = ast.parse (code)
detector.visit (tree)
detector.find_clones()

print(detector.generate_report())
\`\`\`

### AST-Based Similarity Detector

\`\`\`python
import ast
from typing import List, Tuple

class ASTSimilarityDetector:
    """
    Detect similar code using AST comparison.
    More accurate than token-based, handles structure.
    """
    
    def __init__(self, threshold: float = 0.8):
        self.threshold = threshold
    
    def compare_functions (self, func1: ast.FunctionDef, func2: ast.FunctionDef) -> float:
        """
        Compare two functions and return similarity score (0-1).
        
        Args:
            func1: First function AST
            func2: Second function AST
            
        Returns:
            Similarity score from 0 (different) to 1 (identical)
        """
        # Compare AST structure
        similarity = self._compare_ast_nodes (func1, func2)
        return similarity
    
    def _compare_ast_nodes (self, node1: ast.AST, node2: ast.AST) -> float:
        """Recursively compare AST nodes."""
        # Same type?
        if type (node1) != type (node2):
            return 0.0
        
        # Leaf nodes (constants, names)
        if isinstance (node1, ast.Constant):
            # Compare constant types, not values
            return 1.0 if type (node1.value) == type (node2.value) else 0.5
        
        if isinstance (node1, ast.Name):
            # Names don't have to match for similarity
            return 1.0
        
        # Compare children
        children1 = list (ast.iter_child_nodes (node1))
        children2 = list (ast.iter_child_nodes (node2))
        
        if len (children1) != len (children2):
            # Different structure
            return 0.5
        
        if not children1:
            # Leaf node with same type
            return 1.0
        
        # Compare all children
        similarities = []
        for child1, child2 in zip (children1, children2):
            sim = self._compare_ast_nodes (child1, child2)
            similarities.append (sim)
        
        # Average similarity of children
        return sum (similarities) / len (similarities)
    
    def find_similar_functions (self, functions: List[ast.FunctionDef]) -> List[Tuple[str, str, float]]:
        """
        Find all pairs of similar functions.
        
        Returns:
            List of (func1_name, func2_name, similarity_score)
        """
        similar_pairs = []
        
        for i in range (len (functions)):
            for j in range (i + 1, len (functions)):
                func1 = functions[i]
                func2 = functions[j]
                
                similarity = self.compare_functions (func1, func2)
                
                if similarity >= self.threshold:
                    similar_pairs.append((
                        func1.name,
                        func2.name,
                        similarity
                    ))
        
        return similar_pairs

# Example usage
code = """
def calculate_sum (numbers):
    total = 0
    for num in numbers:
        total = total + num
    return total

def compute_total (values):
    sum_val = 0
    for val in values:
        sum_val = sum_val + val
    return sum_val

def different_function (x, y):
    if x > y:
        return x
    else:
        return y
"""

tree = ast.parse (code)
functions = [node for node in ast.walk (tree) if isinstance (node, ast.FunctionDef)]

detector = ASTSimilarityDetector (threshold=0.8)
similar = detector.find_similar_functions (functions)

print("=== Similar Functions ===\\n")
for func1, func2, score in similar:
    print(f"{func1} ↔ {func2}: {score*100:.1f}% similar")
\`\`\`

### Code Pattern Extractor

\`\`\`python
import ast
from collections import Counter
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class CodePattern:
    pattern_type: str
    description: str
    occurrences: int
    examples: List[Tuple[str, int]]  # (function_name, line_number)

class CodePatternExtractor (ast.NodeVisitor):
    """
    Extract common code patterns from codebase.
    Useful for identifying refactoring opportunities.
    """
    
    def __init__(self):
        self.patterns: Dict[str, CodePattern] = {}
        self.current_function: Optional[str] = None
    
    def visit_FunctionDef (self, node: ast.FunctionDef):
        """Track current function."""
        old_func = self.current_function
        self.current_function = node.name
        self.generic_visit (node)
        self.current_function = old_func
    
    def visit_For (self, node: ast.For):
        """Detect for-loop patterns."""
        # Pattern: Accumulator loop
        if self._is_accumulator_pattern (node):
            self._record_pattern(
                'accumulator_loop',
                'For loop with accumulator variable',
                node.lineno
            )
        
        # Pattern: Filter loop
        if self._is_filter_pattern (node):
            self._record_pattern(
                'filter_loop',
                'For loop filtering items',
                node.lineno
            )
        
        self.generic_visit (node)
    
    def visit_If (self, node: ast.If):
        """Detect if-statement patterns."""
        # Pattern: Guard clause
        if self._is_guard_clause (node):
            self._record_pattern(
                'guard_clause',
                'Early return guard clause',
                node.lineno
            )
        
        # Pattern: Null check
        if self._is_null_check (node):
            self._record_pattern(
                'null_check',
                'Check for None/null',
                node.lineno
            )
        
        self.generic_visit (node)
    
    def visit_Try (self, node: ast.Try):
        """Detect try-except patterns."""
        # Pattern: Resource management
        self._record_pattern(
            'try_except',
            'Exception handling block',
            node.lineno
        )
        
        self.generic_visit (node)
    
    def _is_accumulator_pattern (self, node: ast.For) -> bool:
        """Check if for-loop uses accumulator pattern."""
        # Look for: total = 0; total += x pattern
        for stmt in node.body:
            if isinstance (stmt, ast.AugAssign):
                return True
        return False
    
    def _is_filter_pattern (self, node: ast.For) -> bool:
        """Check if for-loop filters items."""
        # Look for: if condition: result.append (item)
        for stmt in node.body:
            if isinstance (stmt, ast.If):
                for if_stmt in stmt.body:
                    if isinstance (if_stmt, ast.Expr):
                        if isinstance (if_stmt.value, ast.Call):
                            if isinstance (if_stmt.value.func, ast.Attribute):
                                if if_stmt.value.func.attr == 'append':
                                    return True
        return False
    
    def _is_guard_clause (self, node: ast.If) -> bool:
        """Check if if-statement is a guard clause."""
        # Guard clause: early return
        if node.body:
            first_stmt = node.body[0]
            return isinstance (first_stmt, ast.Return)
        return False
    
    def _is_null_check (self, node: ast.If) -> bool:
        """Check if if-statement checks for None."""
        test = node.test
        if isinstance (test, ast.Compare):
            for comparator in test.comparators:
                if isinstance (comparator, ast.Constant):
                    if comparator.value is None:
                        return True
        if isinstance (test, ast.UnaryOp):
            if isinstance (test.op, ast.Not):
                return True
        return False
    
    def _record_pattern (self, pattern_type: str, description: str, line: int):
        """Record occurrence of a pattern."""
        if pattern_type not in self.patterns:
            self.patterns[pattern_type] = CodePattern(
                pattern_type=pattern_type,
                description=description,
                occurrences=0,
                examples=[]
            )
        
        pattern = self.patterns[pattern_type]
        pattern.occurrences += 1
        
        if self.current_function:
            pattern.examples.append((self.current_function, line))
    
    def generate_report (self) -> str:
        """Generate pattern analysis report."""
        if not self.patterns:
            return "No patterns detected"
        
        lines = ["=== Code Pattern Analysis ===\\n"]
        
        # Sort by occurrences
        sorted_patterns = sorted(
            self.patterns.values(),
            key=lambda p: p.occurrences,
            reverse=True
        )
        
        for pattern in sorted_patterns:
            lines.append (f"Pattern: {pattern.pattern_type}")
            lines.append (f"  Description: {pattern.description}")
            lines.append (f"  Occurrences: {pattern.occurrences}")
            
            if pattern.examples:
                lines.append("  Examples:")
                for func_name, line in pattern.examples[:3]:  # Show first 3
                    lines.append (f"    - {func_name} (line {line})")
            
            lines.append("")
        
        return "\\n".join (lines)
    
    def suggest_refactorings (self) -> List[str]:
        """Suggest refactorings based on patterns."""
        suggestions = []
        
        for pattern_type, pattern in self.patterns.items():
            if pattern.occurrences >= 3:
                if pattern_type == 'accumulator_loop':
                    suggestions.append(
                        f"Consider using sum() or reduce() instead of {pattern.occurrences} "
                        f"accumulator loops"
                    )
                elif pattern_type == 'filter_loop':
                    suggestions.append(
                        f"Consider using list comprehensions instead of {pattern.occurrences} "
                        f"filter loops"
                    )
        
        return suggestions

# Example usage
code = """
def sum_numbers (numbers):
    total = 0
    for num in numbers:
        total += num
    return total

def sum_scores (scores):
    total = 0
    for score in scores:
        total += score
    return total

def filter_valid (items):
    result = []
    for item in items:
        if item.valid:
            result.append (item)
    return result

def validate_user (user):
    if user is None:
        return False
    if not user.name:
        return False
    return True

def process_data (data):
    if data is None:
        return None
    
    try:
        result = transform (data)
        return result
    except Exception as e:
        log_error (e)
        return None
"""

extractor = CodePatternExtractor()
tree = ast.parse (code)
extractor.visit (tree)

print(extractor.generate_report())

print("\\n=== Refactoring Suggestions ===")
for suggestion in extractor.suggest_refactorings():
    print(f"  💡 {suggestion}")
\`\`\`

### Clone Refactoring Suggester

\`\`\`python
class CloneRefactoringSuggester:
    """
    Suggest refactoring for detected clones.
    Helps eliminate code duplication.
    """
    
    def __init__(self, clones: List[Tuple[CodeFragment, CodeFragment]]):
        self.clones = clones
    
    def suggest_extract_function (self, clone_pair: Tuple[CodeFragment, CodeFragment]) -> str:
        """Suggest extracting common code into function."""
        frag1, frag2 = clone_pair
        
        suggestion = f"""
Refactoring Suggestion: Extract Function

Cloned Functions:
  - {frag1.function_name} (lines {frag1.start_line}-{frag1.end_line})
  - {frag2.function_name} (lines {frag2.start_line}-{frag2.end_line})

Suggestion:
  1. Extract common logic into new function:
     def extracted_logic (param1, param2):
         # Common code here
         pass
  
  2. Replace both functions to call the extracted function
  
  3. Add parameters for any differences between the clones

Benefits:
  - Reduce code duplication
  - Easier maintenance
  - Single source of truth
"""
        return suggestion
    
    def suggest_template_method (self, clones: List[CodeFragment]) -> str:
        """Suggest template method pattern for similar functions."""
        if len (clones) < 2:
            return ""
        
        names = [c.function_name for c in clones]
        
        suggestion = f"""
Refactoring Suggestion: Template Method Pattern

Similar Functions:
  {", ".join (names)}

Suggestion:
  Create a base template with common structure:
  
  def template_method (data, validator):
      results = []
      for item in data:
          if validator (item):
              results.append (process (item))
      return results
  
  Then create specific implementations:
  
  def {names[0]}(data):
      return template_method (data, validate_for_{names[0]})

Benefits:
  - Eliminates duplication
  - Easier to add new variants
  - Clear separation of concerns
"""
        return suggestion
    
    def generate_all_suggestions (self) -> str:
        """Generate all refactoring suggestions."""
        lines = ["=== Refactoring Suggestions for Clones ===\\n"]
        
        for i, clone_pair in enumerate (self.clones, 1):
            lines.append (f"Clone Pair #{i}:")
            suggestion = self.suggest_extract_function (clone_pair)
            lines.append (suggestion)
        
        return "\\n".join (lines)

# Example: Use with detected clones
detector = TokenBasedCloneDetector()
tree = ast.parse (code)
detector.visit (tree)
detector.find_clones()

if detector.clones:
    suggester = CloneRefactoringSuggester (detector.clones)
    print(suggester.generate_all_suggestions())
\`\`\`

## Real-World Case Study: How Cursor Uses Clone Detection

Cursor leverages clone detection for intelligent suggestions:

**1. Duplicate Code Warnings:**
\`\`\`python
def calculate_user_score (user):
    total = 0
    for activity in user.activities:
        total += activity.points
    return total

# When you write similar code:
def calculate_team_score (team):
    total = 0
    for member in team.members:
        total += member.points
    return total
    # Cursor: "⚠️ Similar to calculate_user_score"
\`\`\`

**2. Refactoring Suggestions:**
\`\`\`python
# Cursor detects pattern and suggests:
# "💡 3 functions use similar accumulator pattern.
#     Consider extracting to: calculate_total (items, key)"
\`\`\`

**3. Example Finding:**
\`\`\`python
# When you write:
def new_validation (data):
    # Cursor finds similar validation functions
    # Shows: "Similar to: validate_user, validate_product"
    # Offers to use same pattern
\`\`\`

**4. Pattern Recognition:**
\`\`\`python
# Cursor learns your patterns:
# "You often use guard clauses"
# "You prefer list comprehensions over for loops"
# Applies these patterns to generated code
\`\`\`

## Common Pitfalls

### 1. Too Strict Matching

\`\`\`python
# ❌ Wrong: Requires exact match
def is_clone (func1, func2):
    return ast.dump (func1) == ast.dump (func2)
    # Misses renamed clones

# ✅ Correct: Use similarity threshold
def is_clone (func1, func2):
    similarity = calculate_similarity (func1, func2)
    return similarity > 0.8  # 80% similar
\`\`\`

### 2. Ignoring Context

\`\`\`python
# ❌ Wrong: Flags intentional duplication
# These SHOULD be separate:
def validate_user_input (data):
    if not data:
        return False
    return True

def validate_api_response (data):
    if not data:
        return False
    return True
# Different contexts, intentional duplication

# ✅ Correct: Consider semantic meaning
\`\`\`

### 3. Missing Semantic Clones

\`\`\`python
# ❌ Wrong: Only checks syntax
def sum_list_v1(items):
    total = 0
    for item in items:
        total += item
    return total

def sum_list_v2(items):
    return sum (items)
# These are semantic clones!

# ✅ Correct: Also check behavior
\`\`\`

## Production Checklist

### Detection
- [ ] Support multiple clone types (I, II, III)
- [ ] Adjustable similarity thresholds
- [ ] Handle large codebases efficiently
- [ ] Provide line-level granularity
- [ ] Support cross-file detection

### Analysis
- [ ] Calculate similarity scores
- [ ] Identify clone families (>2 clones)
- [ ] Detect patterns across clones
- [ ] Measure clone coverage
- [ ] Track clone evolution

### Reporting
- [ ] Clear visualization of clones
- [ ] Similarity percentages
- [ ] Code snippets
- [ ] Refactoring suggestions
- [ ] Priority ranking

### Refactoring
- [ ] Suggest extract method
- [ ] Propose template patterns
- [ ] Generate refactored code
- [ ] Validate suggestions
- [ ] Track refactoring impact

### Performance
- [ ] Cache analysis results
- [ ] Incremental analysis
- [ ] Parallel processing
- [ ] Index-based lookup
- [ ] Memory-efficient algorithms

## Summary

Code similarity and clone detection enable intelligent code maintenance:

- **Clone Detection**: Find duplicate and similar code
- **Pattern Recognition**: Identify common patterns
- **Refactoring Suggestions**: Propose improvements
- **Code Quality**: Maintain DRY principles
- **Context Finding**: Locate similar examples

These capabilities allow AI coding tools like Cursor to help maintain code quality by identifying duplication, suggesting refactorings, and finding relevant examples throughout your codebase—essential for building maintainable software at scale.

In the next section, we'll explore the Language Server Protocol (LSP)—the standard for IDE features like go-to-definition and auto-complete.
`,
};

export default codeSimilarityDetection;
