const buildingCodeUnderstandingEngine = {
  id: 'building-code-understanding-engine',
  title: 'Building a Code Understanding Engine',
  content: `
# Building a Code Understanding Engine

## Introduction

Now we bring together everything we've learned to build a complete code understanding engine—the foundation of tools like Cursor. This engine combines AST parsing, symbol resolution, type inference, documentation extraction, and all other techniques into a unified system that provides comprehensive code intelligence.

**What We're Building:**

A production-ready code understanding engine that:
- Parses and analyzes code in multiple languages
- Builds comprehensive symbol tables
- Infers types and tracks data flow
- Extracts and indexes documentation
- Detects issues and suggests improvements
- Provides context for LLMs
- Integrates with IDEs via LSP

This is your capstone project for this module.

## Deep Technical Explanation

### Architecture Overview

\`\`\`
┌─────────────────────────────────────────────────┐
│           Code Understanding Engine             │
├─────────────────────────────────────────────────┤
│                                                 │
│  ┌──────────────┐  ┌─────────────────────┐    │
│  │   Parser     │→ │   Symbol Table      │    │
│  │   Layer      │  │   Builder           │    │
│  └──────────────┘  └─────────────────────┘    │
│         ↓                    ↓                  │
│  ┌──────────────┐  ┌─────────────────────┐    │
│  │   AST        │→ │   Type Inference    │    │
│  │   Cache      │  │   Engine            │    │
│  └──────────────┘  └─────────────────────┘    │
│         ↓                    ↓                  │
│  ┌──────────────┐  ┌─────────────────────┐    │
│  │   Control    │→ │   Static Analysis   │    │
│  │   Flow       │  │   Engine            │    │
│  └──────────────┘  └─────────────────────┘    │
│         ↓                    ↓                  │
│  ┌──────────────┐  ┌─────────────────────┐    │
│  │   Doc        │→ │   LLM Context       │    │
│  │   Extractor  │  │   Builder           │    │
│  └──────────────┘  └─────────────────────┘    │
│                                                 │
│         API: query(), analyze(), suggest()     │
└─────────────────────────────────────────────────┘
\`\`\`

### Core Components

**1. Parser Layer:**
- Multi-language support (tree-sitter)
- Incremental parsing
- Error recovery
- AST caching

**2. Analysis Layer:**
- Symbol tables
- Type inference
- Control/data flow
- Cross-references

**3. Intelligence Layer:**
- Code quality metrics
- Clone detection
- Pattern recognition
- Documentation indexing

**4. Integration Layer:**
- LSP server
- LLM context generation
- IDE integration
- API endpoints

## Code Implementation

### Complete Code Understanding Engine

\`\`\`python
import ast
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from pathlib import Path
import hashlib
import json

@dataclass
class CodeInsight:
    """Complete insight about a code location."""
    symbol_info: Optional[Dict] = None
    type_info: Optional[str] = None
    documentation: Optional[str] = None
    references: List[int] = None
    suggestions: List[str] = None
    diagnostics: List[Dict] = None

class CodeUnderstandingEngine:
    """
    Complete code understanding engine.
    Combines all analysis techniques for comprehensive code intelligence.
    """
    
    def __init__(self):
        # Storage
        self.files: Dict[str, str] = {}  # filepath -> content
        self.asts: Dict[str, ast.AST] = {}  # filepath -> AST
        self.ast_hashes: Dict[str, str] = {}  # filepath -> content hash
        
        # Analyzers (from previous sections)
        self.symbol_tables: Dict[str, Any] = {}
        self.type_info: Dict[str, Any] = {}
        self.doc_info: Dict[str, Any] = {}
        self.diagnostics: Dict[str, List] = {}
        
        # Caches
        self.analysis_cache: Dict[str, Dict] = {}
    
    # ==================== Core Functions ====================
    
    def add_file(self, filepath: str, content: str):
        """Add a file to the engine."""
        self.files[filepath] = content
        self._invalidate_cache(filepath)
    
    def update_file(self, filepath: str, content: str):
        """Update file content (incremental)."""
        if filepath not in self.files or self.files[filepath] != content:
            self.files[filepath] = content
            self._invalidate_cache(filepath)
    
    def remove_file(self, filepath: str):
        """Remove file from engine."""
        self.files.pop(filepath, None)
        self._invalidate_cache(filepath)
    
    def analyze_file(self, filepath: str) -> Dict[str, Any]:
        """
        Perform complete analysis on a file.
        Returns comprehensive analysis results.
        """
        if filepath not in self.files:
            return {'error': 'File not found'}
        
        content = self.files[filepath]
        
        # Check cache
        content_hash = self._hash_content(content)
        if filepath in self.analysis_cache:
            if self.analysis_cache[filepath].get('hash') == content_hash:
                return self.analysis_cache[filepath]['results']
        
        # Parse if needed
        if filepath not in self.asts or self.ast_hashes.get(filepath) != content_hash:
            try:
                self.asts[filepath] = ast.parse(content)
                self.ast_hashes[filepath] = content_hash
            except SyntaxError as e:
                return {
                    'error': 'Syntax error',
                    'message': str(e),
                    'line': e.lineno
                }
        
        tree = self.asts[filepath]
        
        # Run all analyses
        results = {
            'symbols': self._analyze_symbols(filepath, tree),
            'types': self._analyze_types(filepath, tree),
            'complexity': self._analyze_complexity(filepath, tree),
            'quality': self._analyze_quality(filepath, tree),
            'documentation': self._analyze_documentation(filepath, tree),
            'patterns': self._analyze_patterns(filepath, tree),
        }
        
        # Cache results
        self.analysis_cache[filepath] = {
            'hash': content_hash,
            'results': results
        }
        
        return results
    
    # ==================== Analysis Methods ====================
    
    def _analyze_symbols(self, filepath: str, tree: ast.AST) -> Dict:
        """Build symbol table."""
        from dataclasses import asdict
        
        # Use SymbolTableBuilder from previous sections
        builder = SymbolTableBuilder()
        builder.visit(tree)
        
        symbols = {}
        for key, symbol in builder.symbols.items():
            symbols[key] = {
                'name': symbol.name,
                'type': symbol.type.value,
                'line': symbol.defined_at,
                'scope': symbol.scope,
                'references': symbol.references
            }
        
        return symbols
    
    def _analyze_types(self, filepath: str, tree: ast.AST) -> Dict:
        """Infer and extract types."""
        # Use TypeAnnotationExtractor + TypeInferenceEngine
        extractor = TypeAnnotationExtractor()
        extractor.visit(tree)
        
        inferencer = TypeInferenceEngine()
        inferencer.visit(tree)
        
        return {
            'annotated': {k: v.type_annotation for k, v in extractor.types.items()},
            'inferred': inferencer.inferred_types
        }
    
    def _analyze_complexity(self, filepath: str, tree: ast.AST) -> Dict:
        """Calculate complexity metrics."""
        analyzer = ComplexityAnalyzer()
        analyzer.visit(tree)
        
        metrics = {}
        for name, complexity in analyzer.functions.items():
            metrics[name] = {
                'cyclomatic': complexity.cyclomatic,
                'cognitive': complexity.cognitive,
                'lines': complexity.lines
            }
        
        return metrics
    
    def _analyze_quality(self, filepath: str, tree: ast.AST) -> Dict:
        """Run quality checks."""
        # Bug detection
        bug_detector = BugPatternDetector()
        bug_detector.visit(tree)
        
        # Code smells
        smell_detector = CodeSmellDetector()
        smell_detector.visit(tree)
        
        # Security
        security_scanner = SecurityScanner()
        security_scanner.visit(tree)
        
        return {
            'bugs': len(bug_detector.issues),
            'smells': len(smell_detector.smells),
            'security_issues': len(security_scanner.issues),
            'issues': [
                {'type': 'bug', 'severity': i.severity, 'line': i.line, 'message': i.message}
                for i in bug_detector.issues
            ] + [
                {'type': 'smell', 'impact': s.impact, 'line': s.line, 'message': s.description}
                for s in smell_detector.smells
            ] + [
                {'type': 'security', 'severity': i.severity, 'line': i.line, 'message': i.description}
                for i in security_scanner.issues
            ]
        }
    
    def _analyze_documentation(self, filepath: str, tree: ast.AST) -> Dict:
        """Extract documentation."""
        extractor = DocstringExtractor()
        extractor.visit(tree)
        
        docs = {}
        for name, doc_info in extractor.docstrings.items():
            docs[name] = {
                'summary': doc_info.summary,
                'parameters': doc_info.parameters,
                'returns': doc_info.returns
            }
        
        return docs
    
    def _analyze_patterns(self, filepath: str, tree: ast.AST) -> Dict:
        """Detect code patterns."""
        extractor = CodePatternExtractor()
        extractor.visit(tree)
        
        patterns = {}
        for pattern_type, pattern in extractor.patterns.items():
            patterns[pattern_type] = {
                'occurrences': pattern.occurrences,
                'description': pattern.description
            }
        
        return patterns
    
    # ==================== Query Interface ====================
    
    def get_symbol_at_position(self, filepath: str, line: int, character: int) -> Optional[CodeInsight]:
        """
        Get complete information about symbol at position.
        This is what IDE uses for hover, completion, etc.
        """
        if filepath not in self.files:
            return None
        
        # Ensure file is analyzed
        analysis = self.analyze_file(filepath)
        
        # Find symbol at position
        symbols = analysis.get('symbols', {})
        
        # Find closest symbol to position
        closest = None
        min_distance = float('inf')
        
        for symbol_key, symbol in symbols.items():
            symbol_line = symbol['line']
            distance = abs(symbol_line - line)
            if distance < min_distance:
                min_distance = distance
                closest = symbol
        
        if not closest:
            return None
        
        # Build comprehensive insight
        insight = CodeInsight(
            symbol_info=closest,
            type_info=self._get_type_for_symbol(filepath, closest['name']),
            documentation=self._get_doc_for_symbol(filepath, closest['name']),
            references=closest.get('references', []),
            suggestions=self._get_suggestions_for_symbol(filepath, closest),
            diagnostics=self._get_diagnostics_for_line(filepath, line)
        )
        
        return insight
    
    def get_completions(self, filepath: str, line: int, character: int) -> List[Dict]:
        """
        Get completion suggestions for position.
        """
        analysis = self.analyze_file(filepath)
        symbols = analysis.get('symbols', {})
        
        # Return all symbols as completions (simplified)
        completions = []
        for symbol_key, symbol in symbols.items():
            completions.append({
                'label': symbol['name'],
                'kind': symbol['type'],
                'detail': f"{symbol['type']} at line {symbol['line']}"
            })
        
        return completions
    
    def get_diagnostics(self, filepath: str) -> List[Dict]:
        """Get all diagnostics for file."""
        analysis = self.analyze_file(filepath)
        quality = analysis.get('quality', {})
        return quality.get('issues', [])
    
    def suggest_refactorings(self, filepath: str) -> List[Dict]:
        """Suggest refactoring opportunities."""
        analysis = self.analyze_file(filepath)
        
        suggestions = []
        
        # High complexity functions
        complexity = analysis.get('complexity', {})
        for func_name, metrics in complexity.items():
            if metrics['cyclomatic'] > 10:
                suggestions.append({
                    'type': 'reduce_complexity',
                    'function': func_name,
                    'current_complexity': metrics['cyclomatic'],
                    'suggestion': f"Function '{func_name}' has high complexity. Consider breaking it into smaller functions."
                })
        
        # Patterns that could be refactored
        patterns = analysis.get('patterns', {})
        if patterns.get('accumulator_loop', {}).get('occurrences', 0) >= 3:
            suggestions.append({
                'type': 'extract_pattern',
                'pattern': 'accumulator_loop',
                'suggestion': "Multiple accumulator loops detected. Consider extracting to a helper function."
            })
        
        return suggestions
    
    # ==================== LLM Context Generation ====================
    
    def generate_llm_context(self, filepath: str, focus_line: Optional[int] = None) -> str:
        """
        Generate rich context for LLM.
        This is what Cursor sends to ChatGPT/Claude.
        """
        analysis = self.analyze_file(filepath)
        
        lines = [f"# Context for {filepath}\\n"]
        
        # File-level summary
        symbols = analysis.get('symbols', {})
        docs = analysis.get('documentation', {})
        
        lines.append("## Code Structure")
        
        # Classes
        classes = {k: v for k, v in symbols.items() if v['type'] == 'class'}
        if classes:
            lines.append("### Classes")
            for name, info in list(classes.items())[:5]:
                doc = docs.get(name, {}).get('summary', 'No documentation')
                lines.append(f"- {name} (line {info['line']}): {doc}")
        
        # Functions
        functions = {k: v for k, v in symbols.items() if v['type'] == 'function'}
        if functions:
            lines.append("\\n### Functions")
            for name, info in list(functions.items())[:10]:
                doc = docs.get(name, {}).get('summary', 'No documentation')
                lines.append(f"- {name} (line {info['line']}): {doc}")
        
        # Types (if available)
        types = analysis.get('types', {}).get('annotated', {})
        if types:
            lines.append("\\n### Type Information")
            for name, type_str in list(types.items())[:5]:
                lines.append(f"- {name}: {type_str}")
        
        # Quality issues
        quality = analysis.get('quality', {})
        issues = quality.get('issues', [])
        if issues:
            lines.append(f"\\n### Issues ({len(issues)} total)")
            for issue in issues[:5]:
                lines.append(f"- Line {issue['line']}: {issue['message']}")
        
        # Focus area (if specified)
        if focus_line:
            lines.append(f"\\n### Focus Area (Line {focus_line})")
            # Add nearby symbols
            nearby = {
                k: v for k, v in symbols.items()
                if abs(v['line'] - focus_line) < 10
            }
            if nearby:
                lines.append("Nearby symbols:")
                for name, info in nearby.items():
                    lines.append(f"- {name} ({info['type']}) at line {info['line']}")
        
        return "\\n".join(lines)
    
    # ==================== Utility Methods ====================
    
    def _hash_content(self, content: str) -> str:
        """Generate hash of content for caching."""
        return hashlib.md5(content.encode()).hexdigest()
    
    def _invalidate_cache(self, filepath: str):
        """Invalidate cache for file."""
        self.analysis_cache.pop(filepath, None)
        self.asts.pop(filepath, None)
        self.ast_hashes.pop(filepath, None)
    
    def _get_type_for_symbol(self, filepath: str, symbol_name: str) -> Optional[str]:
        """Get type information for symbol."""
        analysis = self.analyze_file(filepath)
        types = analysis.get('types', {})
        
        # Check annotated types
        annotated = types.get('annotated', {})
        for key, type_str in annotated.items():
            if symbol_name in key:
                return type_str
        
        # Check inferred types
        inferred = types.get('inferred', {})
        for key, type_str in inferred.items():
            if symbol_name in key:
                return type_str
        
        return None
    
    def _get_doc_for_symbol(self, filepath: str, symbol_name: str) -> Optional[str]:
        """Get documentation for symbol."""
        analysis = self.analyze_file(filepath)
        docs = analysis.get('documentation', {})
        
        doc_info = docs.get(symbol_name)
        if doc_info:
            return doc_info.get('summary')
        
        return None
    
    def _get_suggestions_for_symbol(self, filepath: str, symbol: Dict) -> List[str]:
        """Get suggestions related to symbol."""
        suggestions = []
        
        # No documentation
        if not self._get_doc_for_symbol(filepath, symbol['name']):
            suggestions.append("Add docstring")
        
        # No type hints
        if not self._get_type_for_symbol(filepath, symbol['name']):
            suggestions.append("Add type hints")
        
        return suggestions
    
    def _get_diagnostics_for_line(self, filepath: str, line: int) -> List[Dict]:
        """Get diagnostics affecting a specific line."""
        all_diagnostics = self.get_diagnostics(filepath)
        return [d for d in all_diagnostics if d.get('line') == line]
    
    # ==================== Statistics & Reporting ====================
    
    def get_codebase_statistics(self) -> Dict[str, Any]:
        """Get statistics about the entire codebase."""
        stats = {
            'total_files': len(self.files),
            'total_functions': 0,
            'total_classes': 0,
            'total_lines': 0,
            'documented_functions': 0,
            'average_complexity': 0,
            'total_issues': 0,
        }
        
        complexities = []
        
        for filepath in self.files:
            analysis = self.analyze_file(filepath)
            
            symbols = analysis.get('symbols', {})
            stats['total_functions'] += sum(1 for s in symbols.values() if s['type'] == 'function')
            stats['total_classes'] += sum(1 for s in symbols.values() if s['type'] == 'class')
            
            docs = analysis.get('documentation', {})
            stats['documented_functions'] += len(docs)
            
            complexity = analysis.get('complexity', {})
            for metrics in complexity.values():
                complexities.append(metrics['cyclomatic'])
            
            quality = analysis.get('quality', {})
            stats['total_issues'] += len(quality.get('issues', []))
            
            stats['total_lines'] += len(self.files[filepath].split('\\n'))
        
        if complexities:
            stats['average_complexity'] = sum(complexities) / len(complexities)
        
        return stats

# =================================================================
# Example Usage: Complete Workflow
# =================================================================

# Create engine
engine = CodeUnderstandingEngine()

# Add files
engine.add_file('user_service.py', """
from typing import Optional

class UserService:
    ''Manage user operations.''
    
    def __init__(self, db):
        self.db = db
    
    def get_user(self, user_id: int) -> Optional[dict]:
        ''Get user by ID.
        
        Args:
            user_id: User ID to fetch
            
        Returns:
            User data or None if not found
        ''
        return self.db.query(user_id)
    
    def create_user(self, name: str, email: str):
        # Missing docstring!
        # TODO: Add validation
        user = {'name': name, 'email': email}
        self.db.save(user)
        return user
""")

# Analyze file
print("=== Complete Analysis ===")
analysis = engine.analyze_file('user_service.py')
print(json.dumps(analysis, indent=2, default=str))

# Get symbol information
print("\\n=== Symbol at Line 12 ===")
insight = engine.get_symbol_at_position('user_service.py', 12, 0)
if insight:
    print(f"Symbol: {insight.symbol_info}")
    print(f"Type: {insight.type_info}")
    print(f"Documentation: {insight.documentation}")
    print(f"Suggestions: {insight.suggestions}")

# Get completions
print("\\n=== Completions ===")
completions = engine.get_completions('user_service.py', 15, 10)
for comp in completions[:5]:
    print(f"  {comp['label']} ({comp['kind']})")

# Get diagnostics
print("\\n=== Diagnostics ===")
diagnostics = engine.get_diagnostics('user_service.py')
for diag in diagnostics:
    print(f"  Line {diag['line']}: {diag['message']}")

# Get refactoring suggestions
print("\\n=== Refactoring Suggestions ===")
suggestions = engine.suggest_refactorings('user_service.py')
for sug in suggestions:
    print(f"  {sug['type']}: {sug['suggestion']}")

# Generate LLM context
print("\\n=== LLM Context ===")
context = engine.generate_llm_context('user_service.py', focus_line=16)
print(context)

# Get codebase statistics
print("\\n=== Codebase Statistics ===")
stats = engine.get_codebase_statistics()
for key, value in stats.items():
    print(f"  {key}: {value}")
\`\`\`

## Real-World Case Study: How Cursor Uses Code Understanding

Cursor's code understanding engine powers all its features:

**1. Real-Time Intelligence:**
- As you type, engine continuously analyzes
- Updates symbol tables incrementally
- Provides instant feedback and suggestions
- All analysis cached for performance

**2. Multi-File Understanding:**
- Analyzes entire codebase
- Tracks cross-file dependencies
- Maintains global symbol index
- Provides project-wide refactoring

**3. LLM Integration:**
- Generates rich context from analysis
- Sends relevant symbols and types
- Includes documentation and patterns
- Enables accurate code generation

**4. Intelligent Suggestions:**
- Combines all analysis results
- Understands code patterns
- Suggests improvements
- Learns from codebase

## Production Checklist

### Core Functionality
- [ ] Multi-language support
- [ ] Incremental parsing
- [ ] Symbol table management
- [ ] Type inference
- [ ] Documentation extraction
- [ ] Quality analysis

### Performance
- [ ] AST caching
- [ ] Analysis result caching
- [ ] Incremental updates
- [ ] Parallel file processing
- [ ] Memory management
- [ ] Fast query response (<100ms)

### Robustness
- [ ] Error recovery
- [ ] Graceful degradation
- [ ] Handle syntax errors
- [ ] Validate all inputs
- [ ] Comprehensive logging
- [ ] Monitoring and metrics

### Features
- [ ] Completion suggestions
- [ ] Hover information
- [ ] Go-to-definition
- [ ] Find references
- [ ] Diagnostics
- [ ] Refactoring suggestions

### Integration
- [ ] LSP server
- [ ] IDE plugins
- [ ] CLI interface
- [ ] REST API
- [ ] WebSocket support
- [ ] LLM context generation

### Quality
- [ ] Comprehensive tests
- [ ] Performance benchmarks
- [ ] Real-world validation
- [ ] Documentation
- [ ] Example usage
- [ ] Migration guides

## Summary

Building a complete code understanding engine brings together:

- **AST Parsing**: Foundation for all analysis
- **Symbol Resolution**: Understanding names and references
- **Type Inference**: Knowing what types exist
- **Quality Analysis**: Finding issues and patterns
- **Documentation**: Extracting human context
- **Integration**: Providing useful interfaces

This engine forms the core of modern AI coding tools like Cursor, enabling intelligent code suggestions, refactoring, and generation that understand not just syntax, but the structure, types, patterns, and intent of your entire codebase.

**Congratulations!** You've completed Module 4 and built a production-grade code understanding engine. You now have the foundation to build sophisticated code analysis and generation tools that rival professional IDEs and AI coding assistants.

## Next Steps

With this code understanding engine, you can:
- Build IDE plugins with intelligent features
- Create custom linters and analysis tools
- Generate code that fits your codebase
- Build AI coding assistants
- Implement automated refactoring
- Create documentation generators

The code understanding engine you've built is the foundation for the next modules where we'll explore code generation, tool use, and building complete AI coding products like Cursor.
`,
};

export default buildingCodeUnderstandingEngine;
