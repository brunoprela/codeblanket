/**
 * Code Review & Bug Detection Section
 * Module 5: Building Code Generation Systems
 */

export const codereviewbugdetectionSection = {
  id: 'code-review-bug-detection',
  title: 'Code Review & Bug Detection',
  content: `# Code Review & Bug Detection

Master using LLMs to automatically review code, detect bugs, find security issues, and suggest improvements.

## Overview: Automated Code Review

LLMs can perform comprehensive code reviews:
- Detect bugs and logic errors
- Find security vulnerabilities
- Identify performance issues
- Suggest best practices
- Check code style
- Detect code smells

### Why Automated Review?

**Manual Review Limitations:**
- Time-consuming
- Inconsistent
- Misses subtle bugs
- Limited by reviewer knowledge

**AI-Powered Review:**
- Fast and comprehensive
- Consistent standards
- Catches subtle issues
- Learns from patterns

## Bug Detection

### Comprehensive Bug Detector

\`\`\`python
from openai import OpenAI
from typing import List, Dict
from dataclasses import dataclass

@dataclass
class Bug:
    """Represents a detected bug."""
    severity: str  # "critical", "high", "medium", "low"
    category: str  # "logic", "type", "null_pointer", etc.
    line: int
    description: str
    suggestion: str

class BugDetector:
    """Detect bugs in code using LLM."""
    
    def __init__(self):
        self.client = OpenAI()
    
    def detect_bugs(
        self,
        code: str,
        language: str = "python"
    ) -> List[Bug]:
        """Detect potential bugs in code."""
        
        prompt = f"""Analyze this {language} code for bugs:

\`\`\`{language}
{code}
\`\`\`

Find:
1. Logic errors (off-by-one, wrong conditions, etc.)
2. Null/None pointer issues
3. Type errors
4. Division by zero
5. Index out of bounds
6. Infinite loops
7. Resource leaks
8. Race conditions
9. Exception handling issues

For each bug, provide:
- Severity (critical/high/medium/low)
- Category
- Line number
- Description
- Fix suggestion

Output as JSON array:
[
    {{
        "severity": "high",
        "category": "null_pointer",
        "line": 15,
        "description": "...",
        "suggestion": "..."
    }}
]
"""
        
        response = self.client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert code reviewer specializing in bug detection."
                },
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.2
        )
        
        import json
        result = json.loads(response.choices[0].message.content)
        
        bugs = []
        for bug_data in result.get("bugs", []):
            bugs.append(Bug(
                severity=bug_data["severity"],
                category=bug_data["category"],
                line=bug_data["line"],
                description=bug_data["description"],
                suggestion=bug_data["suggestion"]
            ))
        
        return bugs

# Usage
detector = BugDetector()

code = """
def calculate_average(numbers):
    total = 0
    for i in range(len(numbers) + 1):  # Bug: off-by-one
        total += numbers[i]
    return total / len(numbers)  # Bug: division by zero if empty

def get_user(user_id):
    user = database.query(user_id)
    return user.name  # Bug: user might be None

def process_items(items):
    for item in items:
        if item.price > 100:
            item.discount = item.price * 0.1
        return item  # Bug: returns after first iteration
"""

bugs = detector.detect_bugs(code)

for bug in bugs:
    print(f"\\n[{bug.severity.upper()}] Line {bug.line}: {bug.category}")
    print(f"  Issue: {bug.description}")
    print(f"  Fix: {bug.suggestion}")
\`\`\`

## Security Vulnerability Detection

### Security Scanner

\`\`\`python
@dataclass
class SecurityIssue:
    """Represents a security vulnerability."""
    severity: str
    type: str  # "sql_injection", "xss", "path_traversal", etc.
    line: int
    description: str
    cwe: Optional[str]  # CWE identifier
    fix: str

class SecurityScanner:
    """Detect security vulnerabilities."""
    
    def __init__(self):
        self.client = OpenAI()
    
    def scan_for_vulnerabilities(
        self,
        code: str,
        language: str = "python"
    ) -> List[SecurityIssue]:
        """Scan code for security issues."""
        
        prompt = f"""Scan this {language} code for security vulnerabilities:

\`\`\`{language}
{code}
\`\`\`

Check for:
1. SQL injection
2. Cross-site scripting (XSS)
3. Path traversal
4. Command injection
5. Insecure deserialization
6. Hardcoded secrets
7. Weak cryptography
8. Authentication issues
9. Access control problems
10. Input validation issues

For each vulnerability:
- Severity (critical/high/medium/low)
- Type
- Line number
- Description
- CWE number (if applicable)
- Fix recommendation

Output as JSON.
"""
        
        response = self.client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {
                    "role": "system",
                    "content": "You are a security expert specialized in vulnerability detection."
                },
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.2
        )
        
        import json
        result = json.loads(response.choices[0].message.content)
        
        issues = []
        for issue_data in result.get("vulnerabilities", []):
            issues.append(SecurityIssue(
                severity=issue_data["severity"],
                type=issue_data["type"],
                line=issue_data["line"],
                description=issue_data["description"],
                cwe=issue_data.get("cwe"),
                fix=issue_data["fix"]
            ))
        
        return issues

# Usage
scanner = SecurityScanner()

code = """
def login(username, password):
    query = f"SELECT * FROM users WHERE username='{username}' AND password='{password}'"  # SQL injection
    user = database.execute(query)
    return user

def serve_file(filename):
    path = "/uploads/" + filename  # Path traversal
    with open(path) as f:
        return f.read()

API_KEY = "sk-1234567890abcdef"  # Hardcoded secret

def encrypt_data(data):
    return base64.b64encode(data)  # Weak encryption
"""

vulnerabilities = scanner.scan_for_vulnerabilities(code)

for vuln in vulnerabilities:
    print(f"\\n[{vuln.severity.upper()}] {vuln.type}")
    print(f"  Line {vuln.line}: {vuln.description}")
    if vuln.cwe:
        print(f"  CWE: {vuln.cwe}")
    print(f"  Fix: {vuln.fix}")
\`\`\`

## Performance Issue Detection

### Performance Analyzer

\`\`\`python
@dataclass
class PerformanceIssue:
    """Represents a performance problem."""
    severity: str
    type: str  # "n_squared", "repeated_calc", "memory_leak", etc.
    line: int
    description: str
    impact: str
    optimization: str

class PerformanceAnalyzer:
    """Analyze code for performance issues."""
    
    def __init__(self):
        self.client = OpenAI()
    
    def analyze_performance(
        self,
        code: str,
        language: str = "python"
    ) -> List[PerformanceIssue]:
        """Find performance issues in code."""
        
        prompt = f"""Analyze this {language} code for performance issues:

\`\`\`{language}
{code}
\`\`\`

Look for:
1. O(n²) or worse complexity
2. Repeated calculations
3. Unnecessary loops
4. Memory leaks
5. Inefficient data structures
6. Missing caching
7. Redundant operations
8. String concatenation in loops

For each issue:
- Severity
- Type
- Line number
- Description
- Performance impact
- Optimization suggestion

Output as JSON.
"""
        
        response = self.client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert in code performance optimization."
                },
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.2
        )
        
        import json
        result = json.loads(response.choices[0].message.content)
        
        issues = []
        for issue_data in result.get("performance_issues", []):
            issues.append(PerformanceIssue(
                severity=issue_data["severity"],
                type=issue_data["type"],
                line=issue_data["line"],
                description=issue_data["description"],
                impact=issue_data["impact"],
                optimization=issue_data["optimization"]
            ))
        
        return issues

# Usage
perf_analyzer = PerformanceAnalyzer()

code = """
def find_duplicates(items):
    duplicates = []
    for i in range(len(items)):
        for j in range(len(items)):  # O(n²)
            if i != j and items[i] == items[j]:
                duplicates.append(items[i])
    return duplicates

def process_data(data):
    result = ""
    for item in data:
        result += str(item)  # String concatenation in loop
    return result

def calculate_stats(numbers):
    mean = sum(numbers) / len(numbers)
    for num in numbers:
        diff = num - sum(numbers) / len(numbers)  # Repeated calculation
"""

issues = perf_analyzer.analyze_performance(code)

for issue in issues:
    print(f"\\n[{issue.severity}] {issue.type}")
    print(f"  Line {issue.line}: {issue.description}")
    print(f"  Impact: {issue.impact}")
    print(f"  Fix: {issue.optimization}")
\`\`\`

## Code Smell Detection

### Code Smell Detector

\`\`\`python
@dataclass
class CodeSmell:
    """Represents a code smell."""
    type: str  # "long_method", "god_class", "duplicate_code", etc.
    location: str
    description: str
    refactoring: str

class CodeSmellDetector:
    """Detect code smells and anti-patterns."""
    
    def __init__(self):
        self.client = OpenAI()
    
    def detect_smells(
        self,
        code: str,
        language: str = "python"
    ) -> List[CodeSmell]:
        """Detect code smells."""
        
        prompt = f"""Analyze this {language} code for code smells:

\`\`\`{language}
{code}
\`\`\`

Check for:
1. Long methods (>50 lines)
2. God classes (too many responsibilities)
3. Duplicate code
4. Long parameter lists
5. Magic numbers
6. Feature envy
7. Dead code
8. Shotgun surgery
9. Primitive obsession

For each smell:
- Type
- Location
- Description
- Refactoring suggestion

Output as JSON.
"""
        
        response = self.client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert at identifying code smells."
                },
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.2
        )
        
        import json
        result = json.loads(response.choices[0].message.content)
        
        smells = []
        for smell_data in result.get("code_smells", []):
            smells.append(CodeSmell(
                type=smell_data["type"],
                location=smell_data["location"],
                description=smell_data["description"],
                refactoring=smell_data["refactoring"]
            ))
        
        return smells

# Usage
smell_detector = CodeSmellDetector()

code = """
class DataProcessor:
    def process(self, data, format, output_path, validate, transform, filter_func, sort_key, limit, offset, callback):  # Long parameter list
        if format == "json":  # Magic string
            # 50+ lines of processing logic...
            pass
        elif format == "csv":
            # 50+ lines of similar logic (duplicate code)
            pass
        # ... many more responsibilities (god class)
"""

smells = smell_detector.detect_smells(code)

for smell in smells:
    print(f"\\n{smell.type} at {smell.location}")
    print(f"  {smell.description}")
    print(f"  Refactoring: {smell.refactoring}")
\`\`\`

## Comprehensive Code Review

### Complete Code Reviewer

\`\`\`python
@dataclass
class ReviewComment:
    """A code review comment."""
    category: str  # "bug", "security", "performance", "style", "best_practice"
    severity: str
    line: int
    message: str
    suggestion: str

class CodeReviewer:
    """Comprehensive code reviewer."""
    
    def __init__(self):
        self.client = OpenAI()
        self.bug_detector = BugDetector()
        self.security_scanner = SecurityScanner()
        self.perf_analyzer = PerformanceAnalyzer()
        self.smell_detector = CodeSmellDetector()
    
    def review_code(
        self,
        code: str,
        language: str = "python",
        focus_areas: List[str] = None
    ) -> Dict[str, List]:
        """Perform comprehensive code review."""
        
        if focus_areas is None:
            focus_areas = ["bugs", "security", "performance", "style", "best_practices"]
        
        review = {
            "bugs": [],
            "security": [],
            "performance": [],
            "style": [],
            "smells": [],
            "suggestions": []
        }
        
        # Run all checks
        if "bugs" in focus_areas:
            review["bugs"] = self.bug_detector.detect_bugs(code, language)
        
        if "security" in focus_areas:
            review["security"] = self.security_scanner.scan_for_vulnerabilities(code, language)
        
        if "performance" in focus_areas:
            review["performance"] = self.perf_analyzer.analyze_performance(code, language)
        
        if "style" in focus_areas or "best_practices" in focus_areas:
            suggestions = self._get_style_suggestions(code, language)
            review["suggestions"] = suggestions
        
        return review
    
    def _get_style_suggestions(
        self,
        code: str,
        language: str
    ) -> List[ReviewComment]:
        """Get style and best practice suggestions."""
        
        prompt = f"""Review this {language} code for style and best practices:

\`\`\`{language}
{code}
\`\`\`

Check:
1. Naming conventions
2. Code organization
3. Documentation
4. Error handling
5. Type hints (Python)
6. DRY principle
7. SOLID principles
8. Language-specific best practices

Provide suggestions for improvement.
Output as JSON with suggestions array.
"""
        
        response = self.client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert code reviewer."
                },
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.2
        )
        
        import json
        result = json.loads(response.choices[0].message.content)
        
        comments = []
        for suggestion in result.get("suggestions", []):
            comments.append(ReviewComment(
                category="style",
                severity=suggestion.get("severity", "low"),
                line=suggestion.get("line", 0),
                message=suggestion["message"],
                suggestion=suggestion["suggestion"]
            ))
        
        return comments
    
    def generate_review_report(
        self,
        review: Dict[str, List],
        format: str = "markdown"
    ) -> str:
        """Generate formatted review report."""
        
        if format == "markdown":
            return self._generate_markdown_report(review)
        else:
            return self._generate_text_report(review)
    
    def _generate_markdown_report(self, review: Dict) -> str:
        """Generate markdown review report."""
        lines = ["# Code Review Report\\n"]
        
        # Summary
        total_issues = sum(len(issues) for issues in review.values())
        lines.append(f"**Total Issues Found:** {total_issues}\\n")
        
        # Bugs
        if review["bugs"]:
            lines.append("## 🐛 Bugs\\n")
            for bug in review["bugs"]:
                lines.append(f"### [{bug.severity}] Line {bug.line}: {bug.category}")
                lines.append(f"**Issue:** {bug.description}")
                lines.append(f"**Fix:** {bug.suggestion}\\n")
        
        # Security
        if review["security"]:
            lines.append("## 🔒 Security Issues\\n")
            for issue in review["security"]:
                lines.append(f"### [{issue.severity}] Line {issue.line}: {issue.type}")
                lines.append(f"**Issue:** {issue.description}")
                if issue.cwe:
                    lines.append(f"**CWE:** {issue.cwe}")
                lines.append(f"**Fix:** {issue.fix}\\n")
        
        # Performance
        if review["performance"]:
            lines.append("## ⚡ Performance Issues\\n")
            for issue in review["performance"]:
                lines.append(f"### [{issue.severity}] Line {issue.line}: {issue.type}")
                lines.append(f"**Issue:** {issue.description}")
                lines.append(f"**Impact:** {issue.impact}")
                lines.append(f"**Optimization:** {issue.optimization}\\n")
        
        # Suggestions
        if review["suggestions"]:
            lines.append("## 💡 Suggestions\\n")
            for suggestion in review["suggestions"]:
                lines.append(f"### Line {suggestion.line}")
                lines.append(f"**{suggestion.message}**")
                lines.append(f"{suggestion.suggestion}\\n")
        
        return "\\n".join(lines)
    
    def _generate_text_report(self, review: Dict) -> str:
        """Generate plain text report."""
        # Similar to markdown but without formatting
        pass

# Usage
reviewer = CodeReviewer()

code = """
def process_user_data(username, pwd):
    query = f"SELECT * FROM users WHERE name='{username}'"  # SQL injection
    user = db.execute(query)
    
    results = []
    for i in range(len(user.orders)):  # Should use enumerate
        for j in range(len(user.orders)):  # O(n²)
            if user.orders[i].date == user.orders[j].date:
                results.append(user.orders[i])
    
    return results  # No type hints
"""

# Run review
review = reviewer.review_code(code, language="python")

# Generate report
report = reviewer.generate_review_report(review, format="markdown")
print(report)

# Save to file
with open("review_report.md", "w") as f:
    f.write(report)
\`\`\`

## Fix Generation

### Automatic Fix Generator

\`\`\`python
class FixGenerator:
    """Generate fixes for detected issues."""
    
    def __init__(self):
        self.client = OpenAI()
    
    def generate_fix(
        self,
        code: str,
        issue: Union[Bug, SecurityIssue, PerformanceIssue],
        language: str = "python"
    ) -> str:
        """Generate a fix for an issue."""
        
        prompt = f"""Generate a fix for this issue in {language} code:

Original code:
\`\`\`{language}
{code}
\`\`\`

Issue: {issue.description}
Line: {issue.line}
Suggested fix: {issue.suggestion if hasattr(issue, 'suggestion') else issue.fix}

Generate the fixed code using SEARCH/REPLACE format:

<<<<<<< SEARCH
[exact code to replace]
=======
[fixed code]
>>>>>>> REPLACE
"""
        
        response = self.client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert at fixing code issues."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        
        return response.choices[0].message.content

# Usage
fix_gen = FixGenerator()

code = """
def calculate_average(numbers):
    return sum(numbers) / len(numbers)
"""

bug = Bug(
    severity="high",
    category="division_by_zero",
    line=2,
    description="Division by zero when numbers list is empty",
    suggestion="Check if list is empty before calculating average"
)

fix = fix_gen.generate_fix(code, bug)
print(fix)
\`\`\`

## Best Practices Summary

### ✅ DO:
1. **Run comprehensive checks** - bugs, security, performance, style
2. **Prioritize by severity** - fix critical issues first
3. **Provide clear descriptions** for each issue
4. **Suggest concrete fixes**
5. **Generate reports** for documentation
6. **Automate reviews** in CI/CD
7. **Check security vulnerabilities**
8. **Analyze performance**

### ❌ DON'T:
1. **Skip security scanning**
2. **Ignore performance issues**
3. **Give vague feedback**
4. **Miss edge cases**
5. **Forget to check all categories**
6. **Skip fix suggestions**
7. **Ignore code smells**
8. **Only focus on bugs**

## Next Steps

You've mastered code review! Next:
- Interactive code editing
- Code execution and validation
- Building complete code generation systems

Remember: **Thorough Review = Quality Code**
`,
};
