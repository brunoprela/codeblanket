export const codeQualityTools = {
  title: 'Code Quality Tools',
  id: 'code-quality-tools',
  content: `
# Code Quality Tools

## Introduction

**Code quality tools automate the detection of bugs, style violations, security issues, and code smells**—catching problems before they reach production. Professional Python development relies on a suite of tools: **Black** for formatting, **Ruff** for linting, **mypy** for type checking, **bandit** for security, and **pylint** for comprehensive analysis.

These tools enforce consistency, prevent bugs, and improve maintainability. This section covers configuring and using each tool professionally, including CI/CD integration and pre-commit hooks.

---

## The Code Quality Stack

| Tool | Purpose | Speed | Fix Capability |
|------|---------|-------|----------------|
| **Black** | Code formatting | ⚡️⚡️⚡️ | Auto-fix |
| **Ruff** | Fast linting (replaces Flake8, isort, etc.) | ⚡️⚡️⚡️ | Auto-fix (partial) |
| **mypy** | Static type checking | ⚡️⚡️ | Manual fix |
| **pylint** | Comprehensive linting | ⚡️ | Manual fix |
| **bandit** | Security vulnerability scanning | ⚡️⚡️ | Manual fix |
| **radon** | Complexity metrics | ⚡️⚡️⚡️ | Manual refactor |

---

## Black: The Uncompromising Code Formatter

**Black formats Python code automatically**—no configuration needed, no debates about style.

### Installation & Basic Usage

\`\`\`bash
pip install black

# Format file
black myapp/models.py

# Format directory
black myapp/

# Check without modifying (CI mode)
black --check myapp/

# Show what would change
black --diff myapp/
\`\`\`

### Configuration

\`\`\`toml
# pyproject.toml
[tool.black]
line-length = 100  # Default: 88
target-version = ['py311']
include = '\\.pyi?$'
extend-exclude = '''
/(
  # Exclude migrations
  | migrations
  # Exclude generated code
  | generated
  # Exclude vendor
  | vendor
)/
'''
\`\`\`

### Before and After

**Before Black**:
\`\`\`python
def calculate_total (items,tax_rate,discount=0.0,shipping=5.99):
    subtotal=sum([item.price*item.quantity for item in items])
    tax=subtotal*tax_rate
    total=subtotal+tax+shipping-discount
    return total
\`\`\`

**After Black**:
\`\`\`python
def calculate_total (items, tax_rate, discount=0.0, shipping=5.99):
    subtotal = sum([item.price * item.quantity for item in items])
    tax = subtotal * tax_rate
    total = subtotal + tax + shipping - discount
    return total
\`\`\`

### Why Black?

- **No configuration debates**: One style for all
- **Consistent**: Same style across all projects
- **Fast**: Formats 1000 files in seconds
- **Automatic**: No manual formatting
- **Git-friendly**: Minimal diffs

---

## Ruff: Extremely Fast Python Linter

**Ruff is a modern linter** that's 10-100× faster than Flake8, pylint, and isort combined.

### Installation & Usage

\`\`\`bash
pip install ruff

# Lint code
ruff check myapp/

# Auto-fix issues
ruff check --fix myapp/

# Show fixable vs non-fixable
ruff check --statistics myapp/
\`\`\`

### Configuration

\`\`\`toml
# pyproject.toml
[tool.ruff]
# Line length to match Black
line-length = 100

# Python version
target-version = "py311"

# Enable rules
select = [
    "E",   # pycodestyle errors
    "W",   # pycodestyle warnings
    "F",   # Pyflakes
    "I",   # isort (import sorting)
    "B",   # flake8-bugbear
    "C4",  # flake8-comprehensions
    "UP",  # pyupgrade
    "ARG", # flake8-unused-arguments
    "SIM", # flake8-simplify
]

# Ignore specific rules
ignore = [
    "E501",  # Line too long (Black handles this)
    "B008",  # Do not perform function calls in argument defaults
]

# Exclude directories
extend-exclude = [
    "migrations",
    "generated",
    ".venv",
]

# Allow auto-fixing
fixable = ["ALL"]
unfixable = []

[tool.ruff.per-file-ignores]
# Ignore imports in __init__.py
"__init__.py" = ["F401"]
# Ignore assertions in tests
"tests/**/*.py" = ["S101"]

[tool.ruff.isort]
known-first-party = ["myapp"]
\`\`\`

### Common Ruff Rules

**E/W (pycodestyle)**:
- E501: Line too long
- E712: Comparison to True/False
- W291: Trailing whitespace

**F (Pyflakes)**:
- F401: Unused import
- F841: Unused variable
- F821: Undefined name

**B (Bugbear)**:
- B006: Mutable default argument
- B007: Unused loop variable
- B008: Function call in default argument

**Example fixes**:
\`\`\`python
# Before: F401 (unused import)
import os
import sys  # Never used

def main():
    print(os.getcwd())

# After: Ruff removes unused import
import os

def main():
    print(os.getcwd())
\`\`\`

\`\`\`python
# Before: B006 (mutable default)
def add_item (item, items=[]):
    items.append (item)
    return items

# After: Ruff suggests fix
def add_item (item, items=None):
    if items is None:
        items = []
    items.append (item)
    return items
\`\`\`

---

## mypy: Static Type Checker

**mypy analyzes type hints** to catch type errors before runtime.

### Installation & Usage

\`\`\`bash
pip install mypy

# Check types
mypy myapp/

# Strict mode
mypy --strict myapp/

# Generate HTML report
mypy --html-report mypy-report myapp/
\`\`\`

### Configuration

\`\`\`toml
# pyproject.toml
[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true  # Require type hints
disallow_any_generics = true
check_untyped_defs = true
no_implicit_optional = true
warn_redundant_casts = true
warn_unused_ignores = true
warn_unreachable = true
strict_equality = true

# Per-module options
[[tool.mypy.overrides]]
module = "tests.*"
disallow_untyped_defs = false  # Relax for tests

[[tool.mypy.overrides]]
module = "third_party_lib.*"
ignore_missing_imports = true  # Ignore untyped libraries
\`\`\`

### Type Hints Examples

**Basic types**:
\`\`\`python
def greet (name: str) -> str:
    return f"Hello, {name}!"

def add (a: int, b: int) -> int:
    return a + b

def get_users() -> list[str]:
    return ["alice", "bob"]
\`\`\`

**Optional and Union**:
\`\`\`python
from typing import Optional, Union

def find_user (user_id: int) -> Optional[str]:
    """Returns username or None if not found"""
    if user_id == 1:
        return "alice"
    return None

def process (value: Union[int, str]) -> str:
    """Accepts int or str"""
    return str (value)
\`\`\`

**Generic types**:
\`\`\`python
from typing import TypeVar, Generic

T = TypeVar('T')

class Stack(Generic[T]):
    def __init__(self) -> None:
        self.items: list[T] = []
    
    def push (self, item: T) -> None:
        self.items.append (item)
    
    def pop (self) -> T:
        return self.items.pop()

# Usage with type safety
stack: Stack[int] = Stack()
stack.push(1)
stack.push(2)
# stack.push("string")  # mypy error: Expected int
\`\`\`

**Type errors mypy catches**:
\`\`\`python
def calculate (x: int, y: int) -> int:
    return x + y

# mypy catches:
result = calculate("5", 10)  # Error: Expected int, got str
result = calculate(5)  # Error: Missing argument
result: str = calculate(5, 10)  # Error: Incompatible return type
\`\`\`

---

## pylint: Comprehensive Code Analysis

**pylint checks code style, errors, and code smells**.

### Installation & Usage

\`\`\`bash
pip install pylint

# Run pylint
pylint myapp/

# Generate reports
pylint --output-format=json myapp/ > pylint-report.json
\`\`\`

### Configuration

\`\`\`.pylintrc
[MASTER]
# Use multiple cores
jobs=4

# Ignore paths
ignore=migrations,tests,.venv

[MESSAGES CONTROL]
# Disable specific warnings
disable=
    C0111,  # missing-docstring
    C0103,  # invalid-name (Black handles naming)
    R0903,  # too-few-public-methods
    R0913,  # too-many-arguments

[FORMAT]
# Line length (match Black)
max-line-length=100

[DESIGN]
max-args=7
max-attributes=10
max-branches=15
max-locals=20
max-returns=6
max-statements=50

[BASIC]
# Allow short variable names
good-names=i,j,k,x,y,z,id,db,_

[SIMILARITIES]
# Minimum lines for duplicate code detection
min-similarity-lines=4
\`\`\`

### Key pylint Checks

**Code smells**:
- R0914: Too many local variables
- R0915: Too many statements
- R0912: Too many branches
- C0302: Too many lines in module

**Potential bugs**:
- E1101: No member (attribute doesn't exist)
- W0613: Unused argument
- W0612: Unused variable

**Example**:
\`\`\`python
# pylint warns: R0913 (too-many-arguments)
def create_user (username, email, password, first_name, last_name, age, country, phone):
    pass  # Too many parameters

# Better: Use dataclass
from dataclasses import dataclass

@dataclass
class UserData:
    username: str
    email: str
    password: str
    first_name: str
    last_name: str
    age: int
    country: str
    phone: str

def create_user (data: UserData):
    pass  # pylint: compliant
\`\`\`

---

## bandit: Security Vulnerability Scanner

**bandit finds security issues** in Python code.

### Installation & Usage

\`\`\`bash
pip install bandit

# Scan code
bandit -r myapp/

# Generate JSON report
bandit -r myapp/ -f json -o bandit-report.json

# Exclude tests
bandit -r myapp/ --exclude tests/
\`\`\`

### Configuration

\`\`\`yaml
# .bandit
[bandit]
exclude = /test,/tests,/venv,.venv

[bandit.plugins]
# Severity levels: HIGH, MEDIUM, LOW
# Confidence levels: HIGH, MEDIUM, LOW

tests = [
    'B201',  # Flask app debug=True
    'B301',  # pickle
    'B302',  # marshal
    'B303',  # MD5 hash
    'B304',  # insecure cipher
    'B305',  # insecure cipher mode
    'B306',  # insecure temp file
    'B307',  # eval()
    'B308',  # mark_safe()
    'B309',  # HTTPSConnection without cert validation
    'B310',  # URLopen without cert validation
    'B501',  # SSL context without verification
    'B601',  # shell injection
    'B602',  # shell=True
    'B603',  # subprocess without shell injection protection
    'B608',  # SQL injection
]
\`\`\`

### Security Issues bandit Catches

**Critical issues**:
\`\`\`python
# B201: Flask debug mode in production
app.run (debug=True)  # Bandit: HIGH severity

# B608: SQL injection
query = f"SELECT * FROM users WHERE id = {user_id}"
db.execute (query)  # Bandit: HIGH severity

# Better:
query = "SELECT * FROM users WHERE id = ?"
db.execute (query, (user_id,))

# B602: Shell injection
import subprocess
subprocess.call (f"echo {user_input}", shell=True)  # Bandit: HIGH

# Better:
subprocess.call(["echo", user_input])

# B301: Insecure deserialization
import pickle
data = pickle.loads (untrusted_data)  # Bandit: MEDIUM

# Better: Use JSON
import json
data = json.loads (untrusted_data)
\`\`\`

---

## radon: Complexity Metrics

**radon measures code complexity**—cyclomatic complexity, maintainability index, raw metrics.

### Installation & Usage

\`\`\`bash
pip install radon

# Cyclomatic complexity
radon cc myapp/ -a

# Maintainability index
radon mi myapp/

# Raw metrics (LOC, SLOC, comments)
radon raw myapp/

# Halstead metrics
radon hal myapp/
\`\`\`

### Cyclomatic Complexity

**Cyclomatic complexity** measures number of paths through code:

\`\`\`python
def simple_function (x):
    return x * 2
# Complexity: 1 (one path)

def with_condition (x):
    if x > 0:
        return x * 2
    return 0
# Complexity: 2 (two paths)

def complex_function (x, y, z):
    if x > 0:
        if y > 0:
            if z > 0:
                return x + y + z
            return x + y
        return x
    return 0
# Complexity: 4 (four paths)
\`\`\`

**Complexity thresholds**:
- **A (1-5)**: Simple, easy to test
- **B (6-10)**: Moderate, acceptable
- **C (11-20)**: Complex, needs refactoring
- **D (21-30)**: Very complex, hard to maintain
- **F (31+)**: Extremely complex, refactor immediately

---

## Integrating Tools into Workflow

### pyproject.toml (Centralized Configuration)

\`\`\`toml
[project]
name = "myapp"
version = "1.0.0"

[tool.black]
line-length = 100
target-version = ['py311']

[tool.ruff]
line-length = 100
select = ["E", "F", "I", "B", "C4", "UP"]

[tool.mypy]
python_version = "3.11"
warn_return_any = true
disallow_untyped_defs = true

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = "test_*.py"
addopts = "--cov=myapp --cov-report=html"
\`\`\`

### Makefile for Quick Commands

\`\`\`makefile
.PHONY: format lint type-check security test all

format:
\tblack myapp/ tests/
\truff check --fix myapp/ tests/

lint:
\truff check myapp/ tests/
\tpylint myapp/

type-check:
\tmypy myapp/

security:
\tbandit -r myapp/

complexity:
\tradon cc myapp/ -a -nb

test:
\tpytest tests/ --cov=myapp --cov-report=html

all: format lint type-check security test
\`\`\`

**Usage**:
\`\`\`bash
make format      # Format code
make lint        # Run linters
make type-check  # Check types
make all         # Run everything
\`\`\`

---

## CI/CD Integration

### GitHub Actions

\`\`\`yaml
name: Code Quality

on: [push, pull_request]

jobs:
  quality:
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install black ruff mypy pylint bandit pytest pytest-cov
      
      - name: Black (format check)
        run: black --check myapp/ tests/
      
      - name: Ruff (linting)
        run: ruff check myapp/ tests/
      
      - name: mypy (type check)
        run: mypy myapp/
      
      - name: pylint
        run: pylint myapp/ --fail-under=8.0
      
      - name: bandit (security)
        run: bandit -r myapp/
      
      - name: Tests with coverage
        run: |
          pytest --cov=myapp --cov-report=xml --cov-fail-under=80
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          files: ./coverage.xml
\`\`\`

---

## Best Practices

1. **Black for formatting**: No debates, just run it
2. **Ruff for linting**: Fast, catches common bugs
3. **mypy for types**: Add types gradually, enforce in new code
4. **bandit for security**: Run in CI, fail on HIGH severity
5. **radon for complexity**: Monitor, refactor F-rated functions
6. **Centralize config**: Use pyproject.toml
7. **Pre-commit hooks**: Catch issues before commit
8. **CI enforcement**: Block merges on failures

---

## Tool Recommendations by Project Type

| Project Type | Essential Tools | Optional Tools |
|--------------|----------------|----------------|
| **Web API** | Black, Ruff, mypy, pytest | bandit, pylint |
| **Data Science** | Black, Ruff, pytest | mypy, radon |
| **Library** | Black, Ruff, mypy, pytest, bandit | pylint, radon |
| **CLI Tool** | Black, Ruff, mypy, bandit | pylint |
| **Microservice** | Black, Ruff, mypy, pytest, bandit | pylint, radon |

---

## Summary

**Essential code quality tools**:
- **Black**: Auto-format (no config needed)
- **Ruff**: Fast linting (10-100× faster than alternatives)
- **mypy**: Type checking (catch bugs at development time)
- **bandit**: Security scanning (prevent vulnerabilities)
- **pylint**: Comprehensive analysis (code smells, complexity)
- **radon**: Complexity metrics (identify refactoring candidates)

Use these tools together for **professional, maintainable, secure code**.
`,
};
